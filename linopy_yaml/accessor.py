"""YAML accessor for linopy.Model — monkey-patched as model.yaml."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

import linopy

from linopy_yaml.builder import build_model
from linopy_yaml.loader import build_master_coords, load_parameters
from linopy_yaml.schema import MathSchema

# Store accessors keyed by model id, with weak references to models
# so accessors are cleaned up when models are garbage collected.
_ACCESSORS: dict[int, YamlAccessor] = {}
_WEAK_REFS: dict[int, weakref.ref] = {}


def _cleanup(ref: weakref.ref) -> None:
    """Remove accessor when the model is garbage collected."""
    for model_id, wr in list(_WEAK_REFS.items()):
        if wr is ref:
            _ACCESSORS.pop(model_id, None)
            _WEAK_REFS.pop(model_id, None)
            break


def _register_accessor(model: linopy.Model, accessor: YamlAccessor) -> None:
    """Associate an accessor with a model instance."""
    model_id = id(model)
    _ACCESSORS[model_id] = accessor
    _WEAK_REFS[model_id] = weakref.ref(model, _cleanup)


def _get_accessor(model: linopy.Model) -> YamlAccessor | None:
    """Look up the accessor for a model instance."""
    return _ACCESSORS.get(id(model))


class YamlAccessor:
    """Accessor attached to a linopy.Model instance as ``model.yaml``.

    Provides access to the parsed YAML schema, loaded parameter dataset,
    master coordinates, and the ability to add more math from another YAML.
    """

    def __init__(
        self,
        model: linopy.Model,
        schema: MathSchema,
        dataset: xr.Dataset,
        coords: dict[str, pd.Index],
    ) -> None:
        self._model = model
        self._schema = schema
        self._dataset = dataset
        self._coords = coords

    @property
    def schema(self) -> MathSchema:
        """The parsed YAML math definition."""
        return self._schema

    @property
    def dataset(self) -> xr.Dataset:
        """The loaded parameter dataset."""
        return self._dataset

    @property
    def coords(self) -> dict[str, pd.Index]:
        """The master coordinates for all declared dimensions."""
        return dict(self._coords)

    def add(
        self,
        path: str | Path,
        *,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Add variables, constraints, and/or objectives from another YAML.

        The second YAML may reference dimensions and parameters already
        loaded. New parameters can be provided via ``data=``.

        Parameters
        ----------
        path : str or Path
            Path to the additional YAML file.
        data : dict or None
            Additional parameter data for new parameters in this YAML.
        """
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if raw is None:
            raw = {}

        schema = MathSchema.model_validate(raw)

        # Merge master coords: existing + any new dimensions
        merged_coords = dict(self._coords)
        new_coords = build_master_coords(schema, None)
        for dim, idx in new_coords.items():
            if dim not in merged_coords:
                merged_coords[dim] = idx

        # Load new parameters and merge with existing dataset
        new_dataset = load_parameters(schema, data, merged_coords)
        merged_dataset = self._dataset.merge(new_dataset, compat="override")

        # Build new components
        build_model(self._model, schema, merged_dataset, merged_coords)

        # Update stored state
        self._coords = merged_coords
        self._dataset = merged_dataset


class _YamlDescriptor:
    """Descriptor providing .yaml on instances, raising clearly on non-YAML models."""

    def __get__(self, obj: linopy.Model | None, objtype: type | None = None) -> YamlAccessor:
        if obj is None:
            msg = (
                "Access .yaml on a model instance, not the class.\n"
                "Use Model.from_yaml('model.yaml', data={...}) to create one."
            )
            raise AttributeError(msg)
        accessor = _get_accessor(obj)
        if accessor is None:
            msg = (
                "This model was not built from YAML.\n"
                "Use Model.from_yaml('model.yaml', data={...}) to "
                "create a YAML-backed model."
            )
            raise AttributeError(msg)
        return accessor


def _from_yaml(
    path: str | Path,
    *,
    data: dict[str, Any] | None = None,
    coords: dict[str, Any] | None = None,
) -> linopy.Model:
    """Build a linopy.Model from a YAML math definition.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.
    data : dict or None
        Parameter data. Keys are parameter names declared in the YAML.
    coords : dict or None
        Dimension coordinate values. Overrides values declared in YAML.

    Returns
    -------
    linopy.Model
        A fully built model ready to solve. Access YAML metadata via
        ``model.yaml.schema``, ``model.yaml.dataset``, etc.

    Raises
    ------
    ValueError
        For any validation failure (missing dimensions, parameters, etc.).
    pydantic.ValidationError
        If the YAML structure is invalid.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}

    schema = MathSchema.model_validate(raw)

    master_coords = build_master_coords(schema, coords)
    dataset = load_parameters(schema, data, master_coords)

    model = linopy.Model()
    accessor = YamlAccessor(model, schema, dataset, master_coords)
    _register_accessor(model, accessor)

    build_model(model, schema, dataset, master_coords)

    return model


def _install() -> None:
    """Monkey-patch linopy.Model with .yaml descriptor and .from_yaml classmethod."""
    linopy.Model.yaml = _YamlDescriptor()  # type: ignore[attr-defined]
    linopy.Model.from_yaml = staticmethod(_from_yaml)  # type: ignore[attr-defined]
