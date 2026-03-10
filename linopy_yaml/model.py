"""Model subclass with from_yaml() and extend() methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr
import yaml

import linopy

from linopy_yaml.builder import build_model
from linopy_yaml.loader import build_master_coords, load_parameters
from linopy_yaml.schema import MathSchema


class Model(linopy.Model):
    """A linopy Model that can be built from a YAML math definition.

    Inherits all linopy.Model behaviour. Adds ``from_yaml()`` for
    constructing models from YAML files, ``extend()`` for composing
    models from multiple YAML files, and ``.math`` / ``.dataset``
    properties for introspection.
    """

    _math: MathSchema | None
    _dataset: xr.Dataset | None
    _master_coords: dict[str, pd.Index] | None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._math = None
        self._dataset = None
        self._master_coords = None

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        data: dict[str, Any] | None = None,
        coords: dict[str, Any] | None = None,
    ) -> Model:
        """Build a Model from a YAML math definition and user-provided data.

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
        Model
            A fully built model ready to solve.

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

        model = cls()
        model._math = schema
        model._dataset = dataset
        model._master_coords = master_coords

        build_model(model, schema, dataset, master_coords)

        return model

    def extend(
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
        if self._math is None or self._dataset is None or self._master_coords is None:
            msg = "Cannot extend a model that was not built from YAML."
            raise RuntimeError(msg)

        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        if raw is None:
            raw = {}

        schema = MathSchema.model_validate(raw)

        # Merge master coords: existing + any new dimensions
        merged_coords = dict(self._master_coords)
        new_coords = build_master_coords(schema, None)
        for dim, idx in new_coords.items():
            if dim not in merged_coords:
                merged_coords[dim] = idx

        # Load new parameters and merge with existing dataset
        new_dataset = load_parameters(schema, data, merged_coords)
        merged_dataset = self._dataset.merge(new_dataset, compat="override")

        # Build new components
        build_model(self, schema, merged_dataset, merged_coords)

        # Update stored state
        self._master_coords = merged_coords
        self._dataset = merged_dataset

    @property
    def math(self) -> MathSchema:
        """The parsed YAML math definition."""
        if self._math is None:
            msg = "No math definition loaded. Use Model.from_yaml() first."
            raise AttributeError(msg)
        return self._math

    @property
    def dataset(self) -> xr.Dataset:
        """The loaded parameter dataset."""
        if self._dataset is None:
            msg = "No dataset loaded. Use Model.from_yaml() first."
            raise AttributeError(msg)
        return self._dataset
