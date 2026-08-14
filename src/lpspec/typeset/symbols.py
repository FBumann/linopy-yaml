"""Which symbol each declared name prints as — and the sidecar that overrides it.

Derivation aims at *unambiguous*, not beautiful: it runs with no setup, so it
has to be right rather than elegant. :class:`SymbolTable` is where a reader
makes it conventional, in a file of its own — presentation is not language, so
it never becomes keys on ``Model``.

This module decides *which* symbol a name gets; a
:class:`~lpspec.typeset.format.Format` decides how it is written.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lpspec.errors import SchemaError, did_you_mean
from lpspec.language._yaml import read_yaml

if TYPE_CHECKING:
    from lpspec.language.model import Model
    from lpspec.typeset.format import Format

__all__ = ['SymbolTable', 'Symbols']

#: Dimensions whose conventional index letter is not their own first letter.
#: Small on purpose — a lookup table of everybody's naming habits is a
#: maintenance sink; anything unlisted falls back to its own initial.
_INDEX_ALIASES = {'snapshot': 't', 'snapshots': 't', 'time': 't', 'timestep': 't', 'timesteps': 't'}


def _word(name: str, fmt: Format) -> str:
    """One name as one symbol: a letter stays a letter, a word is set italic."""
    return name if len(name) == 1 else fmt.italic(name)


def _derive_name_symbol(name: str, declared: frozenset[str], fmt: Format) -> str:
    r"""``p`` → ``p``; ``load`` → ``\mathit{load}``; ``p_max`` → ``p^{\mathrm{max}}``.

    An underscore is a **qualifier** only when what precedes it is a symbol in
    its own right — a single letter (``p_max``) or another declared name
    (``soc_max``). Everywhere else it is word separation, where splitting
    produces nonsense: ``marginal_cost`` is not *marginal* raised to *cost*.
    The fallback therefore prints the name as written, underscore and all,
    which is plain rather than beautiful; ``--symbols`` is what makes it
    pretty. A qualifier lands in the superscript, the subscript slot being
    spoken for by the dimensions.
    """
    head, _, tail = name.partition('_')
    if tail and (len(head) == 1 or head in declared):
        return fmt.superscript(_word(head, fmt), fmt.upright(tail.replace('_', ',')))
    return _word(name, fmt)


class Symbols:
    r"""How every declared name prints: overrides first, derivation for the rest.

    Assignment order is load-bearing. Name symbols settle *before* dimension
    indices, so an index can be kept off a letter a variable owns — derived
    independently, a model with a dimension ``plant`` and a variable ``p``
    renders ``p_{t,p}`` and no reader can tell which ``p`` is which. Only
    single-letter name symbols are kept off the index letters, a
    ``\mathit{load}`` never colliding with a ``t``.
    """

    def __init__(self, schema: Model, fmt: Format, table: SymbolTable | None = None) -> None:
        table = table or SymbolTable()
        declared = frozenset({*schema.dimensions, *schema.parameters, *schema.variables})

        self.name: dict[str, str] = {
            name: (
                _spelling(table.names[name], fmt, f"'{name}' under names")
                if name in table.names
                else _derive_name_symbol(name, declared, fmt)
            )
            for name in (*schema.parameters, *schema.variables)
        }
        spoken_for = {s for s in self.name.values() if len(s) == 1}

        self.index: dict[str, str] = {}
        self.set: dict[str, str] = {}
        taken_index, taken_set = set(spoken_for), set()
        for dim in schema.dimensions:
            where = f"'{dim}' under dimensions"
            override = _spelling(table.indices[dim], fmt, where) if dim in table.indices else None
            letter = override or _first_free(_index_candidates(dim), taken_index)
            taken_index.add(letter)
            self.index[dim] = letter if len(letter) <= 1 or override else fmt.upright(letter)
            given = _spelling(table.sets[dim], fmt, where) if dim in table.sets else None
            upper = _first_free(_set_candidates(dim, letter), taken_set)
            taken_set.add(upper)
            self.set[dim] = given or fmt.script(upper)

        self.description: dict[str, str] = {
            name: _spelling(entry, fmt, f"'{name}' under descriptions") for name, entry in table.descriptions.items()
        }


def _index_candidates(dim: str) -> list[str]:
    alias = _INDEX_ALIASES.get(dim)
    letters = [c for c in dim.lower() if c.isalpha()]
    return [*([alias] if alias else []), *letters, *string.ascii_lowercase, dim]


def _set_candidates(dim: str, index_letter: str) -> list[str]:
    first = next((c for c in index_letter if c.isalpha()), '')
    letters = [c.upper() for c in dim if c.isalpha()]
    return [*([first.upper()] if first else []), *letters, *string.ascii_uppercase]


def _first_free(candidates: list[str], taken: set[str]) -> str:
    return next((c for c in candidates if c not in taken), candidates[-1])


def _spelling(entry: str | Mapping[str, str], fmt: Format, where: str) -> str:
    """*entry* in *fmt*'s notation, verbatim.

    The refusals are the whole point: a per-format entry without *fmt*'s
    notation, and — the bug this replaced — a bare LaTeX string reaching
    Typst, which used to pass through silently and fail three tools later.
    The backslash test is a heuristic on *bare* strings only; a spelling
    under an explicit ``typst:`` key is taken on the author's word.

    Raises:
        SchemaError: Naming the entry and the notation it is missing.
    """
    if isinstance(entry, Mapping):
        spelling = entry.get(fmt.notation)
        if spelling is None:
            msg = (
                f'symbol table: {where}: spells {sorted(entry)} but not {fmt.notation}; '
                f"add '{fmt.notation}: …' to render this format."
            )
            raise SchemaError(msg)
        return spelling
    if fmt.notation == 'typst' and '\\' in entry:
        msg = (
            f'symbol table: {where}: {entry!r} contains a backslash, which is LaTeX rather than Typst. '
            f'Spell it per format: {{latex: {entry!r}, typst: …}}.'
        )
        raise SchemaError(msg)
    return entry


# ---------------------------------------------------------------------------
# the symbol table (a sidecar file, not the model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolTable:
    r"""How a *reader* wants the model to print — kept out of the model.

    Presentation is not language: nothing here changes what the file means, no
    lane reads it, and a model with no table still renders.

    An entry is a *spelling*, taken verbatim — nothing parses LaTeX or Typst.
    A bare string is used by every format; a mapping keyed ``latex:`` /
    ``typst:`` gives each format its own, and a format asked to render an
    entry that does not carry its notation refuses, naming the entry::

        dimensions:
          snapshot: {index: t, set: {latex: "\\mathcal{T}", typst: cal(T)}}
          plant:    {index: n}
        names:
          marginal_cost: {latex: "c^{\\mathrm{marg}}", typst: 'c^(upright("marg"))'}
        descriptions:
          snapshot: hourly, over one year

    Deliberately strict — an unrecognised name is an error naming the near
    miss, the failure mode of a silent typo being a symbol that never applies
    and a reader who never finds out.
    """

    indices: dict[str, str | dict[str, str]] = field(default_factory=dict)
    sets: dict[str, str | dict[str, str]] = field(default_factory=dict)
    names: dict[str, str | dict[str, str]] = field(default_factory=dict)
    descriptions: dict[str, str | dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, source: str | Path | Mapping[str, Any]) -> SymbolTable:
        raw = dict(source) if isinstance(source, Mapping) else read_yaml(Path(source))
        unknown = set(raw) - {'dimensions', 'names', 'descriptions'}
        if unknown:
            msg = (
                f'symbol table: unknown section(s) {sorted(unknown)}. Valid sections: dimensions, names, descriptions.'
            )
            raise SchemaError(msg)

        indices: dict[str, str | dict[str, str]] = {}
        sets: dict[str, str | dict[str, str]] = {}
        for dim, spec in (raw.get('dimensions') or {}).items():
            if not isinstance(spec, Mapping):
                msg = f"symbol table: dimension '{dim}' must be a mapping like {{index: t, set: '\\\\mathcal{{T}}'}}"
                raise SchemaError(msg)
            extra = set(spec) - {'index', 'set'}
            if extra:
                msg = f"symbol table: dimension '{dim}' has unknown key(s) {sorted(extra)}. Valid keys: index, set."
                raise SchemaError(msg)
            if 'index' in spec:
                indices[dim] = _entry(spec['index'], f"'{dim}' under dimensions")
            if 'set' in spec:
                sets[dim] = _entry(spec['set'], f"'{dim}' under dimensions")

        return cls(
            indices=indices,
            sets=sets,
            names={k: _entry(v, f"'{k}' under names") for k, v in (raw.get('names') or {}).items()},
            descriptions={
                k: _entry(v, f"'{k}' under descriptions") for k, v in (raw.get('descriptions') or {}).items()
            },
        )

    def checked_against(self, schema: Model) -> SymbolTable:
        """Reject entries naming nothing in *schema*, with the near miss."""
        dims = set(schema.dimensions)
        everything = dims | set(schema.parameters) | set(schema.variables)
        errors = [
            *(_unknown_entry(d, 'dimensions', dims) for d in {*self.indices, *self.sets} - dims),
            *(_unknown_entry(n, 'names', everything - dims) for n in set(self.names) - everything),
            *(_unknown_entry(n, 'descriptions', everything) for n in set(self.descriptions) - everything),
        ]
        if errors:
            raise SchemaError('\n'.join(sorted(errors)))
        return self


def _unknown_entry(name: str, section: str, known: set[str]) -> str:
    return f"symbol table: '{name}' under {section}: is not declared by the model. {did_you_mean(name, known)}"


#: The notations a per-format entry may be keyed by — the ``Format.notation``
#: values, restated here so a table loads (and errors) without any format.
_NOTATIONS = ('latex', 'typst')


def _entry(value: Any, where: str) -> str | dict[str, str]:
    """One table value, shape-checked at load: a bare spelling, or one per notation.

    Which notation a mapping is *missing* is checked in :func:`_spelling`
    instead, per format at render time — a latex-only entry is fine until
    Typst is asked for.

    Raises:
        SchemaError: A mapping keyed by anything but :data:`_NOTATIONS`.
    """
    if isinstance(value, Mapping):
        unknown = set(value) - set(_NOTATIONS)
        if unknown:
            msg = (
                f'symbol table: {where}: unknown notation(s) {sorted(unknown)}. '
                f'An entry is one string every format uses, or spellings keyed latex/typst.'
            )
            raise SchemaError(msg)
        if not value:
            msg = f'symbol table: {where}: an empty mapping spells nothing; give a string or latex/typst keys.'
            raise SchemaError(msg)
        return {k: str(v) for k, v in value.items()}
    return str(value)
