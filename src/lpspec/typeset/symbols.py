"""Which symbol each declared name prints as — and the sidecar that overrides it.

Derivation aims at *unambiguous*, not beautiful: it runs with no setup, so it
has to be right rather than elegant. :class:`SymbolTable` is where a reader
makes it conventional, in a file of its own — presentation is not language, so
it never becomes keys on ``Model``.

This module decides *which* symbol a name gets; a
:class:`~lpspec.typeset.format.Format` decides how it is written.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lpspec.errors import SchemaError, did_you_mean
from lpspec.language._yaml import read_yaml
from lpspec.typeset.format import SYMBOL_NAMES

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


# ---------------------------------------------------------------------------
# the table's notation: neutral, spelled by each format (#321)
# ---------------------------------------------------------------------------

#: What each entry may say. A format-neutral vocabulary rather than LaTeX,
#: because a symbol has to be spelled per format and a table written in one
#: format's syntax silently breaks every other — exactly as ``operators``
#: already works, where the walk names the intent and each format spells it.
_NOTATION = (
    'the notation is a single letter, a letter name (ell, alpha, …, Omega), '
    'cal(X), bar(x), under(x), up(word), it(word) or sup(base, qualifier)'
)

#: Arity per function; the speller below is the one place each is interpreted.
_FUNCTIONS = {'cal': 1, 'bar': 1, 'under': 1, 'up': 1, 'it': 1, 'sup': 2}

_TOKEN = re.compile(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*')

_MATH_SPAN = re.compile(r'\$([^$]+)\$')

#: A parsed entry: a word, or a function applied to parsed arguments.
_Node = str | tuple[str, list['_Node']]


def _parse(entry: str) -> _Node:
    """*entry* as a word or a ``(function, arguments)`` tree.

    Raises:
        SchemaError: The entry does not fit the notation.
    """
    node, i = _parse_expr(entry, 0)
    if entry[i:].strip():
        raise SchemaError(f"symbol notation: trailing '{entry[i:].strip()}' after the expression; {_NOTATION}")
    return node


def _parse_expr(entry: str, i: int) -> tuple[_Node, int]:
    m = _TOKEN.match(entry, i)
    if not m:
        raise SchemaError(f"symbol notation: expected a name at '{entry[i:]}' in '{entry}'; {_NOTATION}")
    word, i = m.group(1), m.end()
    if not entry.startswith('(', i):
        return word, i
    args: list[_Node] = []
    while True:
        node, i = _parse_expr(entry, i + 1)
        args.append(node)
        if not entry.startswith((',', ')'), i):
            raise SchemaError(f"symbol notation: expected ',' or ')' at '{entry[i:]}' in '{entry}'")
        if entry.startswith(')', i):
            return (word, args), i + 1


def _spell_node(node: _Node, fmt: Format) -> str:
    if isinstance(node, str):
        if node in SYMBOL_NAMES:
            return fmt.symbol(node)
        if len(node) == 1:
            return node
        raise SchemaError(
            f"symbol notation: '{node}' is neither a letter nor a letter name — "
            f'write up({node}) or it({node}) to set the word as one symbol; {_NOTATION}'
        )
    function, args = node
    if function not in _FUNCTIONS:
        raise SchemaError(f"symbol notation: unknown function '{function}'; {_NOTATION}")
    if len(args) != _FUNCTIONS[function]:
        raise SchemaError(f'symbol notation: {function}() takes {_FUNCTIONS[function]} argument(s), got {len(args)}')
    if function in ('up', 'it'):
        if not isinstance(args[0], str):
            raise SchemaError(f'symbol notation: {function}() takes a plain word')
        return fmt.upright(args[0]) if function == 'up' else fmt.italic(args[0])
    if function == 'sup':
        if not isinstance(args[1], str):
            raise SchemaError('symbol notation: the qualifier in sup(base, qualifier) is a plain word')
        return fmt.superscript(_spell_node(args[0], fmt), fmt.upright(args[1]))
    inner = _spell_node(args[0], fmt)
    return {'cal': fmt.script, 'bar': fmt.bar, 'under': fmt.underline}[function](inner)


def _spell(entry: str, fmt: Format, where: str) -> str:
    """One table entry through *fmt*, or a :class:`SchemaError` naming *where*."""
    try:
        return _spell_node(_parse(entry), fmt)
    except SchemaError as error:
        raise SchemaError(f"symbol table: '{where}': {error}") from None


def _spell_prose(text: str, fmt: Format, where: str) -> str:
    """A description with each ``$…$`` span spelled through *fmt*."""

    def span(m: re.Match[str]) -> str:
        return fmt.math(_spell(m.group(1), fmt, where))

    return _MATH_SPAN.sub(span, text)


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
                _spell(table.names[name], fmt, name)
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
            override = table.indices.get(dim)
            letter = override or _first_free(_index_candidates(dim), taken_index)
            taken_index.add(letter)
            if override:
                self.index[dim] = _spell(override, fmt, f'{dim}: index')
            else:
                self.index[dim] = letter if len(letter) <= 1 else fmt.upright(letter)
            given = table.sets.get(dim)
            upper = _first_free(_set_candidates(dim, letter), taken_set)
            taken_set.add(upper)
            self.set[dim] = _spell(given, fmt, f'{dim}: set') if given else fmt.script(upper)

        self.description: dict[str, str] = {
            name: _spell_prose(text, fmt, name) for name, text in table.descriptions.items()
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


# ---------------------------------------------------------------------------
# the symbol table (a sidecar file, not the model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolTable:
    r"""How a *reader* wants the model to print — kept out of the model.

    Presentation is not language: nothing here changes what the file means, no
    lane reads it, and a model with no table still renders.

    Entries speak the neutral notation of :func:`_parse` — ``cal(T)``,
    ``ell``, ``bar(p)``, ``sup(c, marg)`` — never one format's syntax, so one
    table prints through every format (#321). Inside a description, math goes
    in ``$…$`` spans of the same notation.

    Deliberately strict — an unrecognised name is an error naming the near
    miss, the failure mode of a silent typo being a symbol that never applies
    and a reader who never finds out::

        dimensions:
          snapshot: {index: t, set: cal(T)}
          plant:    {index: n}
        names:
          marginal_cost: sup(c, marg)
        descriptions:
          snapshot: hourly, over one year
    """

    indices: dict[str, str] = field(default_factory=dict)
    sets: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, source: str | Path | Mapping[str, Any]) -> SymbolTable:
        raw = dict(source) if isinstance(source, Mapping) else read_yaml(Path(source))
        unknown = set(raw) - {'dimensions', 'names', 'descriptions'}
        if unknown:
            msg = (
                f'symbol table: unknown section(s) {sorted(unknown)}. Valid sections: dimensions, names, descriptions.'
            )
            raise SchemaError(msg)

        indices: dict[str, str] = {}
        sets: dict[str, str] = {}
        for dim, spec in (raw.get('dimensions') or {}).items():
            if not isinstance(spec, Mapping):
                msg = f"symbol table: dimension '{dim}' must be a mapping like {{index: t, set: cal(T)}}"
                raise SchemaError(msg)
            extra = set(spec) - {'index', 'set'}
            if extra:
                msg = f"symbol table: dimension '{dim}' has unknown key(s) {sorted(extra)}. Valid keys: index, set."
                raise SchemaError(msg)
            if 'index' in spec:
                indices[dim] = str(spec['index'])
            if 'set' in spec:
                sets[dim] = str(spec['set'])

        return cls(
            indices=indices,
            sets=sets,
            names={k: str(v) for k, v in (raw.get('names') or {}).items()},
            descriptions={k: str(v) for k, v in (raw.get('descriptions') or {}).items()},
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
