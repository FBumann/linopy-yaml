"""The ``lp_file`` sink, rendered inside duckdb — #399's experiment.

The experiment #399 names: build the LP text in SQL and ``COPY`` it out, so the
model never crosses into polars on the way to a file. What the ordinary writer
streams out of frames this streams out of tables, and the bytes must be
identical — `tests/test_duck_lp_text.py` compares them over the whole ports
corpus.

In `bench/` rather than under `writers/` because the answer it produced was
no: it is heavier and slower than the writer it would replace, and the one
shape that would bound it is the one duckdb will not stream. Measurement code
for a question already answered, kept so the answer can be re-checked.

**The float rendering is the whole risk.** LP text is mostly numbers, and the
ordinary writer spells them with polars' float cast. duckdb's differs in two
places and nowhere else, measured over 141,840 doubles including 20,000 drawn
from the one decade they disagree about: it pads a single-digit negative
exponent (``1e-07`` for ``1e-7``), and it switches to scientific notation one
decade earlier, at ``1e-5``. :func:`_number` repairs both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import duckdb

#: A double as the ordinary writer spells it. The two repairs are duckdb's
#: only departures from polars' cast; every other value renders identically.
_NUMBER = r"""
    CASE
        WHEN {v} = 0 OR NOT isfinite({v}) THEN CAST({v} AS VARCHAR)
        WHEN abs({v}) >= 1e-5 AND abs({v}) < 1e-4 THEN
            CASE WHEN {v} < 0 THEN '-0.0000' ELSE '0.0000' END ||
            replace(split_part(CAST(abs({v}) AS VARCHAR), 'e', 1), '.', '')
        ELSE regexp_replace(CAST({v} AS VARCHAR), 'e-0(\d)$', 'e-\1')
    END
"""


def _number(value: str) -> str:
    """*value* as LP text, matching polars' float cast."""
    return _NUMBER.format(v=value)


def _term(coeff: str, col: str) -> str:
    """One ``+1.5 x7`` term, sign always explicit.

    Zero is spelled rather than rendered for the reason the ordinary writer
    spells it: ``-0.0`` takes the ``+`` arm and would give ``+-0.0``, which no
    LP parser accepts.
    """
    sign = f"CASE WHEN {coeff} >= 0 THEN '+' ELSE '' END"
    magnitude = f"CASE WHEN {coeff} = 0 THEN '0.0' ELSE {_number(coeff)} END"
    return f"{sign} || {magnitude} || ' x' || CAST({col} AS BIGINT)::VARCHAR"


def _bound(value: str, infinite: str) -> str:
    return f"CASE WHEN isinf({value}) THEN '{infinite}' ELSE {_number(value)} END"


def _line(section: int, order: str, text: str) -> str:
    """One SELECT contributing lines to the file, keyed for the global order."""
    return f'SELECT {section} AS s, {order}::BIGINT AS k, 0::BIGINT AS j, {text} AS line'


def write(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    *,
    col_tables: list[str],
    obj_tables: list[str],
    row_tables: list[str],
    matrix_tables: list[str],
    vtype_runs: list[tuple[str, int]],
    sense: str,
    constant: float,
) -> None:
    """Render the model at *con* as LP text at *path*, in one statement.

    Every section is a branch of one union ordered by ``(section, key)``, so
    the file is one stream duckdb sorts and writes. That sort is the point of
    the experiment: it is the only part of a write whose size follows the
    model, so whether the whole write stays inside ``memory_limit`` is whether
    duckdb can spill it.
    """
    cols = _union(col_tables, 'SELECT lb, ub FROM {t}')
    objective = _union(obj_tables, 'SELECT col, coeff FROM {t}')
    rows = _union(row_tables, 'SELECT row, sense, rhs FROM {t}')
    matrix = _union(matrix_tables, 'SELECT row, col, coeff FROM {t}')

    numbered = f'SELECT *, (row_number() OVER () - 1) AS col FROM ({cols})'
    vtype = _vtype_case(vtype_runs)

    parts = [
        _line(0, '0', f"'{'min' if sense == 'min' else 'max'}'"),
        _line(0, '1', "''"),
        _line(0, '2', "'obj:'"),
    ]
    if constant:
        parts.append(_line(0, '3', f"printf('%+.17g', {constant!r}::DOUBLE)"))
    parts += [
        f'SELECT 1, col::BIGINT, 0::BIGINT, {_term("coeff", "col")} FROM ({objective})',
        _line(2, '0', "''"),
        _line(2, '1', "'s.t.'"),
        _line(2, '2', "''"),
        # a constraint's lines are keyed (row, slot): header, placeholder, one
        # per term at its column, sense. Two integers rather than the ordinary
        # writer's single product, which it has to bound per chunk to keep from
        # overflowing; ordering on the pair cannot overflow at any model size.
        f"SELECT 3, row::BIGINT, 0::BIGINT, 'c' || CAST(row AS BIGINT)::VARCHAR || ':' FROM ({rows})",
        f"""SELECT 3, r.row::BIGINT, 1::BIGINT, '+0 x0' FROM ({rows}) AS r
            WHERE NOT EXISTS (SELECT 1 FROM ({matrix}) AS m WHERE m.row = r.row)""",
        f'SELECT 3, row::BIGINT, col::BIGINT + 2, {_term("coeff", "col")} FROM ({matrix})',
        f"""SELECT 3, row::BIGINT, 9223372036854775807::BIGINT,
                   replace(sense, '==', '=') || ' ' || {_number('rhs')} FROM ({rows})""",
        _line(4, '0', "''"),
        _line(4, '1', "'bounds'"),
        f"""SELECT 5, col::BIGINT, 0::BIGINT,
                   {_bound('lb', '-infinity')} || ' <= x' || CAST(col AS BIGINT)::VARCHAR
                   || ' <= ' || {_bound('ub', '+infinity')} FROM ({numbered})""",
    ]
    for section, kind, keyword in ((6, 'binary', 'binary'), (7, 'integer', 'general')):
        if not any(t == kind for t, _ in vtype_runs):
            continue
        parts += [
            _line(section, '-2', "''"),
            _line(section, '-1', f"'{keyword}'"),
            f"SELECT {section}, col::BIGINT, 0::BIGINT, 'x' || CAST(col AS BIGINT)::VARCHAR "
            f"FROM ({numbered}) WHERE {vtype} = '{kind}'",
        ]
    parts += [_line(8, '0', "''"), _line(8, '1', "'end'")]

    union = ' UNION ALL '.join(f'({p})' for p in parts)
    con.execute(
        f"COPY (SELECT line FROM ({union}) ORDER BY s, k, j) TO '{path}' "
        "(FORMAT CSV, HEADER false, QUOTE '', ESCAPE '', DELIMITER '')"
    )


def _union(tables: list[str], template: str) -> str:
    """The declarations' shares as one relation, or an empty one."""
    if not tables:
        return template.format(t='(SELECT NULL) AS empty') + ' WHERE false'
    return ' UNION ALL '.join(template.format(t=f'"{t}"') for t in tables)


def _vtype_case(runs: list[tuple[str, int]]) -> str:
    """A column's variable type, from the run lengths rather than a stored word.

    The engine holds one word per *declaration*, not per column, for the reason
    `cols` states: the frame's dtype is an Enum and the runs already say how
    long each is. Rebuilt here as a boundary over ``col`` so the word still
    never exists once per row.
    """
    at, arms = 0, []
    for kind, height in runs:
        arms.append(f"WHEN col < {at + height} THEN '{kind}'")
        at += height
    return 'CASE ' + ' '.join(arms) + " ELSE 'continuous' END" if arms else "'continuous'"
