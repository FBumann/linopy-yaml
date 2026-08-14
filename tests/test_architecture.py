"""docs/ARCHITECTURE.md, enforced.

Each test encodes one hard rule from the architecture document, so the doc
cannot silently drift from the code. Static checks parse source with ``ast``
— they need no optional dependencies and run on a bare install.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

REPO = Path(__file__).parent.parent
PKG = REPO / 'src' / 'lpspec'

FORBIDDEN_RUNTIME = {'linopy', 'xarray'}


def _in_linopy_lane(path: Path) -> bool:
    """The linopy/oracle lane — the ONLY modules allowed to import linopy or
    xarray at module level (they load only via ``import lpspec.linopy``).

    Structural, not a filename allowlist: membership is "lives under
    ``linopy/``". A new eager-lane module therefore cannot land outside the
    fence by being spelled differently.
    """
    return 'linopy' in path.relative_to(PKG).parts


def _module_level_imports(path: Path) -> set[str]:
    """Top-level (non-lazy, non-TYPE_CHECKING) imported root packages.

    Module-level ``try:`` blocks count. An optional-dependency guard is still
    a module-level import, and wrapping one must not evade this check —
    ``linopy/__init__.py`` uses exactly that pattern, so the rule has to see through it.
    """
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    stmts = list(tree.body)  # module level only — function bodies are lazy
    while stmts:
        node = stmts.pop()
        if isinstance(node, ast.Import):
            found.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split('.')[0])
        elif isinstance(node, ast.Try):
            stmts.extend([*node.body, *node.orelse, *node.finalbody])
            for handler in node.handlers:
                stmts.extend(handler.body)
    return found


def _all_modules() -> list[Path]:
    return [p for p in PKG.rglob('*.py') if '__pycache__' not in p.parts]


def _imported(
    tree: ast.AST,
    *,
    nodes: Callable[[ast.AST], Iterator[ast.AST]] = ast.walk,
    relative: bool = False,
) -> list[str]:
    """Every imported name in *tree*, as written.

    ``import a.b`` yields ``a.b``; ``from a.b import c`` yields ``a.b``. With
    ``relative=True`` a relative import keeps its dots (``from ..x import y``
    yields ``..x``); otherwise a module-less relative import is dropped.
    *nodes* picks the walk — ``ast.walk`` sees everything,
    :func:`_runtime_nodes` prunes ``TYPE_CHECKING`` bodies.
    """
    names: list[str] = []
    for node in nodes(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if relative:
                names.append('.' * node.level + (node.module or ''))
            elif node.module:
                names.append(node.module)
    return names


def _reaches_past(
    package: str,
    allowed: tuple[str, ...],
    allowlist: set[str],
    *,
    third_party: frozenset[str] = frozenset(),
    nodes: Callable[[ast.AST], Iterator[ast.AST]] = ast.walk,
) -> dict[str, list[str]]:
    """Modules under *package* importing a name its fence forbids.

    Forbidden is an ``lpspec`` name outside *allowed* and *allowlist*, or a
    name whose root package is in *third_party*. Lazy imports are included by
    default: a fence a function body could step over is not one — *nodes* can
    prune instead (:func:`_runtime_nodes`). Membership is read off the path,
    so a new module cannot land outside the fence by being spelled differently.
    """
    offenders = {}
    for path in (PKG / package).rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        bad = [
            n
            for n in _imported(ast.parse(path.read_text()), nodes=nodes)
            if n.split('.')[0] in third_party
            or (n.startswith('lpspec') and not n.startswith(allowed) and n not in allowlist)
        ]
        if bad:
            offenders[str(path.relative_to(PKG))] = sorted(set(bad))
    return offenders


def _runtime_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Every node the interpreter can reach — ``if TYPE_CHECKING:`` bodies pruned.

    The lane fences below exist to stop *running* code from needing the
    oracle's dependencies: that is what breaks a bare install and what would
    stop ``relational/`` being lifted out. A ``TYPE_CHECKING`` body is erased
    before any of that — it is not lazy, it is not executed at all — so
    counting it buys no isolation and costs a public return type, which is how
    ``to_dataarray`` came to be annotated ``Any`` while its own docstring one
    line below says it returns an ``xarray.DataArray``.

    This is not a new position: :func:`_module_level_imports` has always read
    "top-level (non-lazy, non-TYPE_CHECKING)". These walks simply lost the
    distinction by reaching for :func:`ast.walk`, which sees everything.

    The ``else`` branch of such a guard *does* run, so it stays in.
    """
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        stack.extend(ast.iter_child_nodes(node))


def _is_type_checking(test: ast.expr) -> bool:
    """``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``, however it was spelled."""
    if isinstance(test, ast.Name):
        return test.id == 'TYPE_CHECKING'
    return isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'


def test_the_lane_fences_see_running_code_and_only_running_code():
    """The pruner itself, pinned — because both halves have been wrong once.

    Walking everything cost `Result.to_dataarray` its return type: the fence
    read an erased annotation as a dependency and the method was widened to
    `Any` to satisfy it. Walking too little would be worse — a lazy
    `import xarray` in a function body is exactly what the allowlist exists
    to make deliberate. So the line is *does the interpreter reach it*, and
    it is checked in both directions rather than described.
    """
    erased, executed, otherwise = (
        'if TYPE_CHECKING:\n    import xarray\n',
        'def f():\n    import xarray\n',
        'if TYPE_CHECKING:\n    import xarray\nelse:\n    import linopy\n',
    )

    def imported(source: str) -> set[str]:
        return {
            alias.name
            for node in _runtime_nodes(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }

    assert imported(erased) == set(), 'an annotation-only import is not a dependency'
    assert imported(executed) == {'xarray'}, 'a lazy import inside a function still runs'
    assert imported(otherwise) == {'linopy'}, 'the else branch of a TYPE_CHECKING guard does run'


def test_runtime_lane_never_imports_linopy_or_xarray():
    """Hard rule 3: linopy is the eager/oracle lane only — never a runtime import."""
    offenders = {}
    for path in _all_modules():
        if _in_linopy_lane(path):
            continue
        bad = _module_level_imports(path) & FORBIDDEN_RUNTIME
        if bad:
            offenders[str(path.relative_to(PKG))] = sorted(bad)
    assert not offenders, (
        f'runtime modules import linopy-lane packages at module level: {offenders} '
        f'— make the import lazy or move the module into the linopy lane'
    )


#: Modules outside the linopy lane that may reach the oracle *lazily*, with
#: the reason. Being on this list is a deliberate exception, not a default —
#: an eager-only function living in a language module is what put
#: ``evaluate_where`` in ``where_parser.py`` for as long as it did.
LAZY_ORACLE_ALLOWED = {
    'sources.py': 'convex curvature validation needs xarray broadcast (issue #27: make it numpy-only)',
}


def test_lazy_oracle_imports_stay_on_the_allowlist():
    """Hard rule 3, the half a module-level check cannot see.

    A lazy ``import xarray`` inside a function is still eager-lane code, and
    it hides in a module the streaming lane imports. Every one has to be
    declared, so adding another is a decision rather than an accident.
    """
    offenders = {}
    for path in _all_modules():
        if _in_linopy_lane(path) or path.name in LAZY_ORACLE_ALLOWED:
            continue
        tree = ast.parse(path.read_text())
        bad = set()
        for node in _runtime_nodes(tree):
            if isinstance(node, ast.Import):
                bad |= {a.name for a in node.names if a.name.split('.')[0] in FORBIDDEN_RUNTIME}
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in FORBIDDEN_RUNTIME:
                bad.add(node.module)
        if bad:
            offenders[str(path.relative_to(PKG))] = sorted(bad)
    assert not offenders, (
        f'modules outside the linopy lane reach the oracle lazily: {offenders} — '
        f'move the code to the linopy lane, or add it to LAZY_ORACLE_ALLOWED with a reason'
    )


#: Package modules the engine may import: dependency-free leaves that carry no
#: YAML, schema or AST knowledge. ``errors.py`` is one — without it there is no
#: single exception class a caller can catch across both lanes.
ENGINE_MAY_IMPORT = {'lpspec.errors'}


def test_engine_is_isolated():
    """Hard rule 2: the engine knows nothing about linopy, xarray or YAML.

    Enforced as "imports nothing from the package bar ENGINE_MAY_IMPORT",
    which is stricter than the written rule and deliberately so: the plan is
    fed to the engine, and keeping the import surface at zero is what leaves
    the subpackage extractable. Widening it is a decision — add the module to
    ENGINE_MAY_IMPORT with a reason, the way ``errors.py`` is there.
    """
    offenders = _reaches_past(
        'relational',
        ('lpspec.relational',),
        ENGINE_MAY_IMPORT,
        third_party=FORBIDDEN_RUNTIME | {'yaml'},
        nodes=_runtime_nodes,
    )
    assert not offenders, f'engine reaches outside its subpackage: {offenders}'


#: No contract module may name an implementation — the re-export that once
#: earned ``__init__.py`` a place here is gone. Path relative to ``relational/``.
CONTRACT_MAY_NAME_AN_ENGINE: set[str] = set()


def test_no_contract_module_names_an_engine():
    """``relational/__init__.py``'s own split: contract above, ``engines/`` below.

    ``plan.py``, ``sinks/``, ``status.py``, ``chunking.py``, ``result.py`` and
    ``frames.py`` say what a model *is*, what an engine answers to and what a
    sink reads; ``engines/`` implements that. A contract module naming a class
    out of ``engines/`` inverts the two, and a second engine then has to
    satisfy a type written for the first.

    **Type-only imports count here**, where the lane fences above prune them: a
    ``TYPE_CHECKING`` guard is enough to erase a dependency and nowhere near
    enough to erase a design. ``Result`` held ``_engine: PolarsEngine`` behind
    one and called five of its privates.
    """
    offenders = {}
    for path in (PKG / 'relational').rglob('*.py'):
        rel = path.relative_to(PKG / 'relational').as_posix()
        if '__pycache__' in path.parts or rel.startswith('engines/') or rel in CONTRACT_MAY_NAME_AN_ENGINE:
            continue
        named = _imported(ast.parse(path.read_text()), relative=True)
        engines = sorted({m for m in named if 'engines' in m.split('.')})
        if engines:
            offenders[rel] = engines
    assert not offenders, (
        f'a contract module names an implementation: {offenders}. Either the fact belongs '
        f'under engines/, or what crosses the seam should be a type the contract already owns'
    )


#: What ``language/`` may reach: itself, and the same dependency-free leaves the
#: engine may reach. Both fences point at ``errors.py`` for the same reason —
#: one exception hierarchy, owned by neither side. Widening this is a decision,
#: exactly as widening ``ENGINE_MAY_IMPORT`` is.
LANGUAGE_MAY_IMPORT = ENGINE_MAY_IMPORT


def test_language_never_reaches_a_consumer():
    """Hard rule 1, the other direction: the waist is closed from the front.

    Hard rule 2 keeps the engine from seeing the schema or the AST. This is its
    mirror: what a model *means* may not depend on what any consumer does with
    it, so nothing under ``language/`` imports ``lowering``, ``piecewise``,
    ``sources``, ``api``, or the relational / linopy / typeset subpackages.

    That is what makes ``lps.check()`` a pass with no data and no plan, and a
    second consumer cheap rather than a second opinion.
    """
    offenders = _reaches_past('language', ('lpspec.language',), LANGUAGE_MAY_IMPORT)
    assert not offenders, (
        f'the language reaches forward to a consumer: {offenders} — a front-end module '
        f'may not depend on what is done with the AST it produces'
    )


#: What ``typeset/`` may reach. The language and nothing else: a renderer reads
#: the AST and writes text, so it must not be able to acquire an opinion an
#: engine holds. ``lpspec.errors`` for the exception hierarchy, as everywhere.
TYPESET_MAY_IMPORT = ENGINE_MAY_IMPORT


def test_typeset_reads_the_language_and_reaches_no_engine():
    """The fourth fence — the one that was prose only.

    ``typeset/`` is the proof that a consumer of the AST is cheap: it renders
    any model the lanes can build, from one walk, holding no opinion they do
    not already hold. That claim is only worth anything if it *cannot* reach
    the plan, a sink, a solver or a dataframe — so ``import lpspec.typeset``
    must not drag in an engine. It used to, through ``api.load_model``, and
    nothing failed; the module map said otherwise and no test read it.
    """
    offenders = _reaches_past('typeset', ('lpspec.language', 'lpspec.typeset'), TYPESET_MAY_IMPORT)
    assert not offenders, (
        f'typeset reaches past the language: {offenders} — a renderer reads the AST '
        f'and writes text; it may not reach a plan, a sink, a solver or a dataframe'
    )


def test_typesets_import_closure_needs_no_third_party_engine():
    """The same fence, transitively — the half a one-hop name check misses.

    A renderer that imports only ``language/`` still pays for polars if some
    language module does. So this walks the *closure* of module-level imports
    from ``typeset/`` and asserts no engine third-party lands in it: what a
    consumer binding no data costs is the point of the fence, not which names
    it happens to spell.

    Deliberately not observed via ``import lpspec.typeset`` in a subprocess:
    importing a submodule runs ``lpspec/__init__.py``, which eagerly exposes
    ``build``/``solve`` and so loads the runner whatever this subpackage does.
    That is a property of the top-level namespace, not of ``typeset/``.

    The walk follows module-level ``from lpspec.x import y`` edges, resolved
    back to modules.
    """
    heavy = {'polars', 'highspy', 'numpy', 'pandas'} | FORBIDDEN_RUNTIME
    by_module = {
        f'lpspec.{p.relative_to(PKG).with_suffix("").as_posix().replace("/", ".").removesuffix(".__init__")}': p
        for p in _all_modules()
    }
    seen, stack, reached = set(), ['lpspec.typeset'], {}
    while stack:
        mod = stack.pop()
        if mod in seen or mod not in by_module:
            continue
        seen.add(mod)
        for imported in _module_level_imports(by_module[mod]):
            if imported in heavy:
                reached.setdefault(imported, []).append(mod)
        for node in ast.walk(ast.parse(by_module[mod].read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('lpspec'):
                stack.append(node.module)
            elif isinstance(node, ast.Import):
                stack += [a.name for a in node.names if a.name.startswith('lpspec')]
    assert not reached, (
        f'the typeset import closure reaches an engine dependency: {reached} — '
        f'typesetting a model must not cost a dataframe library or a solver binding'
    )


#: The whole Python surface, by role. Hard rule 5 says the public interface is
#: a declared model rather than a Python API — this is what "rather than" is
#: worth in names. Adding one is a row here, which is a line in a diff a
#: reviewer reads; the fences elsewhere in this file work the same way.
PUBLIC_API = {
    'run it': {'build', 'check', 'solve', 'write'},
    'run it many times': {'solve_over', 'EachCoordinate', 'EachWindow'},
    'load it': {'load_model', 'Model'},
    'show it': {'to_latex', 'to_markdown', 'to_typst', 'SymbolTable'},
    'catch it': {
        'LpspecError',
        'LanguageError',
        'DataError',
        'DimensionError',
        'SchemaError',
        'PiecewiseExpansionError',
    },
}

#: The opt-in shim, which is a surface of its own — deliberately three verbs:
#: two producers, and the named-expression reader both lanes owe (#562).
PUBLIC_API_LINOPY = {'build', 'expression', 'extend'}


def test_the_public_surface_is_exactly_what_is_declared():
    """Hard rule 5, in names: the Python surface is narrow, and stays narrow.

    Narrow is a feature, not an accident — it is the half of "the public
    interface is a declared model" that a reader can count. A model travels as
    YAML; Python is how you *run* it, so the runner has four verbs, the
    language one loader, the readers one each, and the errors one hierarchy.

    Two directions, because either alone rots. ``__all__`` must match the
    table (a name added quietly, or documented and never exported), and no
    public non-module attribute may exist outside it (a helper that leaked
    into the namespace by being imported at the top of ``__init__``).
    """
    import inspect

    import lpspec

    declared = {name for names in PUBLIC_API.values() for name in names}
    assert set(lpspec.__all__) == declared, (
        f'lpspec.__all__ and PUBLIC_API disagree: only in __all__ '
        f'{sorted(set(lpspec.__all__) - declared)}, only in the table '
        f'{sorted(declared - set(lpspec.__all__))} — add the name to PUBLIC_API '
        f'with the role it plays, and to docs/ARCHITECTURE.md'
    )

    leaked = sorted(
        name
        for name in dir(lpspec)
        if not name.startswith('_')
        and name not in declared
        and not inspect.ismodule(getattr(lpspec, name))  # submodules are import paths, not API
    )
    assert not leaked, (
        f'public names outside __all__: {leaked} — a surface that grows by '
        f'accident is not narrow. Import it privately, or declare it.'
    )


def test_the_linopy_shim_stays_three_verbs():
    """The shim attaches YAML math to a model, and reads back what it named.

    ``build`` makes a model, ``extend`` adds to one, ``expression`` evaluates
    a declared named quantity at its solution — the eager half of a reader
    both lanes owe (hard rule 3), pure like the producers. A fourth verb would
    mean the shim had started being a lane of its own, which hard rule 3
    spends its length refusing. Read statically: the module imports linopy,
    and this must run on a bare install.
    """
    tree = ast.parse((PKG / 'linopy' / '__init__.py').read_text())
    declared = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == '__all__' for t in node.targets)
    )
    assert set(declared) == PUBLIC_API_LINOPY, f'the linopy shim exports {sorted(declared)}'


def test_expansion_has_no_mutable_module_state():
    """Hard rule 5: YAML files are self-contained — nothing importable may
    accumulate state that changes what a file means."""
    tree = ast.parse((PKG / 'language' / 'expansion.py').read_text())
    mutable = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, (ast.Dict, ast.List, ast.Set)) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in {'dict', 'list', 'set'}
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                mutable += [ast.unparse(t) for t in targets]
    assert not mutable, (
        f'expansion.py holds mutable module-level state {mutable} — '
        f'macros/expressions must live in the schema, not a registry'
    )


#: The two sink families. The directory *is* the family, so a member cannot
#: land in the wrong one by being spelled differently.
SINKS = PKG / 'relational' / 'sinks'


def _family(name: str) -> set[str]:
    return {p.stem for p in (SINKS / name).glob('*.py') if p.stem != '__init__'}


def test_each_sink_family_is_its_directory_and_its_registry():
    """One shape per family, checked off the path.

    A solver is four things that must agree: a module under ``solvers/`` named
    for the solver, a ``Solver`` subclass defined *in that module*, a
    ``build_<name>`` seam for `bench/`, and the ``SOLVERS`` key holding the
    class. Writers are keyed by suffix instead, but the rule is the same.
    Agreement is what makes adding one mechanical — nothing above the module
    to teach, nothing to remember but the name.

    The class, because a solver holds a model between solves. Defined in its
    own module, so the fence that keeps ``gurobipy`` off a HiGHS caller's
    import path holds.
    """
    import importlib

    from lpspec.relational.sinks import SOLVERS, WRITERS, Solver

    solvers = _family('solvers') - {'base'}
    assert set(SOLVERS) == solvers, f'solver modules and SOLVERS keys disagree: {solvers ^ set(SOLVERS)}'
    for name in sorted(solvers):
        module = importlib.import_module(f'lpspec.relational.sinks.solvers.{name}')
        held = SOLVERS[name]
        assert issubclass(held, Solver), f'SOLVERS[{name!r}] is not a Solver'
        assert held.__module__.rsplit('.', 1)[-1] == name, (
            f"{name}'s solver is defined in {held.__module__} — it belongs to its own module"
        )
        assert held.requires and all(isinstance(package, str) for package in held.requires), (
            f'{name} does not name the packages it needs, so is_available() cannot answer for it'
        )
        assert isinstance(held.is_available(), bool), (
            f'{name}.is_available() must answer without importing the solver or raising'
        )
        assert held.unavailable_message, f'{name} does not say what to do when is_available() is False'
        assert hasattr(module, f'build_{name}'), f'{name} has no build_{name}: the load-only seam `bench/` measures'

    assert {w.__module__.rsplit('.', 1)[-1] for w in WRITERS.values()} == _family('writers')
    assert all(s.startswith('.') for s in WRITERS), 'writers are keyed by file suffix'


def test_the_engine_dtype_table_matches_the_declared_vocabulary():
    """``frames._DECLARED`` spells the dtype set the language validates.

    One vocabulary, two homes by necessity — the engine may not import the
    language (hard rule 2) — so a test is what keeps the copy honest: a dtype
    added to ``DIMENSION_DTYPES`` without a polars dtype here would fail
    ``labels_frame`` on the empty-index path with a ``KeyError``.
    """
    from lpspec.language.model import DIMENSION_DTYPES
    from lpspec.relational.frames import _DECLARED

    assert set(_DECLARED) == set(DIMENSION_DTYPES), 'the two homes of the dimension dtype vocabulary disagree'


def test_the_plan_variable_type_matches_the_declared_domains():
    """``plan.VariableType`` spells the domain set the language validates.

    Same fence, same remedy as the dtype table above: the engine may not
    import the language, so a test keeps the copy honest — the lowering casts
    ``vdef.domain`` straight into ``plan.VariableType``, and a domain added to
    one home without the other would send an unknown type into every sink.
    """
    from typing import get_args

    from lpspec.language.model import VARIABLE_DOMAINS
    from lpspec.relational.plan import VariableType

    assert set(get_args(VariableType)) == set(VARIABLE_DOMAINS), (
        'the two homes of the variable domain vocabulary disagree'
    )


def test_no_sink_reaches_a_sibling():
    """The fence that keeps an optional dependency optional.

    ``gurobipy`` stays the ``gurobi`` module's alone only because no other
    sink imports it, directly or by importing the module that does. A leaf
    reads ``tables.py``, its family's ``base``, and its own dependency —
    nothing else in the family.

    ``base`` is allowed for the reason the rest is not: it imports no solver,
    so it cannot carry one across, and it is what stops the alternative — one
    leaf importing the other to share a rule — from being the tempting option.
    A ``base`` that reached for a leaf would fail the same check.
    """
    shareable = ('.tables', '.base')
    offenders = {}
    for family in ('solvers', 'writers'):
        for path in sorted((SINKS / family).glob('*.py')):
            reached = {
                name
                for name in _imported(ast.parse(path.read_text()))
                if name.startswith('lpspec.relational.sinks.') and not name.endswith(shareable)
            }
            if reached and path.stem != '__init__':
                offenders[f'{family}/{path.name}'] = sorted(reached)
    assert not offenders, (
        f'sink modules reaching a sibling: {offenders} — a sink reads tables.py, its family base '
        f'and its own dependency; anything else shared belongs on one of those two'
    )


def test_every_plan_node_is_handled_by_the_compiler():
    """Two-tier economy: a primitive is not done until the engine consumes it.

    The compiler is the consumer — it is the module that turns plan nodes into
    SQL, so a node it does not mention has no relational meaning however much
    the engine moves around it. Grep-level drift alarm; the differential
    tests prove semantics.
    """
    import lpspec.relational.plan as plan

    compiler_src = (PKG / 'relational' / 'engines' / 'polars' / 'compiler.py').read_text()
    for base in (plan.Expression, plan.Predicate):
        unhandled = [c.__name__ for c in base.__subclasses__() if f'plan.{c.__name__}' not in compiler_src]
        assert not unhandled, f'plan.{base.__name__} nodes unknown to the compiler: {unhandled}'


def test_both_lanes_implement_exactly_the_closed_operator_set():
    """Hard rule 3: one language, two lanes. An operator name the eager lane
    evaluates but the relational lane cannot lower (or vice versa) is a
    dialect split, and it would make the differential tests meaningless.

    Read statically: ``linopy/builder.py`` imports xarray at module level (it
    is linopy lane), and this check must still run on a bare install.

    The eager lane keeps a table; the relational lane spells its cases out in
    ``lowering.py``, so every declared name has to appear there as a lowering
    branch.
    """
    from lpspec.language.operators import BUILTIN_NAMES

    tree = ast.parse((PKG / 'linopy' / 'builder.py').read_text())
    table = next(
        node.value for node in tree.body if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == '_OPERATORS'
    )
    assert isinstance(table, ast.Dict)
    eager = {ast.literal_eval(k) for k in table.keys if k is not None}

    assert eager == set(BUILTIN_NAMES), (
        f'eager lane implements {sorted(eager)}, language declares {sorted(BUILTIN_NAMES)}'
    )

    lowering_src = (PKG / 'lowering.py').read_text()
    missing = [name for name in BUILTIN_NAMES if f"'{name}'" not in lowering_src]
    assert not missing, f'built-in operators with no lowering case: {missing}'


def test_every_module_is_documented_somewhere():
    """No module is undocumented — but the doc need not be docs/ARCHITECTURE.md.

    A subpackage that grows a member per variant (one sink per module) would
    push its whole membership list into the top-level map, which is the thing
    that map exists *not* to be. A ``README.md`` beside the code counts
    instead: it is what you read when you open the directory, and it stays
    next to the thing it describes.

    "Beside" reaches *up* as well as across, because a family may be a
    directory of its own — ``sinks/solvers/highs.py`` is documented by
    ``sinks/README.md``, which is the page describing both families. One
    README per tree, not one per level.
    """
    architecture = (REPO / 'docs/ARCHITECTURE.md').read_text()
    missing = []
    for path in _all_modules():
        name = path.name
        if name.startswith('_'):
            continue  # private plumbing (_notes) needs no doc entry
        if name == '__init__.py':
            continue
        readmes = [d / 'README.md' for d in path.parents if PKG in d.parents or d == PKG]
        documented = name in architecture or any(r.exists() and name in r.read_text() for r in readmes)
        if not documented:
            missing.append(str(path.relative_to(PKG)))
    assert not missing, (
        f'undocumented modules: {missing} — add each to docs/ARCHITECTURE.md, or to a '
        f'README.md in its own directory if it is one member of a family'
    )


def test_every_schema_model_is_strict():
    """A schema model that inherits BaseModel directly silently drops unknown
    keys, which turns a typo into a different model. Strictness lives on
    ``_StrictBlock``, so the check is that nothing bypasses it."""
    tree = ast.parse((PKG / 'language' / 'model.py').read_text())
    loose = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name != '_StrictBlock'
        and any(isinstance(b, ast.Name) and b.id == 'BaseModel' for b in node.bases)
    ]
    assert not loose, (
        f'schema models {loose} inherit BaseModel directly and so accept unknown keys — inherit _StrictBlock instead'
    )


#: Every in-function ``lpspec`` import in the package, with why it is one.
#: Empty, and that is the claim: the layers are ordered with no exception at
#: all. The one entry this used to hold broke a real cycle — ``piecewise``
#: consulted the plan's subset test while lowering had to expand before it
#: could lower. Expansion no longer asks the plan anything, so the cycle is
#: gone rather than deferred, and a lazy import here is once again only ever
#: a leftover.
DELIBERATE_LAZY_IMPORTS: dict[tuple[str, str], str] = {
    ('language/model.py', 'lpspec.language.piecewise'): (
        'Model validates its own expressions, and the checkers read Model. Both '
        'sit in `language/`, so this is one layer calling itself rather than a '
        'reach across layers.'
    ),
    ('language/model.py', 'lpspec.language.validation'): (
        'Same cycle: validation.py imports Model to build one, and Model calls '
        'validate_expressions on itself so the type cannot exist half-checked.'
    ),
    # not a cycle but an extra: _yaml.py needs pyyaml, which ships as
    # `lpspec[yaml]`, so importing it at module level would make pyyaml a
    # hard dependency again. Only the `.yaml` branch may pay for it.
    ('language/validation.py', 'lpspec.language._yaml'): ('the [yaml] extra — see the comment above'),
    ('language/model.py', 'lpspec.language._yaml'): ('the [yaml] extra — see the comment above'),
    ('typeset/symbols.py', 'lpspec.language._yaml'): ('the [yaml] extra — see the comment above'),
}


def test_lazy_intra_package_imports_are_all_declared():
    """Hard rule 0, mechanically: the layers are ordered, with no exception.

    An undeclared in-function import is either a cycle nobody noticed or a
    leftover. Both are worth a line of explanation, so both fail here.
    """
    found = {}
    for path in _all_modules():
        tree = ast.parse(path.read_text())
        module_level = set()
        stack = list(tree.body)
        while stack:
            node = stack.pop()
            module_level.add(id(node))
            if isinstance(node, ast.Try):
                stack += [*node.body, *node.orelse, *node.finalbody]
                stack += [b for h in node.handlers for b in h.body]
            elif isinstance(node, ast.If):
                stack += [*node.body, *node.orelse]
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith('lpspec')
                and id(node) not in module_level
            ):
                found[(str(path.relative_to(PKG)), node.module)] = node.lineno

    undeclared = {k: v for k, v in found.items() if k not in DELIBERATE_LAZY_IMPORTS}
    assert not undeclared, (
        f'undeclared in-function imports {undeclared} — hoist them to module level, '
        f'or add them to DELIBERATE_LAZY_IMPORTS with the cycle they break'
    )
    stale = set(DELIBERATE_LAZY_IMPORTS) - set(found)
    assert not stale, f'DELIBERATE_LAZY_IMPORTS lists imports that no longer exist: {stale}'
