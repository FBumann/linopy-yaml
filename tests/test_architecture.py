"""docs/about/architecture.md, enforced.

Each test encodes one hard rule from the architecture document, so the doc
cannot silently drift from the code. Static checks parse source with ``ast``
— they need no optional dependencies and run on a bare install.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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
    'sources.py': 'curvature validation needs xarray broadcast (issue #27: make it numpy-only)',
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
#: single exception class a caller can catch across both lanes. ``frames.py``
#: is the second and was earned rather than granted: it is the one place that
#: knows what a caller's table library is, and all three consumers — the front
#: door, the driver and the linopy lane — read it, so living under the engine
#: it happens to be nearest was a lie about who owns it.
ENGINE_MAY_IMPORT = {'lpspec.errors', 'lpspec.frames'}


def test_engine_is_isolated():
    """Hard rule 2: the engine knows nothing about linopy, xarray or YAML.

    Enforced as "imports nothing from the package bar ENGINE_MAY_IMPORT",
    which is stricter than the written rule and deliberately so: the plan is
    fed to the engine, and keeping the import surface at zero is what leaves
    the subpackage extractable. Widening it is a decision — add the module to
    ENGINE_MAY_IMPORT with a reason, the way ``errors.py`` is there.

    What the engine *names* is checked here; what those names cost is not.
    ``errors.py`` re-exports the language's half of the hierarchy, so importing
    it now loads the language package too. That is deliberate and stated in
    hard rule 2: a root class cannot live downstream of what extends it. The
    day the engine stops raising ``LanguageError`` it could be a leaf again.
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


#: Where python this repository owns lives. ``.pixi`` and a worktree parked
#: under the checkout are neither ours nor scanned.
SOURCE_DIRS = ('src', 'tests', 'tools', 'bench', 'examples')


def _repository_modules() -> list[Path]:
    return [p for d in SOURCE_DIRS for p in (REPO / d).rglob('*.py') if '__pycache__' not in p.parts]


def test_the_language_is_imported_as_one_package():
    """Hard rule 1, the half of it that is still ours to keep.

    The language moved to ``math_spec`` and took its fence with it: the
    allowlist that said the directory imports nothing from this package is a
    dependency edge now, and no test here can step over it. What a test here
    *can* still hold is the traffic in the other direction — that this
    repository depends on the one ``__all__`` math-spec pins rather than on the
    union of whatever its submodules expose.

    A submodule path is a contract nobody agreed to. It can carry a private
    name, it is not counted in the surface upstream pins in both directions,
    and it survives a refactor there that the package export would have caught.
    ``from math_spec import Model`` fails loudly the day ``Model`` stops being
    exported; ``from math_spec.model import Model`` keeps working until it does
    not.

    Nothing is exempt, by directory or by name. A test reaching inside is the
    same unagreed contract as a module doing it, and the exemption this rule
    used to grant to ``tests/`` is where every one of them had accumulated.
    """
    offenders = {}
    for path in _repository_modules():
        inside = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('math_spec.'):
                inside += [f'{node.module}.{alias.name}' for alias in node.names]
            elif isinstance(node, ast.Import):
                inside += [alias.name for alias in node.names if alias.name.startswith('math_spec.')]
        if inside:
            offenders[str(path.relative_to(REPO))] = sorted(inside)
    assert not offenders, (
        f'modules reach inside the language package: {offenders} — import the name from '
        f'`math_spec` itself, which is the surface it pins'
    )


#: Directory prefixes a workflow can name that are files in this repository.
#: Anything else in a `run:` block is a runner path, a container path or a shell
#: variable, and none of those are ours to check.
REPO_PREFIXES = ('examples/', 'tests/', 'src/', 'docs/', 'tools/', 'schema/', 'bench/')


def test_every_repository_path_a_workflow_names_exists():
    """A workflow step reads files by path, and a move makes it read nothing.

    Filed as a guard because it happened: `tests/golden/` moved to
    `tests/typeset/golden/`, the whole suite stayed green locally, and CI went
    red on a step that renders every model by path. Nothing else looks here —
    the fences read imports, and the crossings check answers "does this file
    name something that moves", not "is every path it names still right".

    Globs are resolved rather than skipped: `examples/*.yaml` matching nothing
    is the same silent hole as a missing file.
    """
    missing = []
    for workflow in sorted((REPO / '.github' / 'workflows').glob('*.y*ml')):
        for token in workflow.read_text().split():
            token = token.strip('\'"`,')
            if not token.startswith(REPO_PREFIXES):
                continue
            hits = list(REPO.glob(token)) if any(c in token for c in '*?[') else [REPO / token]
            if not any(path.exists() for path in hits):
                missing.append(f'{workflow.name}: {token}')
    assert not missing, (
        f'a workflow names paths that do not exist: {missing} — a step reading them '
        f'reads nothing, and no test outside CI would notice'
    )


#: The whole Python surface, by role. Hard rule 5 says the public interface is
#: a declared model rather than a Python API — this is what "rather than" is
#: worth in names. Adding one is a row here, which is a line in a diff a
#: reviewer reads; the fences elsewhere in this file work the same way.
PUBLIC_API = {
    'run it': {'build', 'check', 'solve', 'write'},
    'run it many times': {'solve_over', 'EachCoordinate', 'EachWindow'},
    'catch it': {
        'LpspecError',
        'LanguageError',
        'LaneError',
        'DataError',
        'DimensionError',
        'SchemaError',
        'PiecewiseExpansionError',
    },
}

#: The opt-in lane, which is a surface of its own — deliberately two verbs:
#: the producer, and the named-expression reader both lanes owe (#562).
PUBLIC_API_LINOPY = {'build', 'expression'}


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
        f'with the role it plays, and to docs/about/architecture.md'
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


def test_the_linopy_lane_stays_two_verbs():
    """The lane constructs a model, and reads back what the file named.

    ``build`` makes a model and ``expression`` evaluates a declared named
    quantity at its solution — the eager half of a reader both lanes owe
    (hard rule 3), pure like the producer. What is refused here is a verb that
    *attaches* to a model something else built: a file references only what it
    declares (hard rule 5), and the verb that made an exception of that is
    gone (#845). Read statically: the module imports linopy, and this must run
    on a bare install.
    """
    tree = ast.parse((PKG / 'linopy' / '__init__.py').read_text())
    declared = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == '__all__' for t in node.targets)
    )
    assert set(declared) == PUBLIC_API_LINOPY, f'the linopy lane exports {sorted(declared)}'


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

    assert {w.write.__module__.rsplit('.', 1)[-1] for w in WRITERS.values()} == _family('writers') - {'base'}
    assert all(s.startswith('.') for s in WRITERS), 'writers are keyed by file suffix'


def test_every_sink_declares_what_it_can_ingest():
    """Both families answer the capability axis, in one vocabulary.

    Three silent failures: a sink declaring nothing reads as ``absent``
    everywhere and is refused a model it can take, one naming a capability
    outside the vocabulary is refused nothing, since no required set can
    contain a name that is not in it, and one *answering* outside the
    vocabulary is read as ``absent`` by every comparison in the family — a
    ``'Native'`` on gurobi silently takes the big-M rewrite instead of the sets
    it branches on.
    """
    from typing import get_args

    from lpspec.relational.sinks import SOLVERS, WRITERS
    from lpspec.relational.sinks.capabilities import (
        CAPABILITIES,
        REWRITTEN_AS_INTEGRALITY,
        Capabilities,
        Support,
    )

    described = {f'solver {name}': held.capabilities for name, held in SOLVERS.items()}
    described |= {f'writer {suffix}': found.capabilities for suffix, found in WRITERS.items()}
    for sink, capabilities in described.items():
        assert isinstance(capabilities, Capabilities), f'{sink} declares no capabilities'
        strangers = sorted(set(capabilities.supports) - set(CAPABILITIES))
        assert not strangers, f'{sink} names capabilities the vocabulary has not got: {strangers}'
        answers = sorted(set(capabilities.supports.values()) - set(get_args(Support)))
        assert not answers, f'{sink} answers {answers}, which no comparison in the family reads as support'
        spent = sorted(c for c in REWRITTEN_AS_INTEGRALITY if capabilities.support(c) == 'reformulated')
        if spent:
            assert capabilities.support('integrality') != 'absent', (
                f'{sink} rewrites {spent} into binaries and linking rows, which is integrality it '
                f'does not declare — the rewrite it promises is one it cannot perform'
            )
        for combination in capabilities.excludes:
            unsupported = sorted(c for c in combination if capabilities.support(c) == 'absent')
            assert not unsupported, (
                f'{sink} excludes the combination {sorted(combination)} while lacking {unsupported} '
                f'outright — an exclusion is about a *pair* it has both halves of, and a capability '
                f'it simply does not have is already refused on its own'
            )


def test_the_engine_dtype_table_matches_the_declared_vocabulary():
    """``frames._DECLARED`` spells the dtype set the language validates.

    One vocabulary, two homes by necessity — the engine may not import the
    language (hard rule 2) — so a test is what keeps the copy honest: a dtype
    added to ``DIMENSION_DTYPES`` without a polars dtype here would fail
    ``labels_frame`` on the empty-index path with a ``KeyError``.
    """
    from math_spec import DIMENSION_DTYPES

    from lpspec.frames import _DECLARED

    assert set(_DECLARED) == set(DIMENSION_DTYPES), 'the two homes of the dimension dtype vocabulary disagree'


def test_the_relational_lane_accepts_the_declared_parameter_dtype_vocabulary():
    """Every declared dtype has a column table entry.

    Same fence, same remedy as the dimension table above: the engine may not
    import the language, and a dtype added to ``PARAMETER_DTYPES`` without an
    entry here would fail at bind with a ``KeyError`` on the first parameter
    that declared it, rather than at load with a sentence.

    The widening is pinned with it: ``int`` serves ``float`` and nothing else
    is widened, because whole numbers are numbers and the shipped instances
    carry them. A second exception added quietly is what this catches.
    """
    from math_spec import PARAMETER_DTYPES

    from lpspec.relational.engines.polars.data_validation import _COLUMNS, ACCEPTED_VALUE_TYPES

    assert set(_COLUMNS) == set(PARAMETER_DTYPES), 'the column table and the language disagree'
    assert set(ACCEPTED_VALUE_TYPES) == set(PARAMETER_DTYPES), 'the accepted table and the language disagree'

    widened = {name: set(types) - set(_COLUMNS[name]) for name, types in ACCEPTED_VALUE_TYPES.items()}
    assert widened == {'float': set(_COLUMNS['int']), 'int': set(), 'bool': set(), 'str': set()}, (
        'int-for-float is the only widening'
    )


def test_the_eager_lane_takes_the_same_vocabulary_and_the_same_widening():
    """The second lane's copy of both, which is where they could drift apart.

    Imported inside the test rather than at module scope: this module runs on
    the bare-install job, where the ``[linopy]`` extra is absent by design, and
    a top-level import of the lane would fail collection there rather than skip.
    """
    pytest.importorskip('linopy', reason='needs the [linopy] extra')

    from math_spec import PARAMETER_DTYPES

    from lpspec.linopy.loader import _ACCEPTED_KINDS, _KINDS

    assert set(_KINDS) == set(PARAMETER_DTYPES), 'the eager kind table and the language disagree'
    assert _ACCEPTED_KINDS == {'float': 'fiu', 'int': 'iu', 'bool': 'b', 'str': 'OUS'}, (
        'and the eager lane widens the same one the relational lane does'
    )


def test_the_plan_variable_type_matches_the_declared_domains():
    """``plan.VariableType`` spells the domain set the language validates.

    Same fence, same remedy as the dtype table above: the engine may not
    import the language, so a test keeps the copy honest — the lowering casts
    ``vdef.domain`` straight into ``plan.VariableType``, and a domain added to
    one home without the other would send an unknown type into every sink.
    """
    from typing import get_args

    from math_spec import VARIABLE_DOMAINS

    from lpspec.relational.plan import VariableType

    assert set(get_args(VariableType)) == set(VARIABLE_DOMAINS), (
        'the two homes of the variable domain vocabulary disagree'
    )


def test_the_plan_absence_matches_the_declared_absence():
    """``plan.VariableAbsence`` spells the absence set the language validates.

    The same fence again, and a sharper failure: the lowering casts
    ``vdef.absence`` straight into the plan, and the compiler tests it with
    ``== 'undefined'``. A reading added to one home alone would arrive as a
    string no branch recognises and be sent down the *propagating* path by
    default — rows deleted under a spelling that asked for the opposite.
    """
    from typing import get_args

    from math_spec import VARIABLE_ABSENCE

    from lpspec.relational.plan import VariableAbsence

    assert set(get_args(VariableAbsence)) == set(VARIABLE_ABSENCE), (
        'the two homes of the variable absence vocabulary disagree'
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

    ``capabilities`` joined it on the same argument: a frozen descriptor and
    two ``Literal`` vocabularies, read by **both** families — where one per
    family would be two spellings of a single axis. It names ``plan`` for the
    type of the program ``required`` reads, and that is a ``TYPE_CHECKING``
    import: it runs nothing, so a leaf still carries nothing across.
    """
    shareable = ('.tables', '.base', '.capabilities')
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
    from math_spec import BUILTIN_NAMES

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
    """No module is undocumented — but the doc need not be docs/about/architecture.md.

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
    architecture = (REPO / 'docs/about/architecture.md').read_text()
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
        f'undocumented modules: {missing} — add each to docs/about/architecture.md, or to a '
        f'README.md in its own directory if it is one member of a family'
    )


#: Every in-function ``lpspec`` import in the package, with why it is one.
#: Empty, and that is the claim: the layers are ordered with no exception at
#: all. The one entry this used to hold broke a real cycle — ``piecewise``
#: consulted the plan's subset test while lowering had to expand before it
#: could lower. Expansion no longer asks the plan anything, so the cycle is
#: gone rather than deferred, and a lazy import here is once again only ever
#: a leftover. Empty since the language left: both entries were `language/model.py`
#: calling its own layer, and that layer is now another package.
DELIBERATE_LAZY_IMPORTS: dict[tuple[str, str], str] = {}


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
