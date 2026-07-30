"""
Structural regression guard: app/main.py's main() must call db.init_db()
unconditionally before load_everything() runs.

Real production bug (Streamlit Cloud, fresh/empty deployment): main()
called load_everything() (which writes to the experiment_runs and
cell_summaries tables via _train_and_predict()/_persist_cell_data()) right
after the `if not render_login(): return` gate. render_login() only calls
db.init_db() on the script run where the user isn't authenticated yet --
every later run (including the one that actually reaches load_everything(),
since it's only reached because authenticated is already True) skips that
call entirely, relying on an earlier run having created the tables first.
That's usually true but not guaranteed (e.g. concurrent sessions racing a
cold, empty deployment) -- reproduced locally with a genuinely fresh
sqlite DB: sqlalchemy.exc.OperationalError: no such table: experiment_runs.

Every AppTest-based test in this suite uses the isolated_db fixture, which
calls db.init_db() directly during fixture setup before any AppTest ever
runs -- so this ordering bug was invisible to the entire existing test
suite despite exercising main()'s full script body repeatedly. Hence a
structural guard instead of (or in addition to) a slow full end-to-end
AppTest against a truly empty database.
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent
MAIN_PY = REPO_ROOT / "app" / "main.py"


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"could not find a top-level function named {name!r} in {MAIN_PY}")


def _calls_load_everything(stmt: ast.stmt) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "load_everything"
        for n in ast.walk(stmt)
    )


def _is_unconditional_init_db_call(stmt: ast.stmt) -> bool:
    """True if stmt is a top-level (not nested inside an if/for/while/try)
    expression statement calling something.init_db()."""
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    func = stmt.value.func
    return isinstance(func, ast.Attribute) and func.attr == "init_db"


def test_main_calls_init_db_unconditionally_before_load_everything():
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    main_fn = _find_function(tree, "main")

    load_everything_idx = None
    for i, stmt in enumerate(main_fn.body):
        if _calls_load_everything(stmt):
            load_everything_idx = i
            break
    assert load_everything_idx is not None, "main() no longer calls load_everything() -- update this guard"

    unconditional_init_db_before = any(
        _is_unconditional_init_db_call(stmt) for stmt in main_fn.body[:load_everything_idx]
    )
    assert unconditional_init_db_before, (
        "main() must call db.init_db() as an unconditional (not nested inside "
        "an if/for/while/try) top-level statement before load_everything() -- "
        "load_everything() writes to tables that must already exist. Relying "
        "solely on render_login()'s conditional init_db() call is not "
        "sufficient: that call is skipped on any run that reaches "
        "load_everything() (since it's only reached because 'authenticated' "
        "is already True), leaving table creation dependent on an earlier "
        "run having happened first -- not guaranteed on a fresh deployment."
    )
