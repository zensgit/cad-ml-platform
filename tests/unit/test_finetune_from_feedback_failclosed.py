"""L3: prove offline scripts/finetune_from_feedback.py is fail-closed at the eval-integrity gate.

This script trains AND promotes (reload_model force=True). The unconditional gate that
already stops auto_retrain.sh must also be the FIRST executable statement of main() —
before ArgumentParser / parse_args, export, data reads, training, serialization,
directory creation, or reload — and the CLI bootstrap must run the gate before
third-party/application imports (numpy/src) and sys.path mutation.

Discriminating cases (each proves the pre-seal path would have continued):
  - gate raises GateBlocked
  - gate module / symbol missing or unusable
  - gate raises an arbitrary exception
  - gate returns unexpectedly (subverted)
  - --help and malformed/unknown argv still exit 1 from the gate (not argparse 0/2)
  - BaseException SystemExit(0)/KeyboardInterrupt remapped to exit 1
  - GateBlocked path secrets never logged; CLI never imports poisoned numpy/src

No CLI arg or env var may open the path. See PRODUCT_STRATEGY.md §5.2, §8.1.
"""

from __future__ import annotations

import ast
import builtins
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts import eval_integrity_gate as gate_mod
from scripts import finetune_from_feedback as module

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "finetune_from_feedback.py"


def _spy_downstream(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace every side-effectful step after the gate with counters that must stay zero."""
    hits: dict[str, int] = {
        "get_active_learner": 0,
        "load_training_data": 0,
        "train_model": 0,
        "makedirs": 0,
        "pickle_dump": 0,
        "classifier_import": 0,
        "reload_model": 0,
    }

    def _boom(name: str) -> Callable[..., Any]:
        def _inner(*_a: Any, **_k: Any) -> Any:
            hits[name] += 1
            raise AssertionError(f"{name} must never run — gate must fail closed first")

        return _inner

    monkeypatch.setattr(module, "get_active_learner", _boom("get_active_learner"))
    monkeypatch.setattr(module, "load_training_data", _boom("load_training_data"))
    monkeypatch.setattr(module, "train_model", _boom("train_model"))
    monkeypatch.setattr(module.os, "makedirs", _boom("makedirs"))
    monkeypatch.setattr(module.pickle, "dump", _boom("pickle_dump"))

    # Explicit reload proof: count classifier import (main lazy-imports reload_model from
    # src.ml.classifier) and boom the symbol if that module is already loaded.
    real_import = builtins.__import__

    def _tracked_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        if name == "src.ml.classifier" or name.endswith("ml.classifier"):
            hits["classifier_import"] += 1
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _tracked_import)

    clf = sys.modules.get("src.ml.classifier")
    if clf is not None:
        monkeypatch.setattr(clf, "reload_model", _boom("reload_model"), raising=False)

    return hits


def _assert_zero_downstream(hits: dict[str, int]) -> None:
    assert hits == {
        "get_active_learner": 0,
        "load_training_data": 0,
        "train_model": 0,
        "makedirs": 0,
        "pickle_dump": 0,
        "classifier_import": 0,
        "reload_model": 0,
    }


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        module.main()
    code = exc.value.code
    return 1 if code is None else int(code)


def test_gate_raises_blocks_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical path: check() raises GateBlocked → exit 1, zero training/export/write/reload."""
    hits = _spy_downstream(monkeypatch)
    # Real gate always raises; keep the real check so this is not a stubbed happy path.
    assert callable(gate_mod.check)

    code = _run_main(
        monkeypatch,
        [
            "finetune_from_feedback.py",
            "--force",
            "--allow-mock",
            "--output-dir",
            "/tmp/should-not-exist",
        ],
    )
    assert code == 1
    _assert_zero_downstream(hits)


def _is_gate_module_name(name: str) -> bool:
    """Both package and direct-CLI sibling import names must be denyable."""
    return name in ("scripts.eval_integrity_gate", "eval_integrity_gate")


def test_missing_gate_module_blocks_with_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import/module failure must fail closed — pre-seal code would fall through to train."""
    hits = _spy_downstream(monkeypatch)
    # Chain onto the spy's import tracker so classifier_import still counts.
    current_import = builtins.__import__

    def _deny(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        # Deny both package and sibling names (package vs direct-CLI import contexts).
        if _is_gate_module_name(name):
            raise ImportError("simulated missing evaluation-integrity gate")
        return current_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _deny)

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1
    _assert_zero_downstream(hits)


@pytest.mark.parametrize(
    "gate_name", ["scripts.eval_integrity_gate", "eval_integrity_gate"]
)
def test_missing_each_gate_import_name_blocks(
    monkeypatch: pytest.MonkeyPatch,
    gate_name: str,
) -> None:
    """Each deterministic import name, when denied, must fail closed with zero side effects."""
    hits = _spy_downstream(monkeypatch)
    current_import = builtins.__import__

    def _deny(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        if name == gate_name:
            raise ImportError(f"simulated missing {gate_name}")
        return current_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _deny)

    # Force the import context that uses this name.
    if gate_name == "eval_integrity_gate":
        monkeypatch.setattr(module, "__package__", None)
        scripts_dir = str(REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            monkeypatch.syspath_prepend(scripts_dir)
    else:
        monkeypatch.setattr(module, "__package__", "scripts")

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1
    _assert_zero_downstream(hits)


def test_unusable_gate_symbol_blocks_with_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module present but check missing/unusable must not fall through."""
    hits = _spy_downstream(monkeypatch)
    monkeypatch.delattr(gate_mod, "check", raising=False)

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1
    _assert_zero_downstream(hits)


def test_arbitrary_gate_exception_blocks_with_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any unexpected exception from check() is fail-closed, not a train path."""
    hits = _spy_downstream(monkeypatch)
    monkeypatch.setattr(
        gate_mod,
        "check",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated gate fault")),
    )

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1
    _assert_zero_downstream(hits)


def test_subverted_gate_return_is_invariant_breach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check() has no pass path; a return can only mean subversion → still exit 1."""
    hits = _spy_downstream(monkeypatch)
    monkeypatch.setattr(gate_mod, "check", lambda: None)

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1
    _assert_zero_downstream(hits)


def _force_gate_import_context(
    monkeypatch: pytest.MonkeyPatch,
    gate_name: str,
) -> None:
    """Select package vs sibling import path for deterministic gate-name tests."""
    if gate_name == "eval_integrity_gate":
        monkeypatch.setattr(module, "__package__", None)
        scripts_dir = str(REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            monkeypatch.syspath_prepend(scripts_dir)
    else:
        monkeypatch.setattr(module, "__package__", "scripts")


@pytest.mark.parametrize(
    "gate_name", ["scripts.eval_integrity_gate", "eval_integrity_gate"]
)
def test_import_time_systemexit_zero_is_remapped_to_one(
    monkeypatch: pytest.MonkeyPatch,
    gate_name: str,
) -> None:
    """Import raising SystemExit(0) must not succeed the process — remap to exit 1."""
    hits = _spy_downstream(monkeypatch)
    current_import = builtins.__import__

    def _exit_zero_on_gate(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        if name == gate_name:
            raise SystemExit(0)
        return current_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _exit_zero_on_gate)
    _force_gate_import_context(monkeypatch, gate_name)

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1, "import-time SystemExit(0) must become SystemExit(1)"
    _assert_zero_downstream(hits)


@pytest.mark.parametrize(
    "gate_name", ["scripts.eval_integrity_gate", "eval_integrity_gate"]
)
def test_import_time_keyboardinterrupt_is_remapped_to_one(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    gate_name: str,
) -> None:
    """Import raising KeyboardInterrupt must remap to exit 1 with type-only log."""
    hits = _spy_downstream(monkeypatch)
    current_import = builtins.__import__

    def _ki_on_gate(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ):
        if name == gate_name:
            raise KeyboardInterrupt()
        return current_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _ki_on_gate)
    _force_gate_import_context(monkeypatch, gate_name)

    with caplog.at_level("ERROR", logger="finetune"):
        code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])

    assert code == 1, "import-time KeyboardInterrupt must become SystemExit(1)"
    # Fixed type-only unavailable log — exception message content is never interpolated.
    assert (
        "evaluation-integrity gate unavailable (KeyboardInterrupt) — refusing to retrain"
        in caplog.text
    )
    _assert_zero_downstream(hits)


def test_check_time_systemexit_zero_is_remapped_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check() raising SystemExit(0) is BaseException — must remap to exit 1, not success."""
    hits = _spy_downstream(monkeypatch)
    monkeypatch.setattr(
        gate_mod,
        "check",
        lambda: (_ for _ in ()).throw(SystemExit(0)),
    )

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1, "check-time SystemExit(0) must become SystemExit(1)"
    _assert_zero_downstream(hits)


def test_check_time_keyboardinterrupt_is_remapped_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KeyboardInterrupt from check() must not escape raw — fail-closed as exit 1."""
    hits = _spy_downstream(monkeypatch)
    monkeypatch.setattr(
        gate_mod,
        "check",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1, "check-time KeyboardInterrupt must become SystemExit(1)"
    _assert_zero_downstream(hits)


def test_help_exits_one_from_gate_not_argparse(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--help must not short-circuit via argparse exit 0; gate is first and exits 1."""
    hits = _spy_downstream(monkeypatch)

    with caplog.at_level("ERROR", logger="finetune"):
        code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--help"])
    captured = capsys.readouterr()
    combined_stdio = (captured.out + captured.err).lower()
    log_text = caplog.text.lower()

    assert code == 1, "--help must exit 1 from the gate, not argparse's 0"
    # Argparse --help would print usage to stdio and exit 0 before any gate ran.
    assert "usage:" not in combined_stdio
    assert "retraining blocked" in log_text or "evaluation-integrity" in log_text
    _assert_zero_downstream(hits)


@pytest.mark.parametrize(
    "argv",
    [
        ["finetune_from_feedback.py", "--not-a-real-flag"],
        ["finetune_from_feedback.py", "--label-field", "not_a_valid_choice"],
        ["finetune_from_feedback.py", "unexpected-positional"],
    ],
)
def test_malformed_argv_exits_one_from_gate_not_argparse(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    argv: list[str],
) -> None:
    """Malformed/unknown argv must not die in argparse (exit 2); gate exits 1 first."""
    hits = _spy_downstream(monkeypatch)

    with caplog.at_level("ERROR", logger="finetune"):
        code = _run_main(monkeypatch, argv)
    captured = capsys.readouterr()
    combined_stdio = captured.out + captured.err
    log_text = caplog.text.lower()

    assert code == 1, "malformed argv must exit 1 from the gate, not argparse's 2"
    # Argparse error path prints these to stderr and exits 2 — must not appear.
    assert "unrecognized arguments" not in combined_stdio
    assert "invalid choice" not in combined_stdio
    assert "retraining blocked" in log_text or "evaluation-integrity" in log_text
    _assert_zero_downstream(hits)


@pytest.mark.parametrize(
    "argv",
    [
        ["finetune_from_feedback.py"],
        ["finetune_from_feedback.py", "--force"],
        ["finetune_from_feedback.py", "--allow-mock"],
        ["finetune_from_feedback.py", "--force", "--allow-mock"],
        ["finetune_from_feedback.py", "--output-dir", "models/should-not-write"],
    ],
)
def test_no_cli_arg_opens_the_gate(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    hits = _spy_downstream(monkeypatch)
    code = _run_main(monkeypatch, argv)
    assert code == 1
    _assert_zero_downstream(hits)


@pytest.mark.parametrize(
    "var,value",
    [
        ("EVAL_INTEGRITY_SKIP", "1"),
        ("SKIP_EVAL_INTEGRITY", "true"),
        ("FORCE_RETRAIN", "1"),
        ("ALLOW_PROMOTION", "yes"),
        ("EVAL_INTEGRITY_ARTIFACT", "/tmp/anything.json"),
    ],
)
def test_no_env_var_opens_the_gate(
    monkeypatch: pytest.MonkeyPatch,
    var: str,
    value: str,
) -> None:
    hits = _spy_downstream(monkeypatch)
    monkeypatch.setenv(var, value)
    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1
    _assert_zero_downstream(hits)


def test_reload_model_never_imported_or_called_on_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promotion sink must not be reached: no classifier import and no reload_model call."""
    hits = _spy_downstream(monkeypatch)
    code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])
    assert code == 1
    assert hits["classifier_import"] == 0
    assert hits["reload_model"] == 0
    _assert_zero_downstream(hits)


def test_gateblocked_path_secret_absent_from_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GateBlocked messages must not be interpolated — path-bearing text stays out of logs."""
    hits = _spy_downstream(monkeypatch)
    secret_path = "/secret/user/supplied/model/candidate.pkl"
    monkeypatch.setattr(
        gate_mod,
        "check",
        lambda: (_ for _ in ()).throw(
            gate_mod.GateBlocked(f"blocked for artifact at {secret_path}")
        ),
    )

    with caplog.at_level("ERROR", logger="finetune"):
        code = _run_main(monkeypatch, ["finetune_from_feedback.py", "--force"])

    assert code == 1
    assert secret_path not in caplog.text
    assert "/secret/" not in caplog.text
    assert "retraining blocked by the evaluation-integrity gate" in caplog.text
    _assert_zero_downstream(hits)


def _is_cli_gate_bootstrap_if(node: ast.AST) -> bool:
    """True for ``if __name__ == "__main__": _enforce_evaluation_integrity_gate()``."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    is_dunder_main = (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(
            isinstance(c, ast.Constant) and c.value == "__main__"
            for c in test.comparators
        )
    )
    if not is_dunder_main:
        return False
    for stmt in node.body:
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "_enforce_evaluation_integrity_gate"
        ):
            return True
    return False


def _is_sys_path_mutation(node: ast.AST) -> bool:
    """True for module-level ``sys.path.append(...)`` / ``sys.path.insert(...)``."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    if not isinstance(func, ast.Attribute) or func.attr not in ("append", "insert"):
        return False
    # sys.path.append / sys.path.insert
    if not isinstance(func.value, ast.Attribute) or func.value.attr != "path":
        return False
    return isinstance(func.value.value, ast.Name) and func.value.value.id == "sys"


def _assert_cli_bootstrap_before_heavy_side_effects(tree: ast.Module) -> None:
    """CLI gate bootstrap must precede numpy/src imports and sys.path mutations."""
    saw_cli_bootstrap_gate = False
    for node in tree.body:
        if _is_sys_path_mutation(node):
            assert (
                saw_cli_bootstrap_gate
            ), "sys.path.append/insert must not precede CLI gate bootstrap"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                mod = node.module or ""
                names = [mod] + [
                    f"{mod}.{a.name}" if mod else a.name for a in node.names
                ]
            for name in names:
                if (
                    name == "numpy"
                    or name.startswith("numpy.")
                    or name == "src"
                    or name.startswith("src.")
                ):
                    assert (
                        saw_cli_bootstrap_gate
                    ), f"import {name!r} must not precede CLI gate bootstrap"
        if _is_cli_gate_bootstrap_if(node):
            saw_cli_bootstrap_gate = True

    assert saw_cli_bootstrap_gate, (
        "CLI if __name__ == '__main__' bootstrap must call "
        "_enforce_evaluation_integrity_gate before numpy/src imports "
        "and sys.path mutations"
    )


def test_ast_main_body0_is_gate_and_cli_bootstrap_before_heavy_imports() -> None:
    """Structural lock: main().body[0] is the gate; CLI bootstrap precedes heavy work."""
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))

    main_fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    main_body = [
        n
        for n in main_fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    first = main_body[0]
    assert isinstance(first, ast.Expr) and isinstance(
        first.value, ast.Call
    ), "main() first statement must be a call"
    call = first.value
    assert isinstance(call.func, ast.Name), "main() must call a bare name first"
    assert call.func.id == "_enforce_evaluation_integrity_gate"
    assert call.args == [] and call.keywords == []

    _assert_cli_bootstrap_before_heavy_side_effects(tree)


def test_ast_sys_path_mutation_before_bootstrap_is_observed_red() -> None:
    """Mutation fixture: moving sys.path.append before CLI bootstrap must RED the AST lock."""
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH))
    body = list(tree.body)

    bootstrap_idxs = [i for i, n in enumerate(body) if _is_cli_gate_bootstrap_if(n)]
    path_idxs = [i for i, n in enumerate(body) if _is_sys_path_mutation(n)]
    assert bootstrap_idxs, "fixture requires a CLI gate bootstrap if-block"
    assert path_idxs, "fixture requires a module-level sys.path.append/insert"
    assert min(path_idxs) > min(
        bootstrap_idxs
    ), "source under test must currently place path mutation after bootstrap"

    # Temporarily move the first path mutation to immediately before the bootstrap.
    path_i = min(path_idxs)
    boot_i = min(bootstrap_idxs)
    mut = body.pop(path_i)
    # After pop, bootstrap index shifts left if path was after it (it was).
    body.insert(boot_i, mut)
    tree.body = body

    with pytest.raises(AssertionError, match="sys.path.append/insert"):
        _assert_cli_bootstrap_before_heavy_side_effects(tree)


@pytest.mark.parametrize(
    "argv_extra",
    [
        ["--help"],  # exact direct-CLI repro: gate before argparse
        ["--force"],
    ],
)
def test_cli_subprocess_hits_canonical_check_without_repo_pythonpath(
    argv_extra: list[str],
) -> None:
    """Direct CLI (no repo-root PYTHONPATH) must run check() and log fixed GateBlocked text.

    False-green guard: exit 1 from ModuleNotFoundError on scripts.eval_integrity_gate
    is NOT enough — stderr must show the canonical block message, not import failure.
    Sibling import (empty __package__) must resolve eval_integrity_gate from script dir.
    """
    env = {**os.environ}
    env.pop("PYTHONPATH", None)
    # Ensure no inherited repo-root path helps the package import by accident.
    env["PYTHONPATH"] = ""

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *argv_extra],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (proc.stdout + proc.stderr).lower()

    assert proc.returncode == 1
    assert "retraining blocked by the evaluation-integrity gate" in combined, (
        "must execute canonical check() → GateBlocked fixed log, not import failure; "
        f"argv={argv_extra!r} stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "modulenotfounderror" not in combined
    assert "gate unavailable" not in combined


def test_cli_subprocess_exits_before_poisoned_numpy_import(tmp_path: Path) -> None:
    """Real CLI must exit 1 before importing a poisoned numpy that would SystemExit(0).

    PYTHONPATH is only the fake package dir (no repo root) so sibling gate import is used.
    """
    sentinel = tmp_path / "heavy_import.sentinel"
    fake_pkg = tmp_path / "fake_pkgs"
    fake_pkg.mkdir()
    # Fake numpy: if imported, write sentinel and exit 0 (pre-gate success bypass).
    (fake_pkg / "numpy.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('numpy-imported\\n', encoding='utf-8')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    # Fake application package: same observed-red if src is imported too early.
    src_dir = fake_pkg / "src"
    src_dir.mkdir()
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    core_dir = src_dir / "core"
    core_dir.mkdir()
    (core_dir / "__init__.py").write_text("", encoding="utf-8")
    (core_dir / "active_learning.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('src-imported\\n', encoding='utf-8')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        # Only fake packages — no repo-root PYTHONPATH (sibling gate import must work).
        "PYTHONPATH": str(fake_pkg),
    }
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--force"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (proc.stdout + proc.stderr).lower()

    assert proc.returncode == 1, (
        f"CLI must exit 1 from the gate before heavy imports; "
        f"got {proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "retraining blocked by the evaluation-integrity gate" in combined
    assert "modulenotfounderror" not in combined
    assert (
        not sentinel.exists()
    ), "poisoned numpy/src must not run — sentinel proves a pre-gate heavy import"
