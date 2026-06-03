"""Shared fixtures.

Tests run against the ``seir-vaccine`` example CAS store. If it hasn't been
generated yet, we build it via the example Makefile when a ``camdl`` binary is
available; otherwise the dependent tests skip with a clear message.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from camdl_viewer import cas

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "seir-vaccine"
RESULTS = EXAMPLE / "results"


def _find_camdl() -> str | None:
    installed = Path.home() / ".local" / "bin" / "camdl"
    if installed.exists():
        return str(installed)
    return shutil.which("camdl")


def _generated() -> bool:
    return any(RESULTS.glob("**/run.json"))


@pytest.fixture(scope="session")
def results_dir() -> Path:
    if not _generated():
        camdl = _find_camdl()
        if camdl is None:
            pytest.skip("no camdl binary and no fixture; run `make -C examples/seir-vaccine`")
        subprocess.run(["make", "-C", str(EXAMPLE), f"CAMDL={camdl}"], check=True)
    if not _generated():
        pytest.skip("seir-vaccine fixture not generated")
    return RESULTS


@pytest.fixture(scope="session")
def store(results_dir: Path) -> cas.CasStore:
    return cas.discover_store(results_dir)
