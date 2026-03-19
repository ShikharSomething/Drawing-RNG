from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .adapters import AlgorithmAdapter
from .common import BenchmarkConfig, TestResult


def run_external_batteries(adapter: AlgorithmAdapter, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    sample_bytes = max(config.sample_bytes, 1_048_576)

    with tempfile.TemporaryDirectory(prefix="rng-bench-tools-") as tempdir:
        sample_path = Path(tempdir) / "sample.bin"
        sample_path.write_bytes(adapter.random_bytes(sample_bytes))

        results.append(_run_dieharder(sample_path))
        results.extend(_run_testu01(sample_path))
        results.append(_run_practrand(sample_path))
        results.append(_run_nist_sts(sample_path))

    return results


def _run_dieharder(sample_path: Path) -> TestResult:
    executable = shutil.which("dieharder")
    if not executable:
        return TestResult(
            category="external",
            name="Dieharder Full Battery",
            status="skip",
            rigor="external",
            summary="dieharder is not installed",
            metrics={"hint": "Install dieharder and rerun to execute `dieharder -a -g 201 -f sample.bin`."},
        )

    completed = subprocess.run(
        [executable, "-a", "-g", "201", "-f", str(sample_path)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    return TestResult(
        category="external",
        name="Dieharder Full Battery",
        status="ok" if completed.returncode == 0 else "warn",
        rigor="external",
        summary="Executed dieharder full battery",
        metrics={"returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-1000:]},
    )


def _run_testu01(sample_path: Path) -> list[TestResult]:
    env_commands = {
        "TestU01 SmallCrush": os.environ.get("TESTU01_SMALLCRUSH_CMD"),
        "TestU01 Crush": os.environ.get("TESTU01_CRUSH_CMD"),
        "TestU01 BigCrush": os.environ.get("TESTU01_BIGCRUSH_CMD"),
    }
    results: list[TestResult] = []
    for name, command in env_commands.items():
        if not command:
            results.append(
                TestResult(
                    category="external",
                    name=name,
                    status="skip",
                    rigor="external",
                    summary="Environment variable for this TestU01 command is not configured",
                    metrics={"hint": "Set the corresponding TESTU01_*_CMD environment variable and include `{sample}` in the command."},
                )
            )
            continue

        expanded = command.format(sample=str(sample_path))
        completed = subprocess.run(expanded, shell=True, capture_output=True, text=True, timeout=3600, check=False)
        results.append(
            TestResult(
                category="external",
                name=name,
                status="ok" if completed.returncode == 0 else "warn",
                rigor="external",
                summary=f"Executed {name}",
                metrics={"returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-1000:]},
            )
        )
    return results


def _run_practrand(sample_path: Path) -> TestResult:
    executable = shutil.which("RNG_test")
    if not executable:
        return TestResult(
            category="external",
            name="PractRand Stream Test",
            status="skip",
            rigor="external",
            summary="PractRand executable `RNG_test` is not installed",
            metrics={"hint": "Install PractRand and expose `RNG_test` on PATH to stream sample data into it."},
        )

    payload = sample_path.read_bytes()
    completed = subprocess.run(
        [executable, "stdin64"],
        input=payload,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    stdout_text = completed.stdout.decode("utf-8", errors="replace")
    stderr_text = completed.stderr.decode("utf-8", errors="replace")
    return TestResult(
        category="external",
        name="PractRand Stream Test",
        status="ok" if completed.returncode == 0 else "warn",
        rigor="external",
        summary="Executed PractRand",
        metrics={"returncode": completed.returncode, "stdout_tail": stdout_text[-2000:], "stderr_tail": stderr_text[-1000:]},
    )


def _run_nist_sts(sample_path: Path) -> TestResult:
    command = os.environ.get("NIST_STS_CMD")
    if not command:
        return TestResult(
            category="external",
            name="NIST SP 800-22",
            status="skip",
            rigor="external",
            summary="NIST STS command is not configured",
            metrics={"hint": "Set NIST_STS_CMD to a runnable command containing `{sample}`."},
        )

    expanded = command.format(sample=str(sample_path))
    completed = subprocess.run(expanded, shell=True, capture_output=True, text=True, timeout=3600, check=False)
    return TestResult(
        category="external",
        name="NIST SP 800-22",
        status="ok" if completed.returncode == 0 else "warn",
        rigor="external",
        summary="Executed NIST SP 800-22 command",
        metrics={"returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-1000:]},
    )
