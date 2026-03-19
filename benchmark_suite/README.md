# Benchmark Suite

This folder contains a reusable Python benchmark harness for custom RNG and crypto-like generators.

## Default usage

Run it against the current image-seeded generator:

```powershell
.\.venv\Scripts\python.exe -m benchmark_suite.run_benchmarks --factory main:DrawChaoticRNG --seed img.png --seed-type image --output benchmark_report.json
```

## Run external batteries too

```powershell
.\.venv\Scripts\python.exe -m benchmark_suite.run_benchmarks --factory main:DrawChaoticRNG --seed img.png --seed-type image --include-external
```

External tools are optional. The script will skip them cleanly unless they are installed or configured:

- `dieharder` for the Dieharder battery
- `RNG_test` for PractRand
- `NIST_STS_CMD` environment variable for NIST SP 800-22
- `TESTU01_SMALLCRUSH_CMD`, `TESTU01_CRUSH_CMD`, `TESTU01_BIGCRUSH_CMD` for TestU01 wrappers

Each configured command should include `{sample}` where the temporary sample file path should be inserted.

## Benchmark target contract

The target factory should be importable via `module_name:callable_name` and should return an object with:

- `random_bytes(n: int) -> bytes`

The current suite also uses optional internals when available:

- `_step()` for step-level probes
- `state`, `x`, `y`, `z`, `w`, `weyl`, `counter` for state-oriented tests

## Notes

- Some tests are fully implemented.
- Some are heuristic probes rather than formal cryptanalysis.
- Heavy batteries like PractRand, Dieharder, and TestU01 are wrapped but not bundled.
- This suite is best used as a benchmarking aid, not as a proof of security.
