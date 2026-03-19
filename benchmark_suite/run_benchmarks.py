from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import AlgorithmAdapter
from .common import BenchmarkConfig
from .external_tools import run_external_batteries
from .tests import run_internal_benchmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a custom RNG / crypto-like generator.")
    parser.add_argument("--factory", default="main:DrawChaoticRNG", help="Import path to the class or factory, e.g. main:DrawChaoticRNG")
    parser.add_argument("--seed", default="img.png", help="Seed input. For image seed type, this is an image path.")
    parser.add_argument("--seed-type", choices=("image", "bytes", "text"), default="image", help="Type of seed expected by the target factory")
    parser.add_argument("--factory-kwargs", default=None, help="JSON object of keyword arguments passed to the target factory")
    parser.add_argument("--sample-bytes", type=int, default=262_144, help="Primary sample size for internal tests")
    parser.add_argument("--small-sample-bytes", type=int, default=32_768, help="Smaller sample size for sensitivity and attack probes")
    parser.add_argument("--chunk-bytes", type=int, default=4_096, help="Chunk size used by chunked statistical probes")
    parser.add_argument("--mutation-samples", type=int, default=8, help="How many deterministic seed mutations to test")
    parser.add_argument("--cycle-steps", type=int, default=20_000, help="How many transition steps to explore in cycle tests")
    parser.add_argument("--state-steps", type=int, default=4_096, help="How many state probes to take for structure checks")
    parser.add_argument("--prediction-bits", type=int, default=65_536, help="Bit budget used by simple prediction attacks")
    parser.add_argument("--benchmark-bytes", type=int, default=1_048_576, help="Output size used for throughput benchmarking")
    parser.add_argument("--include-external", action="store_true", help="Also run external batteries such as dieharder and PractRand if available")
    parser.add_argument("--output", default=None, help="Optional JSON file path for the full report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        sample_bytes=args.sample_bytes,
        small_sample_bytes=args.small_sample_bytes,
        chunk_bytes=args.chunk_bytes,
        mutation_samples=args.mutation_samples,
        cycle_steps=args.cycle_steps,
        state_steps=args.state_steps,
        prediction_bits=args.prediction_bits,
        benchmark_bytes=args.benchmark_bytes,
    )
    adapter = AlgorithmAdapter(
        factory_path=args.factory,
        seed=args.seed,
        seed_type=args.seed_type,
        factory_kwargs=AlgorithmAdapter.parse_factory_kwargs(args.factory_kwargs),
    )

    try:
        results = run_internal_benchmarks(adapter, config)
        if args.include_external:
            results.extend(run_external_batteries(adapter, config))

        report = {
            "factory": args.factory,
            "seed": args.seed,
            "seed_type": args.seed_type,
            "config": config.__dict__,
            "results": [result.to_dict() for result in results],
        }

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        _print_summary(report)
    finally:
        adapter.close()


def _print_summary(report: dict) -> None:
    print(f"Target: {report['factory']}  seed={report['seed']}  seed_type={report['seed_type']}")
    print(f"Total tests: {len(report['results'])}")

    counts: dict[str, int] = {}
    for result in report["results"]:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print("Status counts:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))

    for result in report["results"]:
        print(f"[{result['status']:>4}] {result['category']} :: {result['name']}")
        print(f"       {result['summary']}")
        if result["metrics"]:
            trimmed = json.dumps(result["metrics"], default=str)
            if len(trimmed) > 180:
                trimmed = trimmed[:177] + "..."
            print(f"       metrics={trimmed}")


if __name__ == "__main__":
    main()
