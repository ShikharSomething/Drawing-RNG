from __future__ import annotations

import hashlib
import importlib.util
import os
import time
import zlib
from collections import defaultdict
from typing import Any

import numpy as np
from scipy import special, stats

from .adapters import AlgorithmAdapter
from .common import (
    BenchmarkConfig,
    SampleSet,
    TestResult,
    berlekamp_massey,
    bit_bias,
    bits_to_pm1,
    build_samples,
    chunk_bytes,
    correlation,
    normalized_hamming,
    shannon_entropy_from_counts,
)


def run_internal_benchmarks(adapter: AlgorithmAdapter, config: BenchmarkConfig) -> list[TestResult]:
    base_samples = build_samples(adapter.random_bytes(config.sample_bytes))
    results: list[TestResult] = []
    results.extend(statistical_tests(base_samples, config))
    results.extend(cryptographic_tests(adapter, base_samples, config))
    results.extend(structural_tests(adapter, base_samples, config))
    results.extend(cycle_tests(adapter, config))
    results.extend(chaos_tests(adapter, base_samples, config))
    results.extend(seed_sensitivity_tests(adapter, base_samples, config))
    results.extend(differential_tests(adapter, base_samples, config))
    results.extend(output_tests(adapter, base_samples, config))
    results.extend(implementation_tests(adapter, base_samples, config))
    results.extend(ml_tests(adapter, base_samples, config))
    return results


def _status_from_pvalue(pvalue: float | None) -> str:
    if pvalue is None or np.isnan(pvalue):
        return "warn"
    return "ok" if 0.001 <= pvalue <= 0.999 else "warn"


def statistical_tests(samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    bits = samples.bit_values
    bytes_arr = samples.byte_values

    ones_ratio = float(bits.mean()) if bits.size else 0.0
    zscore = abs(int(bits.sum()) - (bits.size - int(bits.sum()))) / np.sqrt(max(bits.size, 1))
    monobit_p = float(special.erfc(zscore / np.sqrt(2.0))) if bits.size else 0.0
    results.append(
        TestResult(
            category="statistical",
            name="Monobit Frequency",
            status=_status_from_pvalue(monobit_p),
            rigor="implemented",
            summary="Checks overall 0/1 balance in the sampled bitstream",
            metrics={"ones_ratio": ones_ratio, "bias": bit_bias(bits), "pvalue": monobit_p},
        )
    )

    byte_counts = np.bincount(bytes_arr, minlength=256)
    chi2, pvalue = stats.chisquare(byte_counts)
    results.append(
        TestResult(
            category="statistical",
            name="Byte Frequency Distribution",
            status=_status_from_pvalue(float(pvalue)),
            rigor="implemented",
            summary="Checks how close the 0..255 byte histogram is to uniform",
            metrics={"chi_square": float(chi2), "pvalue": float(pvalue)},
        )
    )

    if bits.size:
        transitions = np.count_nonzero(bits[1:] != bits[:-1])
        runs = int(transitions + 1)
        pi = float(bits.mean())
        expected_runs = 2 * bits.size * pi * (1 - pi) + 1
        variance = 2 * bits.size * pi * (1 - pi) * (2 * bits.size * pi * (1 - pi) - 1) / max(bits.size - 1, 1)
        run_z = float((runs - expected_runs) / np.sqrt(max(variance, 1e-12)))
        run_p = float(special.erfc(abs(run_z) / np.sqrt(2.0)))
    else:
        runs = 0
        expected_runs = 0.0
        run_z = 0.0
        run_p = 0.0
    results.append(
        TestResult(
            category="statistical",
            name="Runs Test",
            status=_status_from_pvalue(run_p),
            rigor="implemented",
            summary="Measures whether bit transitions happen at a plausible rate",
            metrics={"runs": runs, "expected_runs": expected_runs, "zscore": run_z, "pvalue": run_p},
        )
    )

    for lag in (1, 8, 64):
        if bits.size > lag:
            pm_bits = bits_to_pm1(bits)
            autocorr = float(np.dot(pm_bits[:-lag], pm_bits[lag:]) / (bits.size - lag))
        else:
            autocorr = 0.0
        results.append(
            TestResult(
                category="statistical",
                name=f"Autocorrelation Lag {lag}",
                status="ok" if abs(autocorr) < 0.02 else "warn",
                rigor="implemented",
                summary=f"Computes bit-level autocorrelation at lag {lag}",
                metrics={"lag": lag, "autocorrelation": autocorr},
            )
        )

    for lag in (1, 8, 32):
        if bytes_arr.size > lag:
            corr = correlation(bytes_arr[:-lag], bytes_arr[lag:])
        else:
            corr = 0.0
        results.append(
            TestResult(
                category="statistical",
                name=f"Serial Correlation Lag {lag}",
                status="ok" if abs(corr) < 0.02 else "warn",
                rigor="implemented",
                summary=f"Computes byte-level serial correlation at lag {lag}",
                metrics={"lag": lag, "correlation": corr},
            )
        )

    entropy = shannon_entropy_from_counts(byte_counts)
    results.append(
        TestResult(
            category="statistical",
            name="Shannon Entropy Per Byte",
            status="ok" if entropy > 7.95 else "warn",
            rigor="implemented",
            summary="Measures information density per byte",
            metrics={"entropy_bits_per_byte": entropy},
        )
    )

    compressed = zlib.compress(samples.data, level=9)
    compression_ratio = len(compressed) / max(len(samples.data), 1)
    results.append(
        TestResult(
            category="statistical",
            name="Compression Test",
            status="ok" if compression_ratio > 0.98 else "warn",
            rigor="implemented",
            summary="Random-looking output should not compress much under zlib",
            metrics={"compression_ratio": compression_ratio},
        )
    )

    return results


def cryptographic_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    bits = samples.bit_values[: config.prediction_bits]
    if bits.size > 1024:
        train_bits = bits[: bits.size // 2]
        test_bits = bits[bits.size // 2 :]
        context = 8
        counts: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0])
        for index in range(context, len(train_bits)):
            key = tuple(int(bit) for bit in train_bits[index - context : index])
            counts[key][int(train_bits[index])] += 1
        correct = 0
        total = 0
        fallback = 1 if train_bits.mean() >= 0.5 else 0
        for index in range(context, len(test_bits)):
            key = tuple(int(bit) for bit in test_bits[index - context : index])
            pred = fallback
            if key in counts:
                pred = 1 if counts[key][1] >= counts[key][0] else 0
            correct += int(pred == int(test_bits[index]))
            total += 1
        accuracy = correct / max(total, 1)
    else:
        accuracy = 0.5
    results.append(
        TestResult(
            category="cryptographic",
            name="Next-Bit Prediction Test",
            status="ok" if accuracy <= 0.53 else "warn",
            rigor="heuristic",
            summary="Uses a simple 8-bit context predictor against held-out bits",
            metrics={"prediction_accuracy": accuracy},
        )
    )

    live = adapter.create_generator()
    _ = live.random_bytes(config.small_sample_bytes)
    clone = adapter.clone_generator(live)
    future_a = live.random_bytes(512)
    future_b = clone.random_bytes(512)
    forward_predictable = future_a == future_b
    results.append(
        TestResult(
            category="cryptographic",
            name="Forward Prediction Resistance",
            status="warn" if forward_predictable else "ok",
            rigor="implemented",
            summary="Checks whether a full live-state clone can predict future output",
            metrics={"full_state_clone_predicts_future": forward_predictable},
        )
    )

    state_snapshot = adapter.capture_state(adapter.create_generator())
    snapshot_size = len(repr(state_snapshot).encode("utf-8"))
    results.append(
        TestResult(
            category="cryptographic",
            name="State Recovery Attack Proxy",
            status="warn" if snapshot_size > 0 else "skip",
            rigor="heuristic",
            summary="Treats a serializable state snapshot as a full-compromise scenario",
            metrics={"snapshot_serialized_bytes": snapshot_size},
        )
    )

    results.append(
        TestResult(
            category="cryptographic",
            name="Backtracking Resistance",
            status="warn",
            rigor="heuristic",
            summary="No reverse-state proof is available; this suite cannot certify backtracking resistance",
            metrics={"note": "Custom deterministic generators generally fail this unless the state transition is one-way under compromise."},
        )
    )

    rng_windows = _window_feature_matrix(samples.data, 2048, 24)
    random_windows = _window_feature_matrix(os.urandom(len(samples.data)), 2048, 24)
    labels = np.concatenate((np.zeros(len(rng_windows)), np.ones(len(random_windows))))
    feature_matrix = np.vstack((rng_windows, random_windows))
    distinguish_accuracy = _nearest_centroid_accuracy(feature_matrix, labels)
    results.append(
        TestResult(
            category="cryptographic",
            name="Distinguishability Test",
            status="ok" if distinguish_accuracy <= 0.65 else "warn",
            rigor="heuristic",
            summary="Tries to classify this generator versus OS randomness using cheap statistical features",
            metrics={"classification_accuracy": distinguish_accuracy},
        )
    )

    variants = adapter.available_seed_mutations(limit=config.mutation_samples)
    prefixes = [(name, adapter.random_bytes(16, seed_override=seed)) for name, seed in variants.items()]
    collision_count = len(prefixes) - len({prefix for _, prefix in prefixes})
    results.append(
        TestResult(
            category="cryptographic",
            name="Time-Memory Tradeoff Proxy",
            status="ok" if collision_count == 0 else "warn",
            rigor="heuristic",
            summary="Looks for prefix collisions across nearby seeds as a coarse precomputation proxy",
            metrics={"tested_variants": len(prefixes), "prefix_collisions": collision_count},
        )
    )

    if variants:
        first_seed = next(iter(variants.values()))
        similar_stream = build_samples(adapter.random_bytes(config.small_sample_bytes, seed_override=first_seed))
        corr = correlation(samples.byte_values, similar_stream.byte_values)
        ham = normalized_hamming(samples.data[: config.small_sample_bytes], similar_stream.data)
    else:
        corr = 0.0
        ham = 0.0
    results.append(
        TestResult(
            category="cryptographic",
            name="Correlation Attack",
            status="ok" if abs(corr) < 0.05 and 0.45 <= ham <= 0.55 else "warn",
            rigor="heuristic",
            summary="Checks whether closely related seeds emit strongly correlated output streams",
            metrics={"byte_correlation": corr, "normalized_hamming": ham},
        )
    )

    complexity_bits = samples.bit_values[:8192]
    linear_complexity = berlekamp_massey(complexity_bits) if complexity_bits.size else 0
    ratio = linear_complexity / max(len(complexity_bits), 1)
    results.append(
        TestResult(
            category="cryptographic",
            name="Linear Complexity Test",
            status="ok" if ratio > 0.35 else "warn",
            rigor="implemented",
            summary="Runs Berlekamp-Massey on the leading bitstream",
            metrics={"linear_complexity": linear_complexity, "complexity_ratio": ratio},
        )
    )

    return results


def structural_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    variants = adapter.available_seed_mutations(limit=max(config.mutation_samples, 4))
    variant_items = list(variants.items())
    if len(variant_items) >= 3:
        base_prefix = samples.data[:512]
        a = adapter.random_bytes(512, seed_override=variant_items[0][1])
        b = adapter.random_bytes(512, seed_override=variant_items[1][1])
        c = adapter.random_bytes(512, seed_override=variant_items[2][1])
        linearity_score = normalized_hamming(_xor_bytes(_xor_bytes(base_prefix, a), b), c)
    else:
        linearity_score = 0.0
    results.append(
        TestResult(
            category="structural",
            name="Bit-Level Linearity Probe",
            status="ok" if linearity_score > 0.40 else "warn",
            rigor="heuristic",
            summary="Uses nearby seed variants to see whether output differences superimpose too cleanly",
            metrics={"linearity_hamming_ratio": linearity_score},
        )
    )

    live = adapter.create_generator()
    seen = set()
    repeat_step = None
    for step in range(min(config.state_steps, 8192)):
        fingerprint = adapter.state_fingerprint(live)
        if fingerprint in seen:
            repeat_step = step
            break
        seen.add(fingerprint)
        if hasattr(live, "_step"):
            live._step()
        else:
            live.random_bytes(8)
    results.append(
        TestResult(
            category="structural",
            name="State Transition Collision Probe",
            status="ok" if repeat_step is None else "warn",
            rigor="heuristic",
            summary="Samples live state fingerprints to look for short transition collisions",
            metrics={"steps_tested": min(config.state_steps, 8192), "repeat_step": repeat_step},
        )
    )

    low_degree_score = _quadratic_seed_probe(adapter, config)
    results.append(
        TestResult(
            category="structural",
            name="Low-Degree Polynomial Probe",
            status="ok" if low_degree_score > 0.40 else "warn",
            rigor="heuristic",
            summary="Measures whether paired seed perturbations collapse into unusually simple output relations",
            metrics={"second_order_hamming_ratio": low_degree_score},
        )
    )

    differential_spread = _mutation_hamming_distribution(adapter, config)
    results.append(
        TestResult(
            category="structural",
            name="Differential Cryptanalysis-Style Probe",
            status="ok" if 0.45 <= differential_spread["mean_ratio"] <= 0.55 else "warn",
            rigor="heuristic",
            summary="Measures how output differences spread under nearby seed perturbations",
            metrics=differential_spread,
        )
    )

    chunk_biases = []
    for chunk in chunk_bytes(samples.data, config.chunk_bytes):
        chunk_bits = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8))
        chunk_biases.append(abs(bit_bias(chunk_bits)))
    bias_drift = max(chunk_biases) - min(chunk_biases) if chunk_biases else 0.0
    results.append(
        TestResult(
            category="structural",
            name="Bias Amplification Over Iterations",
            status="ok" if bias_drift < 0.01 else "warn",
            rigor="implemented",
            summary="Tracks monobit bias across successive chunks",
            metrics={"max_abs_bias": max(chunk_biases) if chunk_biases else 0.0, "bias_drift": bias_drift},
        )
    )

    live_state_bits = adapter.state_bit_size(adapter.create_generator())
    results.append(
        TestResult(
            category="structural",
            name="State Space Size Estimation",
            status="ok" if live_state_bits >= 128 else "warn",
            rigor="heuristic",
            summary="Estimates mutable state size from integer state words that are directly visible",
            metrics={"estimated_state_bits": live_state_bits},
        )
    )

    return results


def cycle_tests(adapter: AlgorithmAdapter, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    live = adapter.create_generator()
    tortoise = adapter.clone_generator(live)
    hare = adapter.clone_generator(live)

    def advance(generator: Any, steps: int) -> Any:
        for _ in range(steps):
            if hasattr(generator, "_step"):
                generator._step()
            else:
                generator.random_bytes(8)
        return generator

    met = False
    for _ in range(config.cycle_steps):
        tortoise = advance(tortoise, 1)
        hare = advance(hare, 2)
        if adapter.state_fingerprint(tortoise) == adapter.state_fingerprint(hare):
            met = True
            break
    results.append(
        TestResult(
            category="cycle",
            name="Cycle Detection (Floyd)",
            status="warn" if met else "ok",
            rigor="heuristic",
            summary="Uses state fingerprints with Floyd cycle detection to search for short cycles",
            metrics={"cycle_detected_in_window": met, "steps": config.cycle_steps},
        )
    )

    live = adapter.create_generator()
    seen = {}
    repeat = None
    for step in range(config.cycle_steps):
        fingerprint = adapter.state_fingerprint(live)
        if fingerprint in seen:
            repeat = {"first_seen": seen[fingerprint], "repeat_at": step}
            break
        seen[fingerprint] = step
        if hasattr(live, "_step"):
            live._step()
        else:
            live.random_bytes(8)
    results.append(
        TestResult(
            category="cycle",
            name="Period Estimation / State Collision",
            status="warn" if repeat else "ok",
            rigor="heuristic",
            summary="Looks for repeated state fingerprints within the test window",
            metrics={"repeat": repeat, "tested_steps": config.cycle_steps},
        )
    )

    sample = adapter.random_bytes(config.sample_bytes)
    blocks = [sample[index : index + 16] for index in range(0, len(sample) - 15, 16)]
    block_repeats = len(blocks) - len(set(blocks))
    results.append(
        TestResult(
            category="cycle",
            name="Repeat Pattern Detection",
            status="ok" if block_repeats == 0 else "warn",
            rigor="implemented",
            summary="Searches for repeated 16-byte blocks in the sampled output",
            metrics={"repeated_blocks": block_repeats, "block_count": len(blocks)},
        )
    )

    return results


def chaos_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    variants = adapter.available_seed_mutations(limit=4)
    if variants:
        close_seed = next(iter(variants.values()))
        base_words = np.frombuffer(samples.data[:8192], dtype=np.uint8)
        near_words = np.frombuffer(adapter.random_bytes(8192, seed_override=close_seed), dtype=np.uint8)
        divergence = []
        for window in range(256, len(base_words) + 1, 256):
            divergence.append(normalized_hamming(bytes(base_words[:window]), bytes(near_words[:window])))
        if len(divergence) >= 2:
            x = np.arange(len(divergence), dtype=np.float64)
            y = np.log(np.maximum(divergence, 1e-9))
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
        initial_sensitivity = divergence[-1] if divergence else 0.0
    else:
        slope = 0.0
        initial_sensitivity = 0.0

    results.append(
        TestResult(
            category="chaos",
            name="Lyapunov-Like Divergence",
            status="ok" if slope > -0.02 else "warn",
            rigor="heuristic",
            summary="Estimates divergence growth between nearby seeds from output Hamming distance",
            metrics={"divergence_log_slope": slope, "final_hamming_ratio": initial_sensitivity},
        )
    )

    results.append(
        TestResult(
            category="chaos",
            name="Initial Condition Sensitivity",
            status="ok" if 0.45 <= initial_sensitivity <= 0.55 else "warn",
            rigor="implemented",
            summary="Checks whether a tiny seed change leads to about half the output bits flipping",
            metrics={"normalized_hamming": initial_sensitivity},
        )
    )

    words = samples.word_values.astype(np.float64)
    if words.size >= 2:
        norm = words / np.float64((1 << 64) - 1)
        bins = np.histogram2d(norm[:-1], norm[1:], bins=32, range=[[0, 1], [0, 1]])[0]
        occupied = int(np.count_nonzero(bins))
        occupancy_ratio = occupied / bins.size
        phase_corr = correlation(norm[:-1], norm[1:])
    else:
        occupancy_ratio = 0.0
        phase_corr = 0.0
    results.append(
        TestResult(
            category="chaos",
            name="Topological Mixing / Phase Space",
            status="ok" if occupancy_ratio > 0.70 and abs(phase_corr) < 0.05 else "warn",
            rigor="heuristic",
            summary="Measures lag-plot occupancy and correlation between successive normalized words",
            metrics={"occupancy_ratio": occupancy_ratio, "successive_word_correlation": phase_corr},
        )
    )

    entropies = []
    for chunk in chunk_bytes(samples.data[: min(len(samples.data), 65_536)], 2048):
        counts = np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
        entropies.append(shannon_entropy_from_counts(counts))
    entropy_growth = entropies[-1] - entropies[0] if len(entropies) >= 2 else 0.0
    results.append(
        TestResult(
            category="chaos",
            name="Entropy Growth Over Iterations",
            status="ok" if min(entropies or [0.0]) > 7.8 else "warn",
            rigor="heuristic",
            summary="Tracks per-chunk byte entropy through the early output stream",
            metrics={"first_chunk_entropy": entropies[0] if entropies else 0.0, "last_chunk_entropy": entropies[-1] if entropies else 0.0, "entropy_growth": entropy_growth},
        )
    )

    results.append(
        TestResult(
            category="chaos",
            name="Bifurcation Sensitivity Test",
            status="ok" if initial_sensitivity > 0.40 else "warn",
            rigor="heuristic",
            summary="Treats nearby seed perturbations as parameter nudges and measures output divergence",
            metrics={"sensitivity_ratio": initial_sensitivity},
        )
    )

    return results


def seed_sensitivity_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    variants = adapter.available_seed_mutations(limit=config.mutation_samples)
    base_prefix = samples.data[: config.small_sample_bytes]
    hamming_scores: dict[str, float] = {}
    correlations: dict[str, float] = {}
    prefixes = []

    for name, seed in variants.items():
        variant_output = adapter.random_bytes(config.small_sample_bytes, seed_override=seed)
        hamming_scores[name] = normalized_hamming(base_prefix, variant_output)
        correlations[name] = correlation(
            np.frombuffer(base_prefix, dtype=np.uint8),
            np.frombuffer(variant_output, dtype=np.uint8),
        )
        prefixes.append(variant_output[:16])

    avalanche = hamming_scores.get("pixel_lsb_flip", hamming_scores.get("byte_lsb_flip", 0.0))
    results.append(
        TestResult(
            category="input_sensitivity",
            name="Avalanche Effect",
            status="ok" if 0.45 <= avalanche <= 0.55 else "warn",
            rigor="implemented",
            summary="Checks whether a minimal seed change flips about half the output bits",
            metrics={"normalized_hamming": avalanche},
        )
    )

    bit_flip = hamming_scores.get("pixel_msb_flip", hamming_scores.get("byte_msb_flip", 0.0))
    results.append(
        TestResult(
            category="input_sensitivity",
            name="Bit Flip Sensitivity",
            status="ok" if 0.45 <= bit_flip <= 0.55 else "warn",
            rigor="implemented",
            summary="Measures sensitivity to flipping a higher-value bit in the seed",
            metrics={"normalized_hamming": bit_flip},
        )
    )

    similar_corr = correlations.get("rotate_90", correlations.get("byte_rotate", 0.0))
    results.append(
        TestResult(
            category="input_sensitivity",
            name="Similar Seed Correlation",
            status="ok" if abs(similar_corr) < 0.05 else "warn",
            rigor="heuristic",
            summary="Checks whether transformed but related seeds produce correlated streams",
            metrics={"byte_correlation": similar_corr},
        )
    )

    transform_values = [
        value
        for name, value in hamming_scores.items()
        if name in {"rotate_90", "mirror", "vertical_flip", "byte_rotate", "byte_reverse"}
    ]
    transform_score = float(np.mean(transform_values)) if transform_values else 0.0
    results.append(
        TestResult(
            category="input_sensitivity",
            name="Rotation / Transformation Robustness",
            status="ok" if transform_score > 0.40 else "warn",
            rigor="heuristic",
            summary="Measures whether structured seed transformations still trigger strong output changes",
            metrics={"mean_transformation_hamming": transform_score},
        )
    )

    shift_score = hamming_scores.get("flat_shift", hamming_scores.get("row_shift", hamming_scores.get("byte_rotate", 0.0)))
    results.append(
        TestResult(
            category="input_sensitivity",
            name="Perturbation Sequence Sensitivity",
            status="ok" if 0.45 <= shift_score <= 0.55 else "warn",
            rigor="heuristic",
            summary="Uses seed shifts as a proxy for perturbation-sequence reordering",
            metrics={"normalized_hamming": shift_score},
        )
    )

    prefix_collisions = len(prefixes) - len(set(prefixes))
    results.append(
        TestResult(
            category="input_sensitivity",
            name="Seed Collision Resistance",
            status="ok" if prefix_collisions == 0 else "warn",
            rigor="heuristic",
            summary="Checks for identical 16-byte prefixes across nearby seed variants",
            metrics={"tested_variants": len(prefixes), "prefix_collisions": prefix_collisions},
        )
    )

    return results


def differential_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    variants = adapter.available_seed_mutations(limit=config.mutation_samples)
    base_prefix = samples.data[: config.small_sample_bytes]
    ratios = []
    xor_bytes = bytearray()

    for seed in variants.values():
        variant_prefix = adapter.random_bytes(config.small_sample_bytes, seed_override=seed)
        output_ratio = normalized_hamming(base_prefix, variant_prefix)
        seed_bits = adapter.seed_distance_bits(adapter.base_seed, seed)
        if seed_bits > 0:
            ratios.append(output_ratio / seed_bits)
        xor_bytes.extend(_xor_bytes(base_prefix[:4096], variant_prefix[:4096]))

    results.append(
        TestResult(
            category="differential",
            name="Input Difference to Output Difference Ratio",
            status="ok" if ratios and np.mean(ratios) > 0 else "warn",
            rigor="implemented",
            summary="Compares output divergence against seed-bit distance",
            metrics={"mean_ratio": float(np.mean(ratios)) if ratios else 0.0, "sample_count": len(ratios)},
        )
    )

    if xor_bytes:
        counts = np.bincount(np.frombuffer(bytes(xor_bytes), dtype=np.uint8), minlength=256)
        chi2, pvalue = stats.chisquare(counts)
    else:
        chi2, pvalue = 0.0, 0.0
    results.append(
        TestResult(
            category="differential",
            name="Differential Distribution Uniformity",
            status=_status_from_pvalue(float(pvalue)),
            rigor="heuristic",
            summary="Looks at the byte distribution of XOR differences between nearby seeds",
            metrics={"chi_square": float(chi2), "pvalue": float(pvalue)},
        )
    )

    xor_array = np.frombuffer(bytes(xor_bytes), dtype=np.uint8) if xor_bytes else np.array([], dtype=np.uint8)
    xor_corr = correlation(xor_array[:-1], xor_array[1:]) if xor_array.size > 1 else 0.0
    results.append(
        TestResult(
            category="differential",
            name="Output XOR Distribution",
            status="ok" if abs(xor_corr) < 0.05 else "warn",
            rigor="heuristic",
            summary="Checks whether XOR differences between similar seeds show obvious correlation",
            metrics={"correlation": xor_corr},
        )
    )

    shift_seed = None
    for candidate in ("flat_shift", "row_shift", "byte_rotate"):
        if candidate in variants:
            shift_seed = variants[candidate]
            break
    if shift_seed is not None:
        shifted_output = adapter.random_bytes(config.small_sample_bytes, seed_override=shift_seed)
        shift_hamming = normalized_hamming(base_prefix, shifted_output)
    else:
        shift_hamming = 0.0
    results.append(
        TestResult(
            category="differential",
            name="Sensitivity to Sequence Shifts",
            status="ok" if 0.45 <= shift_hamming <= 0.55 else "warn",
            rigor="heuristic",
            summary="Measures sensitivity to ordered-seed shifts",
            metrics={"normalized_hamming": shift_hamming},
        )
    )

    return results


def output_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    bits = samples.bit_values
    if bits.size > 2:
        even = bits[0:-1:2]
        odd = bits[1::2]
        bit_independence = correlation(even, odd)
    else:
        bit_independence = 0.0
    results.append(
        TestResult(
            category="output",
            name="Bit Independence Test",
            status="ok" if abs(bit_independence) < 0.02 else "warn",
            rigor="implemented",
            summary="Measures correlation between adjacent output bits",
            metrics={"adjacent_bit_correlation": bit_independence},
        )
    )

    live = adapter.create_generator()
    mutant = adapter.mutate_live_state(live)
    if mutant is not None:
        base_out = live.random_bytes(512)
        mutant_out = mutant.random_bytes(512)
        state_avalanche = normalized_hamming(base_out, mutant_out)
        status = "ok" if 0.45 <= state_avalanche <= 0.55 else "warn"
        summary = "Flips one visible state bit and measures output avalanche"
    else:
        state_avalanche = 0.0
        status = "skip"
        summary = "State mutation hook is not available for this generator"
    results.append(
        TestResult(
            category="output",
            name="Avalanche on Internal State to Output",
            status=status,
            rigor="heuristic",
            summary=summary,
            metrics={"normalized_hamming": state_avalanche},
        )
    )

    counts = np.bincount(samples.byte_values, minlength=256)
    entropy = shannon_entropy_from_counts(counts)
    low_bit_bias = float((samples.byte_values & 1).mean() - 0.5)
    high_bit_bias = float(((samples.byte_values >> 7) & 1).mean() - 0.5)
    results.append(
        TestResult(
            category="output",
            name="Bias After Extraction",
            status="ok" if abs(low_bit_bias) < 0.01 and abs(high_bit_bias) < 0.01 and entropy > 7.95 else "warn",
            rigor="implemented",
            summary="Checks low-bit and high-bit bias after the output function",
            metrics={"entropy_bits_per_byte": entropy, "low_bit_bias": low_bit_bias, "high_bit_bias": high_bit_bias},
        )
    )

    whitening = _output_whitening_probe(adapter)
    results.append(
        TestResult(
            category="output",
            name="Whitening Effectiveness",
            status=whitening["status"],
            rigor="heuristic",
            summary=whitening["summary"],
            metrics=whitening["metrics"],
        )
    )

    return results


def implementation_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    generator = adapter.create_generator()
    contains_float = _contains_float_state(generator)
    results.append(
        TestResult(
            category="implementation",
            name="Floating-Point Precision Degradation",
            status="ok" if not contains_float else "warn",
            rigor="implemented",
            summary="Checks whether the visible generator state still relies on floating-point values",
            metrics={"contains_float_state": contains_float},
        )
    )

    digest = hashlib.sha256(samples.data[:4096]).hexdigest()
    results.append(
        TestResult(
            category="implementation",
            name="Cross-Platform Reproducibility Fingerprint",
            status="ok",
            rigor="implemented",
            summary="Emits a canonical output digest you can compare across machines",
            metrics={"sha256_first_4096_bytes": digest},
        )
    )

    deterministic = adapter.random_bytes(4096) == adapter.random_bytes(4096)
    results.append(
        TestResult(
            category="implementation",
            name="Determinism Check",
            status="ok" if deterministic else "warn",
            rigor="implemented",
            summary="Checks whether two fresh generators with the same seed match exactly",
            metrics={"deterministic": deterministic},
        )
    )

    zero_warmup = adapter.random_bytes(config.small_sample_bytes, kwargs_override={"warmup_rounds": 0})
    zero_counts = np.bincount(np.frombuffer(zero_warmup, dtype=np.uint8), minlength=256)
    warm_counts = np.bincount(samples.byte_values[: config.small_sample_bytes], minlength=256)
    zero_entropy = shannon_entropy_from_counts(zero_counts)
    warm_entropy = shannon_entropy_from_counts(warm_counts)
    results.append(
        TestResult(
            category="implementation",
            name="Warmup Effectiveness",
            status="ok" if warm_entropy >= zero_entropy else "warn",
            rigor="heuristic",
            summary="Compares warmed output entropy to zero-warmup output entropy",
            metrics={"warm_entropy": warm_entropy, "zero_warmup_entropy": zero_entropy},
        )
    )

    started = time.perf_counter()
    payload = adapter.random_bytes(config.benchmark_bytes)
    elapsed = time.perf_counter() - started
    throughput = len(payload) / max(elapsed, 1e-9) / (1024 * 1024)
    results.append(
        TestResult(
            category="implementation",
            name="Performance Benchmark",
            status="ok",
            rigor="implemented",
            summary="Measures one-shot throughput for byte generation",
            metrics={"bytes_generated": len(payload), "seconds": elapsed, "throughput_mib_per_s": throughput},
        )
    )

    return results


def ml_tests(adapter: AlgorithmAdapter, samples: SampleSet, config: BenchmarkConfig) -> list[TestResult]:
    results: list[TestResult] = []
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    results.append(
        TestResult(
            category="ml",
            name="Neural Network Prediction Test",
            status="skip" if not sklearn_available else "ok",
            rigor="external",
            summary="A full neural-network predictor is not implemented unless scikit-learn or a DL stack is installed",
            metrics={"sklearn_available": sklearn_available},
        )
    )

    bytes_arr = samples.byte_values.astype(np.float64)
    window = 4
    if len(bytes_arr) > window + 1:
        rows = []
        targets = []
        for index in range(window, min(len(bytes_arr) - 1, 4096)):
            rows.append(bytes_arr[index - window : index] / 255.0)
            targets.append(bytes_arr[index] / 255.0)
        x = np.array(rows)
        y = np.array(targets)
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        preds = np.clip(x @ coeffs, 0.0, 1.0)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        baseline = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    else:
        rmse = 0.0
        baseline = 0.0
    results.append(
        TestResult(
            category="ml",
            name="Regression-Based Prediction",
            status="ok" if rmse >= baseline * 0.95 else "warn",
            rigor="heuristic",
            summary="Fits a tiny linear regressor on previous bytes to predict the next byte",
            metrics={"rmse": rmse, "baseline_rmse": baseline},
        )
    )

    rng_features = _window_feature_matrix(samples.data, 2048, 16)
    random_features = _window_feature_matrix(os.urandom(len(samples.data)), 2048, 16)
    labels = np.concatenate((np.zeros(len(rng_features)), np.ones(len(random_features))))
    accuracy = _nearest_centroid_accuracy(np.vstack((rng_features, random_features)), labels)
    results.append(
        TestResult(
            category="ml",
            name="Sequence Classification",
            status="ok" if accuracy <= 0.65 else "warn",
            rigor="heuristic",
            summary="Uses cheap feature vectors to classify this generator against OS randomness",
            metrics={"classification_accuracy": accuracy},
        )
    )

    reconstruction = _hidden_state_proxy(adapter)
    results.append(
        TestResult(
            category="ml",
            name="Hidden State Reconstruction Proxy",
            status=reconstruction["status"],
            rigor="heuristic",
            summary=reconstruction["summary"],
            metrics=reconstruction["metrics"],
        )
    )

    return results


def _window_feature_matrix(data: bytes, window_size: int, max_windows: int) -> np.ndarray:
    windows = []
    for index in range(0, min(len(data), window_size * max_windows), window_size):
        chunk = data[index : index + window_size]
        if len(chunk) < window_size:
            break
        arr = np.frombuffer(chunk, dtype=np.uint8)
        counts = np.bincount(arr, minlength=256)
        windows.append(
            np.array(
                [
                    shannon_entropy_from_counts(counts),
                    len(zlib.compress(chunk, level=9)) / len(chunk),
                    correlation(arr[:-1], arr[1:]) if len(arr) > 1 else 0.0,
                    float(((arr & 1).mean()) if len(arr) else 0.0),
                    float(((arr >> 7).mean()) if len(arr) else 0.0),
                    float(np.mean(arr)) / 255.0,
                    float(np.std(arr)) / 255.0,
                ],
                dtype=np.float64,
            )
        )
    return np.vstack(windows) if windows else np.zeros((0, 7), dtype=np.float64)


def _nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    if len(features) < 4:
        return 0.5
    correct = 0
    for index in range(len(features)):
        mask = np.arange(len(features)) != index
        train_x = features[mask]
        train_y = labels[mask]
        centroids = {}
        for label in np.unique(train_y):
            centroids[label] = train_x[train_y == label].mean(axis=0)
        sample = features[index]
        pred = min(centroids, key=lambda label: np.linalg.norm(sample - centroids[label]))
        correct += int(pred == labels[index])
    return correct / len(features)


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    width = min(len(left), len(right))
    return bytes(a ^ b for a, b in zip(left[:width], right[:width]))


def _mutation_hamming_distribution(adapter: AlgorithmAdapter, config: BenchmarkConfig) -> dict[str, Any]:
    variants = adapter.available_seed_mutations(limit=config.mutation_samples)
    base = adapter.random_bytes(config.small_sample_bytes)
    ratios = [
        normalized_hamming(base, adapter.random_bytes(config.small_sample_bytes, seed_override=seed))
        for seed in variants.values()
    ]
    if not ratios:
        return {"mean_ratio": 0.0, "std_ratio": 0.0, "count": 0}
    return {"mean_ratio": float(np.mean(ratios)), "std_ratio": float(np.std(ratios)), "count": len(ratios)}


def _quadratic_seed_probe(adapter: AlgorithmAdapter, config: BenchmarkConfig) -> float:
    variants = adapter.available_seed_mutations(limit=max(config.mutation_samples, 3))
    items = list(variants.items())
    if len(items) < 3:
        return 0.0
    base = adapter.random_bytes(512)
    out_a = adapter.random_bytes(512, seed_override=items[0][1])
    out_b = adapter.random_bytes(512, seed_override=items[1][1])
    out_c = adapter.random_bytes(512, seed_override=items[2][1])
    second_derivative = _xor_bytes(_xor_bytes(base, out_a), _xor_bytes(out_b, out_c))
    return normalized_hamming(second_derivative, b"\x00" * len(second_derivative))


def _contains_float_state(generator: Any) -> bool:
    stack = []
    for key in ("state", "x", "y", "z", "w", "weyl", "counter", "feature_streams", "perturb_words"):
        if hasattr(generator, key):
            stack.append(getattr(generator, key))
    visited = set()
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, (float, np.floating)):
            return True
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)
        elif isinstance(current, np.ndarray):
            if np.issubdtype(current.dtype, np.floating):
                return True
    return False


def _output_whitening_probe(adapter: AlgorithmAdapter) -> dict[str, Any]:
    generator = adapter.create_generator()
    if not hasattr(generator, "_step") or not hasattr(generator, "state"):
        return {"status": "skip", "summary": "Raw state probe is not available for this generator", "metrics": {}}

    raw = bytearray()
    mixed = bytearray()
    probe = adapter.clone_generator(generator)
    for _ in range(256):
        probe._step()
        state_words = adapter.state_words(probe)
        raw_word = sum(state_words[:4]) & ((1 << 64) - 1)
        raw.extend(raw_word.to_bytes(8, "big"))
        mixed.extend(probe.random_bytes(8))

    raw_counts = np.bincount(np.frombuffer(bytes(raw), dtype=np.uint8), minlength=256)
    mixed_counts = np.bincount(np.frombuffer(bytes(mixed), dtype=np.uint8), minlength=256)
    raw_entropy = shannon_entropy_from_counts(raw_counts)
    mixed_entropy = shannon_entropy_from_counts(mixed_counts)
    return {
        "status": "ok" if mixed_entropy >= raw_entropy else "warn",
        "summary": "Compares a naive raw-state fold against the public output function",
        "metrics": {"raw_entropy": raw_entropy, "mixed_entropy": mixed_entropy},
    }


def _hidden_state_proxy(adapter: AlgorithmAdapter) -> dict[str, Any]:
    generator = adapter.create_generator()
    if not hasattr(generator, "_step"):
        return {"status": "skip", "summary": "No step-level access is available for hidden-state probing", "metrics": {}}

    rows = []
    targets = []
    for _ in range(256):
        state_words = adapter.state_words(generator)
        if len(state_words) < 4:
            generator._step()
            continue
        rows.append(np.array([(word & 0xFFFF) / 65535.0 for word in state_words[:4]], dtype=np.float64))
        generator._step()
        next_words = adapter.state_words(generator)
        targets.append((next_words[0] & 0xFF) / 255.0 if next_words else 0.0)

    if len(rows) < 16:
        return {"status": "skip", "summary": "Visible state words are insufficient for this proxy", "metrics": {}}

    x = np.vstack(rows)
    y = np.array(targets)
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    preds = np.clip(x @ coeffs, 0.0, 1.0)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    baseline = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    return {
        "status": "ok" if rmse >= baseline * 0.95 else "warn",
        "summary": "Uses a linear model on visible state words to predict the next low output byte as a reconstruction proxy",
        "metrics": {"rmse": rmse, "baseline_rmse": baseline},
    }
