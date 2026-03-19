from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkConfig:
    sample_bytes: int = 262_144
    small_sample_bytes: int = 32_768
    chunk_bytes: int = 4_096
    mutation_samples: int = 8
    cycle_steps: int = 20_000
    state_steps: int = 4_096
    prediction_bits: int = 65_536
    benchmark_bytes: int = 1_048_576
    warmup_rounds: int = 256


@dataclass
class SampleSet:
    data: bytes
    byte_values: np.ndarray
    bit_values: np.ndarray
    word_values: np.ndarray


@dataclass
class TestResult:
    category: str
    name: str
    status: str
    rigor: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_samples(data: bytes) -> SampleSet:
    byte_values = np.frombuffer(data, dtype=np.uint8)
    bit_values = np.unpackbits(byte_values)
    usable = len(data) - (len(data) % 8)
    if usable > 0:
        word_values = np.frombuffer(data[:usable], dtype=">u8").astype(np.uint64)
    else:
        word_values = np.array([], dtype=np.uint64)
    return SampleSet(data=data, byte_values=byte_values, bit_values=bit_values, word_values=word_values)


def chunk_bytes(data: bytes, chunk_size: int) -> list[bytes]:
    return [data[index : index + chunk_size] for index in range(0, len(data), chunk_size) if data[index : index + chunk_size]]


def bits_to_pm1(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.int8) * 2 - 1


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def hamming_distance(left: bytes, right: bytes) -> int:
    left_bits = np.unpackbits(np.frombuffer(left, dtype=np.uint8))
    right_bits = np.unpackbits(np.frombuffer(right, dtype=np.uint8))
    width = min(len(left_bits), len(right_bits))
    return int(np.count_nonzero(left_bits[:width] != right_bits[:width]))


def normalized_hamming(left: bytes, right: bytes) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    return hamming_distance(left[:width], right[:width]) / (width * 8.0)


def correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    width = min(len(values_a), len(values_b))
    if width < 2:
        return 0.0
    a = values_a[:width].astype(np.float64)
    b = values_b[:width].astype(np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def bit_bias(bits: np.ndarray) -> float:
    if bits.size == 0:
        return 0.0
    return float(bits.mean() - 0.5)


def monobit_zscore(bits: np.ndarray) -> float:
    n = bits.size
    if n == 0:
        return 0.0
    s_obs = abs(int(bits.sum()) - (n - int(bits.sum())))
    return float(s_obs / math.sqrt(n))


def berlekamp_massey(bits: np.ndarray) -> int:
    sequence = bits.astype(np.uint8).tolist()
    n = len(sequence)
    c = [0] * n
    b = [0] * n
    c[0] = 1
    b[0] = 1
    length = 0
    m = -1

    for current in range(n):
        discrepancy = sequence[current]
        for index in range(1, length + 1):
            discrepancy ^= c[index] & sequence[current - index]
        if discrepancy == 0:
            continue
        temp = c[:]
        shift = current - m
        for index in range(n - shift):
            c[index + shift] ^= b[index]
        if 2 * length <= current:
            length = current + 1 - length
            m = current
            b = temp

    return length


def rank_over_gf2(bit_rows: np.ndarray) -> int:
    matrix = bit_rows.copy().astype(np.uint8)
    rows, cols = matrix.shape
    rank = 0
    col = 0
    while rank < rows and col < cols:
        pivot = None
        for row in range(rank, rows):
            if matrix[row, col]:
                pivot = row
                break
        if pivot is None:
            col += 1
            continue
        if pivot != rank:
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(rows):
            if row != rank and matrix[row, col]:
                matrix[row] ^= matrix[rank]
        rank += 1
        col += 1
    return rank
