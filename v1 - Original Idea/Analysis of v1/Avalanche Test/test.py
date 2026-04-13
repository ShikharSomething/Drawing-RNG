from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image


class DrawChaoticRNG:
    _MASK64 = (1 << 64) - 1
    _INIT_STATE = (
        0x243F6A8885A308D3,
        0x13198A2E03707344,
        0xA4093822299F31D0,
        0x082EFA98EC4E6C89,
    )
    _MIX_CONSTS = (
        0x9E3779B97F4A7C15,
        0xD1B54A32D192ED03,
        0x94D049BB133111EB,
        0xBF58476D1CE4E5B9,
        0xDB4F0B9175AE2165,
        0xA24BAED4963EE407,
    )
    _WEYL = 0x61C8864680B583EB

    def __init__(self, image_path: str | Path, warmup_rounds: int = 256) -> None:
        self.image_path = Path(image_path)
        if not self.image_path.is_file():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        if warmup_rounds < 0:
            raise ValueError("warmup_rounds must be non-negative")

        img = Image.open(self.image_path).convert("L")
        self.img = np.array(img, dtype=np.uint8)
        self.entropy = self._compute_entropy(self.img)
        self.feature_names, self.feature_streams = self._extract_feature_streams(self.img)
        self.feature_word_count = sum(len(stream) for stream in self.feature_streams)
        self.state = self._initialize_state(self.feature_streams, self.entropy)
        self.weyl = self._mix64(self.feature_word_count + int(round(self.entropy * 1_000_000)))
        self.counter = 0

        for _ in range(warmup_rounds):
            self._step()

    @staticmethod
    def _compute_entropy(img: np.ndarray) -> float:
        hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    @classmethod
    def _rotl64(cls, value: int, shift: int) -> int:
        shift &= 63
        value &= cls._MASK64
        return ((value << shift) | (value >> (64 - shift))) & cls._MASK64

    @classmethod
    def _mix64(cls, value: int) -> int:
        value &= cls._MASK64
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & cls._MASK64
        value ^= value >> 27
        value = (value * 0x94D049BB133111EB) & cls._MASK64
        value ^= value >> 31
        return value & cls._MASK64

    @classmethod
    def _bytes_to_words(cls, payload: bytes, domain: int) -> tuple[int, ...]:
        framed = bytearray()
        framed.extend((domain & cls._MASK64).to_bytes(8, "big"))
        framed.extend(len(payload).to_bytes(8, "big"))
        framed.extend(payload)

        pad = cls._mix64(domain ^ len(payload) ^ cls._WEYL)
        while len(framed) % 8 != 0:
            pad = cls._mix64(pad + cls._MIX_CONSTS[len(framed) % len(cls._MIX_CONSTS)])
            framed.append(pad & 0xFF)

        return tuple(
            int.from_bytes(framed[index : index + 8], "big") & cls._MASK64
            for index in range(0, len(framed), 8)
        )

    @classmethod
    def _extract_feature_streams(
        cls, img: np.ndarray
    ) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
        flat = img.reshape(-1).astype(np.uint8)
        if flat.size == 0:
            flat = np.array([0], dtype=np.uint8)

        flat16 = flat.astype(np.uint16)
        flat32 = flat.astype(np.int32)

        next_flat = np.roll(flat32, -1)
        next_flat[-1] = flat32[0]
        flat_diff = ((next_flat - flat32) & 0xFF).astype(np.uint8)

        next2_flat = np.roll(flat32, -2)
        next2_flat[-2:] = flat32[:2]
        second_diff = ((next2_flat - flat32) & 0xFF).astype(np.uint8)

        paired = flat
        if paired.size % 2 == 1:
            paired = np.concatenate((paired, paired[:1]))
        lhs = paired[0::2].astype(np.uint16)
        rhs = paired[1::2].astype(np.uint16)
        pair_sum = ((lhs + rhs) & 0xFF).astype(np.uint8)
        pair_xor = (lhs ^ rhs).astype(np.uint8)
        pair_mix = ((3 * lhs + 5 * rhs + (lhs ^ rhs)) & 0xFF).astype(np.uint8)

        hdiff = ((np.diff(img.astype(np.int16), axis=1) + 256) & 0xFF).astype(np.uint8)
        vdiff = ((np.diff(img.astype(np.int16), axis=0) + 256) & 0xFF).astype(np.uint8)
        spatial = np.concatenate((hdiff.ravel(), vdiff.ravel()))
        if spatial.size == 0:
            spatial = flat.copy()

        histogram = np.bincount(flat, minlength=256).astype(np.uint32)
        weighted = ((flat16 * np.arange(1, flat.size + 1, dtype=np.uint16)) & 0xFF).astype(np.uint8)

        named_payloads = (
            ("flat_pixels", flat.tobytes()),
            ("adjacent_diff", flat_diff.tobytes()),
            ("second_diff", second_diff.tobytes()),
            ("pair_sum", pair_sum.tobytes()),
            ("pair_xor", pair_xor.tobytes()),
            ("pair_mix", pair_mix.tobytes()),
            ("spatial_diff", spatial.tobytes()),
            ("histogram", histogram.tobytes()),
            ("position_weighted", weighted.tobytes()),
        )

        feature_names = tuple(name for name, _ in named_payloads)
        feature_streams = tuple(
            cls._bytes_to_words(payload, domain=0xA5A5A5A500000000 | index)
            for index, (_, payload) in enumerate(named_payloads, start=1)
        )
        return feature_names, feature_streams

    @classmethod
    def _scramble_state(cls, state: list[int], injection: int, rounds: int = 2) -> None:
        for round_index in range(rounds):
            a, b, c, d = state
            round_key = cls._mix64(
                injection + cls._MIX_CONSTS[round_index % len(cls._MIX_CONSTS)] + round_index
            )
            a = cls._mix64((a + cls._rotl64(b, 7) + round_key) & cls._MASK64)
            b = cls._mix64((b ^ cls._rotl64(c + round_key, 19) ^ a) & cls._MASK64)
            c = cls._mix64((c + cls._rotl64(d, 31) + (a ^ round_key)) & cls._MASK64)
            d = cls._mix64((d ^ cls._rotl64(a + b + round_key, 43) ^ c) & cls._MASK64)
            state[:] = [a, b, c, d]
            injection = (round_key + cls._rotl64(d, 11) + round_index) & cls._MASK64

    @classmethod
    def _initialize_state(
        cls, feature_streams: tuple[tuple[int, ...], ...], entropy: float
    ) -> list[int]:
        state = list(cls._INIT_STATE)
        entropy_word = cls._mix64(int(round(entropy * 1_000_000)))

        for stream_index, stream in enumerate(feature_streams):
            stream_acc = cls._mix64(
                entropy_word ^ cls._MIX_CONSTS[stream_index % len(cls._MIX_CONSTS)] ^ len(stream)
            )
            for word_index, word in enumerate(stream):
                lane = (stream_index + word_index) & 3
                injection = cls._mix64(
                    (
                        word
                        + cls._rotl64(stream_acc, (word_index * 7 + stream_index * 13) & 63)
                        + ((word_index + 1) * cls._MIX_CONSTS[(stream_index + 1) % len(cls._MIX_CONSTS)])
                    )
                    & cls._MASK64
                )
                state[lane] = (
                    state[lane]
                    + injection
                    + cls._rotl64(state[(lane - 1) & 3], 21)
                    + cls._MIX_CONSTS[lane]
                ) & cls._MASK64
                state[(lane + 1) & 3] ^= cls._rotl64(injection + state[lane], 17)
                state[(lane + 2) & 3] = (
                    state[(lane + 2) & 3]
                    + (injection ^ cls._rotl64(state[(lane + 1) & 3], 9))
                ) & cls._MASK64
                state[(lane + 3) & 3] = cls._mix64(
                    state[(lane + 3) & 3] ^ injection ^ word_index ^ (stream_index << 32)
                )
                stream_acc = cls._mix64(stream_acc ^ injection ^ word)

            cls._scramble_state(state, stream_acc ^ stream_index, rounds=3)

        return [
            lane if lane != 0 else cls._MIX_CONSTS[index]
            for index, lane in enumerate(state)
        ]

    def _step(self) -> None:
        inject = (self.weyl + self.counter * self._MIX_CONSTS[0]) & self._MASK64

        for stream_index, stream in enumerate(self.feature_streams):
            lane = stream_index & 3
            shift = (stream_index * 11 + 5) & 63
            idx = (
                self.counter
                + self._rotl64(self.state[lane], shift)
                + self._rotl64(self.state[(lane + 1) & 3], shift + 9)
                + self._MIX_CONSTS[stream_index % len(self._MIX_CONSTS)]
            ) % len(stream)
            word = stream[idx]
            inject = self._mix64(
                (
                    inject
                    + (word * self._MIX_CONSTS[(stream_index + 2) % len(self._MIX_CONSTS)])
                    + self._rotl64(self.state[(lane + 2) & 3] ^ word, shift + 17)
                )
                & self._MASK64
            )

        a, b, c, d = self.state
        t0 = (a + self._rotl64(b, 7) + inject + self._MIX_CONSTS[1]) & self._MASK64
        t1 = (b ^ self._rotl64(c + inject, 17) ^ self._MIX_CONSTS[2]) & self._MASK64
        t2 = (
            c
            + self._rotl64(d, 29)
            + ((inject * self._MIX_CONSTS[3]) & self._MASK64)
            + self._MIX_CONSTS[4]
        ) & self._MASK64
        t3 = (
            d
            ^ self._rotl64(a + inject, 41)
            ^ self._MIX_CONSTS[5]
            ^ self.counter
        ) & self._MASK64

        self.state[0] = self._mix64(t0 ^ ((t2 * self._MIX_CONSTS[0]) & self._MASK64))
        self.state[1] = self._mix64((t1 + self._rotl64(t3, 11) + inject) & self._MASK64)
        self.state[2] = self._mix64(
            (t2 ^ self._rotl64(self.state[0], 23) ^ ((t0 * self._MIX_CONSTS[2]) & self._MASK64))
            & self._MASK64
        )
        self.state[3] = self._mix64(
            (t3 + self._rotl64(self.state[1], 37) + (self.state[0] ^ self.state[2]) + self.weyl)
            & self._MASK64
        )

        self.weyl = (self.weyl + self._WEYL) & self._MASK64
        self.counter += 1

    def _next_word(self) -> int:
        self._step()
        a, b, c, d = self.state
        folded = (
            a
            + self._rotl64(b, 9)
            + self._rotl64(c, 21)
            + self._rotl64(d, 33)
            + self.weyl
            + self.counter * self._MIX_CONSTS[1]
        ) & self._MASK64
        return self._mix64(
            folded ^ ((b * self._MIX_CONSTS[3]) & self._MASK64) ^ self._rotl64(c + d, 27)
        )

    def random_bytes(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("n must be non-negative")
        output = bytearray()
        while len(output) < n:
            output.extend(self._next_word().to_bytes(8, "big"))
        return bytes(output[:n])


def iter_images(folder: Path, recursive: bool = False):
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    paths = set()
    if recursive:
        for pattern in patterns:
            paths.update(p for p in folder.rglob(pattern) if p.is_file())
    else:
        for pattern in patterns:
            paths.update(p for p in folder.glob(pattern) if p.is_file())
    return sorted(paths)


def run_single_image(image_path: Path, num_bytes: int, warmup: int) -> dict[str, str | int]:
    rng = DrawChaoticRNG(image_path, warmup_rounds=warmup)
    output_bytes = rng.random_bytes(num_bytes)
    return {
        "name": image_path.stem,
        "filename": image_path.name,
        "path": str(image_path.resolve()),
        "entropy": f"{rng.entropy:.12f}",
        "feature_streams": len(rng.feature_streams),
        "feature_words": rng.feature_word_count,
        "num_bytes": num_bytes,
        "hex": output_bytes.hex(),
    }


def write_outputs_csv(rows: list[dict[str, str | int]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "filename",
        "path",
        "entropy",
        "feature_streams",
        "feature_words",
        "num_bytes",
        "hex",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hex_to_bytes(s: str) -> bytes:
    s = s.strip().lower().replace("0x", "").replace(" ", "")
    if len(s) % 2 != 0:
        raise ValueError(f"Hex string has odd length: {len(s)}")
    return bytes.fromhex(s)


def bit_hamming_distance(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def byte_hamming_distance(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)


def compute_hamming_rows(output_rows: list[dict[str, str | int]]) -> list[dict[str, str | int | float]]:
    pairs = []
    data = [(str(r["name"]), hex_to_bytes(str(r["hex"]))) for r in output_rows]
    if len(data) < 2:
        return pairs

    byte_len = len(data[0][1])
    if any(len(b) != byte_len for _, b in data):
        raise ValueError("All outputs must have the same byte length")

    bit_total = byte_len * 8
    for (name1, b1), (name2, b2) in combinations(data, 2):
        bit_hd = bit_hamming_distance(b1, b2)
        byte_hd = byte_hamming_distance(b1, b2)
        pairs.append({
            "image_a": name1,
            "image_b": name2,
            "bytes_compared": byte_len,
            "bits_compared": bit_total,
            "bit_hamming_distance": bit_hd,
            "bit_hamming_ratio": round(bit_hd / bit_total, 6),
            "byte_hamming_distance": byte_hd,
            "byte_hamming_ratio": round(byte_hd / byte_len, 6),
        })
    return pairs


def write_hamming_csv(rows: list[dict[str, str | int | float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_a",
                "image_b",
                "bytes_compared",
                "bits_compared",
                "bit_hamming_distance",
                "bit_hamming_ratio",
                "byte_hamming_distance",
                "byte_hamming_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_matrix(hamming_rows: list[dict[str, str | int | float]], value_column: str):
    names = sorted({str(row["image_a"]) for row in hamming_rows} | {str(row["image_b"]) for row in hamming_rows})
    matrix = {a: {b: "" for b in names} for a in names}
    for name in names:
        matrix[name][name] = "0"
    for row in hamming_rows:
        a = str(row["image_a"])
        b = str(row["image_b"])
        value = str(row[value_column])
        matrix[a][b] = value
        matrix[b][a] = value
    return names, matrix


def write_matrix_csv(names, matrix, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image"] + names)
        for a in names:
            writer.writerow([a] + [matrix[a][b] for b in names])


def write_matrix_html(names, matrix, out_path: Path, title: str) -> None:
    def bg_color(value: str) -> str:
        if value == "":
            return "#ffffff"
        try:
            x = float(value)
        except ValueError:
            return "#ffffff"
        x = max(0.0, min(1.0, x))
        r = 255
        g = int(255 * (1.0 - x))
        b = int(255 * (1.0 - x))
        return f"rgb({r},{g},{b})"

    html = []
    html.append("<!doctype html>")
    html.append("<html><head><meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append("""
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
table { border-collapse: collapse; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: center; min-width: 88px; }
th:first-child, td:first-child { text-align: left; position: sticky; left: 0; background: #f7f7f7; }
thead th { position: sticky; top: 0; background: #f0f0f0; }
.wrapper { overflow: auto; max-width: 100%; max-height: 85vh; border: 1px solid #ddd; }
.caption { margin-bottom: 12px; color: #333; }
.meta { margin: 8px 0 16px; color: #555; }
code { background: #f5f5f5; padding: 2px 4px; }
</style>
</head><body>
""")
    html.append(f"<h2>{title}</h2>")
    html.append("<div class='caption'>Darker red means larger value.</div>")
    html.append(f"<div class='meta'>Metric shown in cells: <code>{title}</code></div>")
    html.append("<div class='wrapper'><table>")
    html.append("<thead><tr><th>image</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr></thead>")
    html.append("<tbody>")
    for a in names:
        row_html = [f"<tr><td>{a}</td>"]
        for b in names:
            v = matrix[a][b]
            color = bg_color(v)
            row_html.append(f"<td style='background:{color}'>{v}</td>")
        row_html.append("</tr>")
        html.append("".join(row_html))
    html.append("</tbody></table></div></body></html>")
    out_path.write_text("".join(html), encoding="utf-8")


def write_summary_json(
    output_rows: list[dict[str, str | int]],
    hamming_rows: list[dict[str, str | int | float]],
    out_path: Path,
    value_column: str,
) -> None:
    summary = {
        "image_count": len(output_rows),
        "pairwise_comparisons": len(hamming_rows),
        "metric_for_matrix": value_column,
        "images": [str(r["name"]) for r in output_rows],
    }
    if hamming_rows:
        values = [float(r[value_column]) for r in hamming_rows]
        summary["min_metric"] = min(values)
        summary["max_metric"] = max(values)
        summary["mean_metric"] = round(sum(values) / len(values), 6)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-in-one avalanche pipeline: images folder -> output.csv -> hamming_results.csv -> matrix CSV/HTML."
    )
    parser.add_argument("--image-dir", type=Path, required=True, help="Folder containing test images")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan image-dir")
    parser.add_argument("-n", "--num-bytes", type=int, default=64, help="Number of output bytes per image")
    parser.add_argument("--warmup", type=int, default=256, help="Warmup rounds")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("avalanche_outputs"),
        help="Folder where all result files will be written",
    )
    parser.add_argument(
        "--matrix-metric",
        default="bit_hamming_ratio",
        choices=[
            "bit_hamming_distance",
            "bit_hamming_ratio",
            "byte_hamming_distance",
            "byte_hamming_ratio",
        ],
        help="Metric to use in the square matrix CSV/HTML",
    )
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {args.image_dir}")

    image_paths = iter_images(args.image_dir, recursive=args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.image_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_rows = [run_single_image(path, args.num_bytes, args.warmup) for path in image_paths]
    output_csv = args.out_dir / "output.csv"
    write_outputs_csv(output_rows, output_csv)

    hamming_rows = compute_hamming_rows(output_rows)
    hamming_csv = args.out_dir / "hamming_results.csv"
    write_hamming_csv(hamming_rows, hamming_csv)

    names, matrix = build_matrix(hamming_rows, args.matrix_metric)
    matrix_csv = args.out_dir / "hamming_matrix.csv"
    matrix_html = args.out_dir / "hamming_matrix.html"
    write_matrix_csv(names, matrix, matrix_csv)
    write_matrix_html(names, matrix, matrix_html, f"Hamming Matrix: {args.matrix_metric}")

    summary_json = args.out_dir / "summary.json"
    write_summary_json(output_rows, hamming_rows, summary_json, args.matrix_metric)

    print(f"Processed {len(output_rows)} images")
    print(f"Wrote: {output_csv.resolve()}")
    print(f"Wrote: {hamming_csv.resolve()}")
    print(f"Wrote: {matrix_csv.resolve()}")
    print(f"Wrote: {matrix_html.resolve()}")
    print(f"Wrote: {summary_json.resolve()}")


if __name__ == "__main__":
    main()