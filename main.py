from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


class DrawChaoticRNG:
    """
    Image-seeded deterministic RNG using multi-state integer bit-mixing.

    The design aims for stronger diffusion than the earlier float-chaos version,
    but it is still a custom generator and should not be treated as a proven
    cryptographically secure RNG for real keys or production secrets.
    """

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

    def random_uint32(self) -> int:
        return int.from_bytes(self.random_bytes(4), "big")

    def random(self) -> float:
        return self.random_uint32() / (1 << 32)

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")

        limit = (1 << 32) - ((1 << 32) % upper)
        while True:
            value = self.random_uint32()
            if value < limit:
                return value % upper

    def randint(self, a: int, b: int) -> int:
        if a > b:
            raise ValueError("a must be <= b")
        return a + self.randbelow(b - a + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate bytes from an image-seeded RNG.")
    parser.add_argument("image", nargs="?", default="img.png", help="Path to the source image")
    parser.add_argument(
        "-n",
        "--num-bytes",
        type=int,
        default=64,
        help="Number of random bytes to generate",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=256,
        help="Number of warmup iterations before output generation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = DrawChaoticRNG(args.image, warmup_rounds=args.warmup)

    print(f"image = {Path(args.image).resolve()}")
    print(f"entropy = {rng.entropy:.6f}")
    print(f"feature_streams = {len(rng.feature_streams)}")
    print(f"feature_words = {rng.feature_word_count}")
    print(f"{args.num_bytes} random bytes = {rng.random_bytes(args.num_bytes).hex()}")
    print(f"sample randint(1, 10) = {rng.randint(1, 10)}")
    print(f"sample random() = {rng.random():.12f}")


if __name__ == "__main__":
    main()
