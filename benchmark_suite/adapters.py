from __future__ import annotations

import copy
import importlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


def resolve_factory(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError("Factory path must look like module_name:callable_name")
    module_name, callable_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, callable_name, None)
    if factory is None:
        raise AttributeError(f"Could not find {callable_name!r} in module {module_name!r}")
    return factory


class AlgorithmAdapter:
    def __init__(
        self,
        factory_path: str,
        seed: str,
        seed_type: str = "image",
        factory_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.factory_path = factory_path
        self.factory = resolve_factory(factory_path)
        self.seed_type = seed_type
        self.seed_arg = seed
        self.factory_kwargs = dict(factory_kwargs or {})
        self._tempdir = tempfile.TemporaryDirectory(prefix="rng-bench-")
        self._image_counter = 0
        self.base_seed = self._load_seed(seed, seed_type)

    def close(self) -> None:
        self._tempdir.cleanup()

    def _load_seed(self, seed: str, seed_type: str) -> Any:
        if seed_type == "image":
            return np.array(Image.open(seed).convert("L"), dtype=np.uint8)
        if seed_type == "bytes":
            path = Path(seed)
            if path.exists():
                return path.read_bytes()
            cleaned = seed.removeprefix("0x")
            return bytes.fromhex(cleaned)
        if seed_type == "text":
            return seed
        raise ValueError(f"Unsupported seed type: {seed_type}")

    def _materialize_seed(self, seed_value: Any) -> Any:
        if self.seed_type == "image":
            path = Path(self._tempdir.name) / f"seed_{self._image_counter:05d}.png"
            self._image_counter += 1
            Image.fromarray(np.array(seed_value, dtype=np.uint8), mode="L").save(path)
            return str(path)
        return seed_value

    def _construct(self, seed_value: Any, kwargs_override: dict[str, Any] | None = None) -> Any:
        kwargs = dict(self.factory_kwargs)
        if kwargs_override:
            kwargs.update(kwargs_override)
        materialized_seed = self._materialize_seed(seed_value)
        return self.factory(materialized_seed, **kwargs)

    def create_generator(self, seed_override: Any | None = None, kwargs_override: dict[str, Any] | None = None) -> Any:
        return self._construct(self.base_seed if seed_override is None else seed_override, kwargs_override)

    def random_bytes(
        self,
        n: int,
        seed_override: Any | None = None,
        kwargs_override: dict[str, Any] | None = None,
    ) -> bytes:
        generator = self.create_generator(seed_override=seed_override, kwargs_override=kwargs_override)
        return generator.random_bytes(n)

    def clone_generator(self, generator: Any) -> Any:
        return copy.deepcopy(generator)

    def capture_state(self, generator: Any) -> dict[str, Any]:
        return copy.deepcopy(getattr(generator, "__dict__", {}))

    def restore_state(self, generator: Any, snapshot: dict[str, Any]) -> Any:
        generator.__dict__ = copy.deepcopy(snapshot)
        return generator

    def state_words(self, generator: Any) -> list[int]:
        words: list[int] = []

        def add_value(value: Any) -> None:
            if isinstance(value, (int, np.integer)):
                words.append(int(value) & ((1 << 64) - 1))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    add_value(item)
            elif isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.integer):
                    words.extend(int(item) & ((1 << 64) - 1) for item in value.ravel()[:64])

        for key in ("state", "weyl", "counter", "x", "y", "z", "w"):
            if hasattr(generator, key):
                add_value(getattr(generator, key))
        return words

    def state_bit_size(self, generator: Any) -> int:
        return 64 * len(self.state_words(generator))

    def mutate_live_state(self, generator: Any) -> Any | None:
        mutant = self.clone_generator(generator)
        if hasattr(mutant, "state") and isinstance(mutant.state, list) and mutant.state:
            mutant.state[0] ^= 1
            return mutant
        for key in ("x", "y", "z", "w", "weyl"):
            if hasattr(mutant, key) and isinstance(getattr(mutant, key), int):
                setattr(mutant, key, getattr(mutant, key) ^ 1)
                return mutant
        return None

    def state_fingerprint(self, generator: Any) -> int:
        words = self.state_words(generator)
        if not words:
            return 0
        acc = 0x9E3779B97F4A7C15
        for index, word in enumerate(words[:16]):
            acc ^= ((word + index * 0xD1B54A32D192ED03) & ((1 << 64) - 1))
            acc = ((acc << 13) | (acc >> 51)) & ((1 << 64) - 1)
            acc = (acc * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return acc

    def available_seed_mutations(self, limit: int = 8) -> dict[str, Any]:
        if self.seed_type == "image":
            return self._image_mutations(limit=limit)
        if self.seed_type in {"bytes", "text"}:
            return self._byte_mutations(limit=limit)
        return {}

    def seed_distance_bits(self, left: Any, right: Any) -> int:
        if self.seed_type == "image":
            left_bits = np.unpackbits(np.asarray(left, dtype=np.uint8).ravel())
            right_bits = np.unpackbits(np.asarray(right, dtype=np.uint8).ravel())
            width = min(left_bits.size, right_bits.size)
            return int(np.count_nonzero(left_bits[:width] != right_bits[:width]))
        left_bytes = self._seed_to_bytes(left)
        right_bytes = self._seed_to_bytes(right)
        width = min(len(left_bytes), len(right_bytes))
        return int(
            np.count_nonzero(
                np.unpackbits(np.frombuffer(left_bytes[:width], dtype=np.uint8))
                != np.unpackbits(np.frombuffer(right_bytes[:width], dtype=np.uint8))
            )
        )

    def _seed_to_bytes(self, seed_value: Any) -> bytes:
        if self.seed_type == "image":
            return np.asarray(seed_value, dtype=np.uint8).tobytes()
        if self.seed_type == "text":
            return str(seed_value).encode("utf-8")
        return bytes(seed_value)

    def _image_mutations(self, limit: int) -> dict[str, Any]:
        image = np.array(self.base_seed, copy=True)
        flat = image.reshape(-1)
        if flat.size == 0:
            return {}

        center = flat.size // 2
        mutations: dict[str, Any] = {}

        lsb_flip = flat.copy()
        lsb_flip[center] ^= 0x01
        mutations["pixel_lsb_flip"] = lsb_flip.reshape(image.shape)

        msb_flip = flat.copy()
        msb_flip[center] ^= 0x80
        mutations["pixel_msb_flip"] = msb_flip.reshape(image.shape)

        pair_flip = flat.copy()
        pair_flip[center] ^= 0x01
        pair_flip[(center + 17) % flat.size] ^= 0x20
        mutations["two_pixel_bit_flip"] = pair_flip.reshape(image.shape)

        mutations["row_shift"] = np.roll(image, 1, axis=1)
        mutations["flat_shift"] = np.roll(flat, 1).reshape(image.shape)
        mutations["rotate_90"] = np.rot90(image).copy()
        mutations["mirror"] = np.fliplr(image).copy()
        mutations["vertical_flip"] = np.flipud(image).copy()

        return dict(list(mutations.items())[:limit])

    def _byte_mutations(self, limit: int) -> dict[str, Any]:
        if self.seed_type == "text":
            seed_bytes = bytearray(str(self.base_seed).encode("utf-8"))
        else:
            seed_bytes = bytearray(self.base_seed)
        if not seed_bytes:
            seed_bytes = bytearray(b"\x00")

        center = len(seed_bytes) // 2
        mutations: dict[str, Any] = {}

        flip1 = bytearray(seed_bytes)
        flip1[center] ^= 0x01
        mutations["byte_lsb_flip"] = self._coerce_seed_back(bytes(flip1))

        flip2 = bytearray(seed_bytes)
        flip2[center] ^= 0x80
        mutations["byte_msb_flip"] = self._coerce_seed_back(bytes(flip2))

        rolled = seed_bytes[1:] + seed_bytes[:1]
        mutations["byte_rotate"] = self._coerce_seed_back(bytes(rolled))

        reversed_seed = bytearray(reversed(seed_bytes))
        mutations["byte_reverse"] = self._coerce_seed_back(bytes(reversed_seed))

        pair = bytearray(seed_bytes)
        pair[center] ^= 0x01
        pair[(center + 3) % len(pair)] ^= 0x20
        mutations["two_byte_flip"] = self._coerce_seed_back(bytes(pair))

        return dict(list(mutations.items())[:limit])

    def _coerce_seed_back(self, value: bytes) -> Any:
        if self.seed_type == "text":
            return value.decode("utf-8", errors="replace")
        return value

    @staticmethod
    def parse_factory_kwargs(raw_value: str | None) -> dict[str, Any]:
        if not raw_value:
            return {}
        return json.loads(raw_value)
