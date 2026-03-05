import hashlib
import numpy as np
from PIL import Image


class DrawChaoticRNG:
    def __init__(self, image_path):
        # Load grayscale image
        img = Image.open(image_path).convert("L")
        self.img = np.array(img, dtype=np.uint8)

        # Compute histogram entropy -> controls r
        hist, _ = np.histogram(self.img, bins=256, range=(0, 256), density=True)
        H = -np.sum(hist * np.log2(hist + 1e-12))
        # Map [0,8] approx entropy to [3.57,4]
        self.r = 3.57 + (H % 0.43)

        # Extract pixel differences modulate chaotic map
        diffs = []
        diffs.append(np.diff(self.img, axis=1).flatten())
        diffs.append(np.diff(self.img, axis=0).flatten())
        diffs = np.concatenate(diffs)
        self.eps = ((diffs.astype(np.int16) + 256) % 256) / 256.0
        self.e_len = len(self.eps)

        # SHA-based initial state
        m = hashlib.sha3_256(self.img.tobytes()).digest()
        self.x = int.from_bytes(m, "big") / (2**256)

        self.counter = 0

    def _step(self):
        """One iteration of chaotic map"""
        ep = self.eps[self.counter % self.e_len]
        self.x = (self.r * self.x * (1 - self.x) + ep) % 1.0
        self.counter += 1

    def random_bytes(self, n):
        out = b""
        while len(out) < n:
            # Generate 64 bytes from chaos, then hash them
            block = b""
            for _ in range(16):  # 16 * 4 bytes = 64
                self._step()
                v = int(self.x * (2**32)) & 0xFFFFFFFF
                block += v.to_bytes(4, "big")
            out += hashlib.sha3_256(block).digest()
        return out[:n]


# Example usage
if __name__ == "__main__":
    rng = DrawChaoticRNG("img.png")

    print("r =", rng.r)
    print("64 random bytes:", rng.random_bytes(64).hex())
