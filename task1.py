"""
Task 1: SHA-256 pseudo-randomness and collision resistance experiments.

Default behavior:
  - Run 3 avalanche trials (16-byte inputs).
  - Run a truncated-digest collision sweep for bits 8..50 (step 2).
  - Print CSV-formatted output to stdout for plotting.
"""

import itertools
import sys
import time

from common import (
    describe_bytes,
    flip_one_bit,
    format_seconds,
    hamming_distance_bits,
    random_bytes,
    sha256_hex,
    truncated_sha256_int,
)
# ---------------------------------------------------------------------------
# Avalanche experiment: flip 1 input bit and observe output diffusion
# ---------------------------------------------------------------------------

AVALANCHE_COUNT = 3
AVALANCHE_LENGTH = 16


def run_avalanche() -> None:
    """Generate message pairs that differ by one bit, then hash them."""
    for i in range(AVALANCHE_COUNT):
        original = random_bytes(AVALANCHE_LENGTH) # function found in common.py
        mutated, bit_index = flip_one_bit(original) # function found in common.py

        # Verify the Hamming distance is exactly 1 bit.
        distance = hamming_distance_bits(original, mutated)

        digest_a = sha256_hex(original)
        digest_b = sha256_hex(mutated)

        print(f"--- Avalanche trial {i + 1} ---")
        print(f"Flipped bit index: {bit_index} (Hamming distance: {distance})")
        print(f"Input A: {describe_bytes(original)}")
        print(f"Input B: {describe_bytes(mutated)}")
        print(f"Digest A: {digest_a}")
        print(f"Digest B: {digest_b}")

# ---------------------------------------------------------------------------
# Collision experiment: truncated SHA-256 birthday collisions + sweep 8..50
# ---------------------------------------------------------------------------

SWEEP_MIN_BITS = 8
SWEEP_MAX_BITS = 50
SWEEP_STEP = 2


def _message_from_counter(counter: int, salt: bytes) -> bytes:
    """
    Deterministically construct a message from a counter and salt.

    This keeps memory usage lower (we store just counters) while still yielding
    distinct-looking inputs.
    """
    return salt + counter.to_bytes(8, "big")


def find_truncated_collision(bits: int) -> tuple[int, bytes, bytes, int, float]:
    """
    Find a collision for a truncated SHA-256 digest (birthday attack).

    Returns (truncated_value, message1, message2, trials, elapsed_seconds).
    """
    if bits < 1 or bits > 50:
        raise ValueError("bits must be between 1 and 50")

    # Store the first counter seen for each truncated digest value.
    seen: dict[int, int] = {}
    salt = random_bytes(8)

    trials = 0
    start = time.perf_counter()

    for counter in itertools.count():
        message = _message_from_counter(counter, salt)
        truncated = truncated_sha256_int(message, bits)
        trials += 1

        if truncated in seen:
            other_counter = seen[truncated]
            if other_counter != counter:
                elapsed = time.perf_counter() - start
                first_message = _message_from_counter(other_counter, salt)
                return truncated, first_message, message, trials, elapsed
        else:
            seen[truncated] = counter

    raise RuntimeError("unreachable")


def run_sweep() -> None:
    """Run the collision search for bits 8..50 and print CSV output."""
    results: list[tuple[int, int, float]] = []

    for bits in range(SWEEP_MIN_BITS, SWEEP_MAX_BITS + 1, SWEEP_STEP):
        print(f"=== Collision search for {bits} bits ===")
        _, _, _, trials, elapsed = find_truncated_collision(bits)
        results.append((bits, trials, elapsed))
        print(f"Result: trials={trials}, elapsed={format_seconds(elapsed)}")

    print("\nCSV output (bits,trials,seconds):")
    print("bits,trials,seconds")
    for bits, trials, elapsed in results:
        print(f"{bits},{trials},{elapsed:.6f}")


def main() -> int:
    run_avalanche()
    print() # blank line
    run_sweep()
    return 0


if __name__ == "__main__":
    sys.exit(main())
