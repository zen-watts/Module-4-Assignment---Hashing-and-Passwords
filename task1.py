#!/usr/bin/env python3
"""
Task 1: SHA-256 pseudo-randomness and collision resistance experiments.

This script includes:
  - A basic SHA-256 hashing tool (full 256-bit hex digest).
  - An avalanche effect demo that flips exactly one bit.
  - A truncated-digest collision search (birthday attack).
  - A sweep that collects timing + trial counts for bits 8..50.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from typing import Dict, List, Tuple

from common import (
    describe_bytes,
    flip_one_bit,
    format_seconds,
    hamming_distance_bits,
    random_bytes,
    sha256_hex,
    truncated_sha256_int,
)


def parse_input_bytes(args: argparse.Namespace) -> bytes:
    """Parse input bytes for the 'hash' subcommand."""
    if args.text is not None:
        return args.text.encode("utf-8")
    if args.hex is not None:
        return bytes.fromhex(args.hex)
    if args.file is not None:
        with open(args.file, "rb") as handle:
            return handle.read()
    raise ValueError("No input specified")


def run_hash(args: argparse.Namespace) -> None:
    """Compute and print the SHA-256 hex digest for arbitrary input."""
    data = parse_input_bytes(args)
    digest_hex = sha256_hex(data)
    print(f"Input ({len(data)} bytes): {describe_bytes(data)}")
    print(f"SHA-256 digest: {digest_hex}")


def run_avalanche(args: argparse.Namespace) -> None:
    """
    Generate pairs of messages that differ by exactly one bit, then hash them.
    """
    for i in range(args.count):
        original = random_bytes(args.length)
        mutated, bit_index = flip_one_bit(original)

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


def _message_from_counter(counter: int, salt: bytes) -> bytes:
    """
    Deterministically construct a message from a counter and salt.

    This keeps memory usage lower (we store just counters) while still yielding
    distinct-looking inputs.
    """
    return salt + counter.to_bytes(8, "big")


def find_truncated_collision(bits: int, max_trials: int = 0) -> Tuple[int, bytes, bytes, int, float]:
    """
    Find a collision for a truncated SHA-256 digest (birthday attack).

    Returns (truncated_value, message1, message2, trials, elapsed_seconds).
    """
    if bits < 1 or bits > 50:
        raise ValueError("bits must be between 1 and 50")

    # Store the first counter seen for each truncated digest value.
    seen: Dict[int, int] = {}
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

        if max_trials and trials >= max_trials:
            raise RuntimeError("max_trials reached without finding a collision")

    raise RuntimeError("unreachable")


def run_collision(args: argparse.Namespace) -> None:
    """Run a single collision search for the requested bit size."""
    truncated, message1, message2, trials, elapsed = find_truncated_collision(
        args.bits, args.max_trials
    )

    print(f"Truncated bits: {args.bits}")
    print(f"Collision value (int): {truncated}")
    print(f"Message 1: {describe_bytes(message1)}")
    print(f"Message 2: {describe_bytes(message2)}")
    print(f"Trials: {trials}")
    print(f"Elapsed: {format_seconds(elapsed)}")


def run_sweep(args: argparse.Namespace) -> None:
    """
    Run the collision search for bits 8..50 (step size configurable).

    Prints CSV-formatted data so it can be used for graphing later.
    """
    results: List[Tuple[int, int, float]] = []

    for bits in range(args.min_bits, args.max_bits + 1, args.step):
        print(f"=== Collision search for {bits} bits ===")
        _, _, _, trials, elapsed = find_truncated_collision(bits, args.max_trials)
        results.append((bits, trials, elapsed))
        print(f"Result: trials={trials}, elapsed={format_seconds(elapsed)}")

    print("\nCSV output (bits,trials,seconds):")
    print("bits,trials,seconds")
    for bits, trials, elapsed in results:
        print(f"{bits},{trials},{elapsed:.6f}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as handle:
            handle.write("bits,trials,seconds\n")
            for bits, trials, elapsed in results:
                handle.write(f"{bits},{trials},{elapsed:.6f}\n")
        print(f"Wrote CSV data to {args.csv}")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments and subcommands."""
    parser = argparse.ArgumentParser(
        description="SHA-256 experiments: hashing, avalanche, and collisions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1A: Basic SHA-256 hashing tool.
    hash_parser = subparsers.add_parser("hash", help="Hash an input with SHA-256")
    group = hash_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", help="Input text (UTF-8)")
    group.add_argument("--hex", help="Input bytes as hex")
    group.add_argument("-f", "--file", help="Read input bytes from a file")
    hash_parser.set_defaults(func=run_hash)

    # 1B: Avalanche effect demo.
    avalanche_parser = subparsers.add_parser(
        "avalanche", help="Flip one bit and show SHA-256 avalanche effect"
    )
    avalanche_parser.add_argument(
        "-n", "--count", type=int, default=3, help="Number of trials to run"
    )
    avalanche_parser.add_argument(
        "-l", "--length", type=int, default=16, help="Random input length in bytes"
    )
    avalanche_parser.set_defaults(func=run_avalanche)

    # 1C: Single collision search.
    collision_parser = subparsers.add_parser(
        "collision", help="Find a collision for a truncated digest size"
    )
    collision_parser.add_argument(
        "--bits", type=int, required=True, help="Truncated digest size (8-50 bits)"
    )
    collision_parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional cap on trials (0 means no cap)",
    )
    collision_parser.set_defaults(func=run_collision)

    # 1C: Sweep across many bit sizes.
    sweep_parser = subparsers.add_parser(
        "sweep", help="Run collision searches for a range of bit sizes"
    )
    sweep_parser.add_argument("--min-bits", type=int, default=8)
    sweep_parser.add_argument("--max-bits", type=int, default=50)
    sweep_parser.add_argument("--step", type=int, default=2)
    sweep_parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional cap on trials per bit size (0 means no cap)",
    )
    sweep_parser.add_argument(
        "--csv",
        default="",
        help="Optional path to write CSV output for plotting",
    )
    sweep_parser.set_defaults(func=run_sweep)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
