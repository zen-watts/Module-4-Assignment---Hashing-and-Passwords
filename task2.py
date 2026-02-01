#!/usr/bin/env python3
"""
Task 2: bcrypt dictionary cracker for a shadow-style file.

Dependencies (install once):
  pip install bcrypt nltk
  python -m nltk.downloader words
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import sys
import time
from typing import Dict, List, Set, Tuple

try:
    import bcrypt
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("Missing dependency: bcrypt. Install with `pip install bcrypt`.") from exc

from common import format_seconds


ShadowEntry = Tuple[str, str]
HashEntry = Tuple[str, bytes]


def load_shadow_entries(path: str) -> List[ShadowEntry]:
    """Read shadow-style lines of the form Username:<bcrypt-hash>."""
    entries: List[ShadowEntry] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"Invalid shadow line (missing ':'): {line}")
            username, hash_str = line.split(":", 1)
            entries.append((username, hash_str))
    return entries


def bcrypt_cost(hash_str: str) -> int:
    """Extract the bcrypt cost parameter from a full hash string."""
    parts = hash_str.split("$")
    if len(parts) < 4:
        raise ValueError(f"Invalid bcrypt hash: {hash_str}")
    return int(parts[2])


def ensure_nltk_words() -> None:
    """
    Ensure the NLTK words corpus is available.

    If it is missing, try to download it; otherwise, print instructions.
    """
    try:
        import nltk
        nltk.data.find("corpora/words")
    except LookupError:
        print("NLTK words corpus not found. Attempting download...")
        try:
            import nltk
            nltk.download("words", quiet=True)
            nltk.data.find("corpora/words")
        except Exception:
            raise SystemExit(
                "NLTK words corpus is missing. Run: python -m nltk.downloader words"
            )
    except ImportError as exc:
        raise SystemExit("Missing dependency: nltk. Install with `pip install nltk`.") from exc


def build_candidate_list(min_len: int, max_len: int) -> List[bytes]:
    """
    Build the candidate list from the NLTK words corpus.

    Only single-word, alphabetic entries of length min_len..max_len are kept.
    """
    ensure_nltk_words()
    from nltk.corpus import words

    candidates: List[bytes] = []
    seen: Set[str] = set()

    for word in words.words():
        word = word.strip()
        if not word.isalpha():
            continue
        word = word.lower()
        if min_len <= len(word) <= max_len and word not in seen:
            seen.add(word)
            candidates.append(word.encode("utf-8"))

    return candidates


def group_by_cost(entries: List[ShadowEntry]) -> Dict[int, List[ShadowEntry]]:
    """Group shadow entries by bcrypt cost (work factor)."""
    groups: Dict[int, List[ShadowEntry]] = {}
    for username, hash_str in entries:
        cost = bcrypt_cost(hash_str)
        groups.setdefault(cost, []).append((username, hash_str))
    return groups


def worker_crack(
    hashes: List[HashEntry],
    candidates: List[bytes],
    start_idx: int,
    end_idx: int,
    solved: "mp.managers.DictProxy",
    result_queue: "mp.Queue",
    stop_event: "mp.Event",
    progress_counter: "mp.Value",
    overall_start: float,
) -> None:
    """
    Worker process: test a slice of the candidate list against all unsolved hashes.

    Each worker owns a contiguous slice [start_idx, end_idx).
    """
    local_checks = 0

    for idx in range(start_idx, end_idx):
        if stop_event.is_set():
            break

        candidate = candidates[idx]

        # Check this candidate against each still-unsolved hash in the group.
        for username, hash_bytes in hashes:
            if username in solved:
                continue
            local_checks += 1
            if bcrypt.checkpw(candidate, hash_bytes):
                # Record the discovery with the parent process.
                if username not in solved:
                    elapsed = time.perf_counter() - overall_start
                    solved[username] = (candidate.decode("utf-8"), elapsed)
                    result_queue.put((username, candidate.decode("utf-8"), elapsed))

        if local_checks >= 200:
            with progress_counter.get_lock():
                progress_counter.value += local_checks
            local_checks = 0

    if local_checks:
        with progress_counter.get_lock():
            progress_counter.value += local_checks


def crack_cost_group(
    cost: int,
    entries: List[ShadowEntry],
    candidates: List[bytes],
    workers: int,
    overall_start: float,
    progress_interval: float,
) -> Dict[str, Tuple[str, float]]:
    """
    Crack all hashes for a single bcrypt cost group using multiprocessing.
    """
    total_candidates = len(candidates)
    if total_candidates == 0:
        return {}
    total_checks = total_candidates * len(entries)

    manager = mp.Manager()
    solved = manager.dict()  # username -> (password, elapsed_seconds)
    progress_counter = mp.Value("i", 0)
    stop_event = mp.Event()
    result_queue: mp.Queue = mp.Queue()

    # Prepare hash bytes once to avoid re-encoding.
    hash_entries: List[HashEntry] = [
        (username, hash_str.encode("utf-8")) for username, hash_str in entries
    ]

    workers = max(1, min(workers, total_candidates))
    chunk_size = math.ceil(total_candidates / workers)

    processes: List[mp.Process] = []
    for worker_id in range(workers):
        start_idx = worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_candidates)
        if start_idx >= end_idx:
            continue

        process = mp.Process(
            target=worker_crack,
            args=(
                hash_entries,
                candidates,
                start_idx,
                end_idx,
                solved,
                result_queue,
                stop_event,
                progress_counter,
                overall_start,
            ),
        )
        processes.append(process)
        process.start()

    last_time = time.perf_counter()
    last_count = 0

    while True:
        # Drain results without blocking, so progress stays responsive.
        while True:
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break

        if len(solved) >= len(entries):
            stop_event.set()

        now = time.perf_counter()
        if now - last_time >= progress_interval:
            tested = progress_counter.value
            rate = (tested - last_count) / (now - last_time) if now > last_time else 0.0
            remaining = max(0, total_checks - tested)
            eta = remaining / rate if rate > 0 else float("inf")
            print(
                f"[cost {cost}] tested {tested}/{total_checks} checks, "
                f"{rate:.1f} checks/sec, ETA {eta:.1f}s"
            )
            last_time = now
            last_count = tested

        if stop_event.is_set() or progress_counter.value >= total_checks:
            break

        time.sleep(0.1)

    stop_event.set()

    for process in processes:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

    # Convert solved proxy dict to a plain dict for easier use by the caller.
    return dict(solved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bcrypt dictionary cracker.")
    parser.add_argument(
        "-s",
        "--shadow",
        default="shadow.txt",
        help="Path to the shadow-style file (default: shadow.txt)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=6,
        help="Minimum candidate length (default: 6)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=10,
        help="Maximum candidate length (default: 10)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="Seconds between progress updates (default: 1.0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    entries = load_shadow_entries(args.shadow)
    if not entries:
        print("No entries found in the shadow file.")
        return 1

    print(f"Loaded {len(entries)} shadow entries.")
    print("Building candidate list from NLTK words corpus...")
    candidates = build_candidate_list(args.min_len, args.max_len)
    print(f"Candidate words: {len(candidates)}")

    groups = group_by_cost(entries)
    overall_start = time.perf_counter()

    # Keep results in original order for neat output.
    results: Dict[str, Tuple[str, float]] = {}

    for cost in sorted(groups.keys()):
        group_entries = groups[cost]
        print(f"\nCracking cost group {cost} with {len(group_entries)} hashes...")
        group_results = crack_cost_group(
            cost,
            group_entries,
            candidates,
            args.workers,
            overall_start,
            args.progress_interval,
        )
        results.update(group_results)

    print("\nResults:")
    for username, _hash_str in entries:
        if username in results:
            password, elapsed = results[username]
            print(f"{username}: {password} found in {format_seconds(elapsed)}")
        else:
            print(f"{username}: NOT FOUND")

    return 0


if __name__ == "__main__":
    # Multiprocessing guard for Windows/macOS spawn behavior.
    sys.exit(main())
