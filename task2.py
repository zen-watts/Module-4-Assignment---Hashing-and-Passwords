"""
Task 2: bcrypt dictionary cracker for a shadow-style file.

Dependencies (install once):
  pip install bcrypt nltk
  python -m nltk.downloader words
"""
import argparse
import math
import multiprocessing as mp
import sys
import time
import bcrypt
from common import format_seconds


ShadowEntry = tuple[str, str]
HashEntry = tuple[str, bytes]


def load_shadow_entries(path: str) -> list[ShadowEntry]:
    """Read shadow-style lines of the form Username:<bcrypt-hash>."""
    entries: list[ShadowEntry] = []
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


def build_candidate_list() -> list[bytes]:
    import nltk
    from nltk.corpus import words
    try:
        ws = words.words()
    except LookupError:
        nltk.download("words", quiet=True)
        ws = words.words()

    cand = {s.lower() for s in ws if s.isalpha() and 6 <= len(s) <= 10}
    return [w.encode("utf-8") for w in sorted(cand)] # we sort for consistency


def group_by_cost(entries: list[ShadowEntry]) -> dict[int, list[ShadowEntry]]:
    """Group shadow entries by bcrypt cost (work factor)."""
    groups: dict[int, list[ShadowEntry]] = {}
    for username, hash_str in entries:
        cost = bcrypt_cost(hash_str)
        groups.setdefault(cost, []).append((username, hash_str))
    return groups


def worker_crack(
    hashes: list[HashEntry],
    candidates: list[bytes],
    start_idx: int,
    end_idx: int,
    solved: "mp.managers.DictProxy",
    stop_event: "mp.Event",
    progress_counter: "mp.Value",
    group_start: float,
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
                    elapsed = time.perf_counter() - group_start
                    solved[username] = (candidate.decode("utf-8"), elapsed)

        if local_checks >= 200:
            with progress_counter.get_lock():
                progress_counter.value += local_checks
            local_checks = 0

    if local_checks:
        with progress_counter.get_lock():
            progress_counter.value += local_checks


def crack_cost_group(
    cost: int,
    entries: list[ShadowEntry],
    candidates: list[bytes],
    workers: int,
    group_start: float,
) -> dict[str, tuple[str, float]]:
    """
    Crack all hashes for a single bcrypt cost group using multiprocessing.
    """
    total_candidates = len(candidates)
    if total_candidates == 0:
        return {}
    total_checks = total_candidates * len(entries)

    manager = mp.Manager()
    solved = manager.dict()  # username -> (password, elapsed_seconds)
    progress_counter = mp.Value("Q", 0)
    stop_event = mp.Event()

    # Prepare hash bytes once to avoid re-encoding.
    hash_entries: list[HashEntry] = [
        (username, hash_str.encode("utf-8")) for username, hash_str in entries
    ]

    workers = max(1, min(workers, total_candidates))
    chunk_size = math.ceil(total_candidates / workers)

    processes: list[mp.Process] = []
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
                stop_event,
                progress_counter,
                group_start,
            ),
        )
        processes.append(process)
        process.start()
    
    # Monitor progress with periodic reports.
    ping_interval = 15.0
    last_report_time = time.perf_counter()
    last_report_checks = 0

    while True:
        if len(solved) >= len(entries):
            stop_event.set()

        current_checks = progress_counter.value
        if stop_event.is_set() or current_checks >= total_checks:
            break

        now = time.perf_counter()
        if now - last_report_time >= ping_interval:
            interval_checks = current_checks - last_report_checks
            interval_time = now - last_report_time
            checks_per_sec = interval_checks / interval_time if interval_time > 0 else 0.0
            print(
                f"Progress: checks={current_checks:>10,d} | "
                f"rate={checks_per_sec:>8.1f} checks/sec | "
                f"solved={len(solved)}/{len(entries)}",
                flush=True,
            )
            last_report_time = now
            last_report_checks = current_checks

        time.sleep(0.2)

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
        "-w",
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes (default: CPU count)",
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
    candidates = build_candidate_list()
    print(f"Candidate words: {len(candidates)}")

    groups = group_by_cost(entries)
    
    # Keep results in original order for neat output.
    results: dict[str, tuple[str, float]] = {}

    for cost in sorted(groups.keys()):
        group_entries = groups[cost]
        print(f"\nCracking cost group {cost} with {len(group_entries)} hashes...")
        group_start = time.perf_counter()
        group_results = crack_cost_group(
            cost,
            group_entries,
            candidates,
            args.workers,
            group_start,
        )
        results.update(group_results)

    print("Note: per-user crack time is measured from the start of that user's cost-group run.")
    print("\nResults:")
    for username, _hash_str in entries:
        if username in results:
            password, elapsed = results[username]
            print(f"{username}: {password} found in {format_seconds(elapsed)}")
        else:
            print(f"{username}: NOT FOUND")

    return 0


if __name__ == "__main__":
    # Multiprocessing guard
    sys.exit(main())
