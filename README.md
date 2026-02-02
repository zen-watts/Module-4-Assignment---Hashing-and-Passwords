# Module 4 Assignment: Hashing and Passwords

## Task 1: SHA-256 experiments

Run:

```bash
python task1.py
```

What it does:
- Prints 3 avalanche trials (16-byte inputs).
- Runs a collision sweep for truncated SHA-256 sizes 8..50 (step 2).
- Prints CSV-formatted output to stdout (`bits,trials,seconds`).

## Task 2: bcrypt dictionary cracker

Run (uses `shadow.txt` by default):

```bash
python task2.py
```

Optional arguments:
- `--shadow PATH` to use a different shadow-style file.
- `--workers N` to set the number of worker processes.
