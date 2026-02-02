"""
Common helpers shared by task1.py and task2.py.

Keep this file small and focused so both tasks can import utilities without
copying code.
"""
import hashlib
import secrets
from typing import Optional


def sha256_digest(data: bytes) -> bytes:
    """Return the raw 32-byte SHA-256 digest for the given data."""
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 digest as a 64-hex-character string."""
    return hashlib.sha256(data).hexdigest()


def truncate_digest_bits(digest: bytes, bits: int) -> int:
    """
    Truncate a SHA-256 digest to the top `bits` bits and return it as an int.

    We consistently keep the most-significant bits so that truncation is
    deterministic across runs.
    """
    if bits < 1 or bits > 256:
        raise ValueError("bits must be between 1 and 256")
    full_value = int.from_bytes(digest, "big")
    return full_value >> (256 - bits)


def truncated_sha256_int(data: bytes, bits: int) -> int:
    """Compute SHA-256(data) and return the truncated integer value."""
    return truncate_digest_bits(sha256_digest(data), bits)


def flip_one_bit(data: bytes, bit_index: Optional[int] = None) -> tuple[bytes, int]:
    """
    Return a copy of `data` with exactly one bit flipped.

    The returned tuple is (mutated_bytes, flipped_bit_index). Bit numbering is
    across the entire byte array, starting from the most-significant bit of the
    first byte.
    """
    if not data:
        raise ValueError("data must be non-empty to flip a bit")

    total_bits = len(data) * 8
    if bit_index is None:
        bit_index = secrets.randbelow(total_bits)
    if bit_index < 0 or bit_index >= total_bits:
        raise ValueError("bit_index is out of range")

    byte_index = bit_index // 8
    bit_in_byte = 7 - (bit_index % 8)

    mutated = bytearray(data)
    mutated[byte_index] ^= 1 << bit_in_byte
    return bytes(mutated), bit_index


def hamming_distance_bits(a: bytes, b: bytes) -> int:
    """Compute the Hamming distance (in bits) between two equal-length bytes."""
    if len(a) != len(b):
        raise ValueError("inputs must be the same length to compute distance")
    distance = 0
    for x, y in zip(a, b):
        distance += bin(x ^ y).count("1")
    return distance


def random_bytes(length: int) -> bytes:
    """Return cryptographically-random bytes of the requested length."""
    if length <= 0:
        raise ValueError("length must be positive")
    return secrets.token_bytes(length)


def describe_bytes(data: bytes, max_len: int = 32) -> str:
    """
    Return a compact hex preview of bytes for logging.

    The preview is truncated after `max_len` bytes to keep output readable.
    """
    if len(data) <= max_len:
        return data.hex()
    return f"{data[:max_len].hex()}...({len(data)} bytes)"


def format_seconds(seconds: float) -> str:
    """Format a duration in seconds with millisecond precision."""
    return f"{seconds:.6f}s"
