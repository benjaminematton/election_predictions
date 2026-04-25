"""Shared download helpers for ingestion modules.

Goals:
  * Idempotent — re-running a download skips files that already exist
  * Streaming — supports >1GB FEC bulk files without loading into memory
  * Verifiable — optional sha256 check
  * Polite — single User-Agent, retries on transient errors

Used by census, fec, fec_ie, results, ratings, pvi.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import requests

USER_AGENT = "oath_score/0.1 (https://github.com/; backtesting research, contact via repo)"
DEFAULT_TIMEOUT_S = 60
MAX_RETRIES = 3
CHUNK_SIZE = 1 << 20  # 1 MB


def download_file(
    url: str,
    dest: Path,
    sha256: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Path:
    """Stream `url` to `dest`. Idempotent: skips if file exists and (optionally) checksum matches.

    Returns the resolved destination path.
    Raises requests.HTTPError on non-recoverable HTTP errors.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if sha256 is None or _sha256_of(dest) == sha256:
            return dest
        # Wrong checksum — drop and re-download
        dest.unlink()

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(
                url, stream=True, timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            ) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(dest)
            break
        except (requests.RequestException, OSError) as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                time.sleep(2**attempt)
            else:
                raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} tries") from last_err

    if sha256 is not None:
        actual = _sha256_of(dest)
        if actual != sha256:
            dest.unlink()
            raise RuntimeError(f"sha256 mismatch for {dest}: expected {sha256}, got {actual}")

    return dest


def unzip(archive: Path, dest_dir: Path, members: list[str] | None = None) -> list[Path]:
    """Extract `archive` to `dest_dir`. If `members` given, extract only those.

    Returns list of extracted paths.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        names = members if members is not None else zf.namelist()
        for name in names:
            zf.extract(name, dest_dir)
            extracted.append(dest_dir / name)
    return extracted


@contextmanager
def staged(path: Path, description: str = "") -> Iterator[Path]:
    """Context manager: yield `path` after announcing whether it was already staged.

    Useful for verbose download orchestration:
        with staged(dest, 'FEC indiv 2024') as p:
            if not p.exists():
                download_file(URL, p)
            process(p)
    """
    path = Path(path)
    if path.exists():
        print(f"[staged] {description or path.name}: present at {path}")
    else:
        print(f"[stage]  {description or path.name}: missing, will fetch")
    yield path


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_local(src: Path, dest: Path) -> Path:
    """Copy a local file into the data tree. Useful for tests with fixtures."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest
