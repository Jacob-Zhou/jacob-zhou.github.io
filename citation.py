# -*- coding: utf-8 -*-

"""Refresh citation counts used by the Jekyll site.

The publication page declares Semantic Scholar paper IDs through the
``citation.html`` include. This script looks up those exact papers with the
Semantic Scholar batch API and stores the last known counts in Jekyll's data
directory. A temporary API failure leaves the existing cache untouched so it
does not prevent the rest of the site from being deployed.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "index.md"
CACHE_PATH = ROOT / "_data" / "citations.json"
API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=citationCount"

PAPER_ID_RE = re.compile(r"^[0-9a-f]{40}$")
INCLUDE_RE = re.compile(
    r"\{%\s*include\s+citation\.html\s+id=[\"']([0-9a-f]{40})[\"']\s*%\}"
)
SEMANTIC_SCHOLAR_URL_RE = re.compile(
    r"https://www\.semanticscholar\.org/paper/[^\"'\s<>]*?([0-9a-f]{40})(?=[/?#\"'\s<>]|$)"
)

MAX_BATCH_SIZE = 500
MAX_ATTEMPTS = 4
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class CitationError(RuntimeError):
    """Raised when citation data cannot be safely refreshed."""


def extract_paper_ids(content: str) -> list[str]:
    """Return unique Semantic Scholar IDs in their page order."""

    ids = INCLUDE_RE.findall(content)
    ids.extend(SEMANTIC_SCHOLAR_URL_RE.findall(content))
    return list(dict.fromkeys(paper_id.lower() for paper_id in ids))


def load_cached_citations(path: Path = CACHE_PATH) -> dict[str, int]:
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CitationError(f"Citation cache must contain a JSON object: {path}")

    counts: dict[str, int] = {}
    for paper_id, count in data.items():
        if (
            isinstance(paper_id, str)
            and PAPER_ID_RE.fullmatch(paper_id)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        ):
            counts[paper_id] = count
    return counts


def _retry_delay(error: urllib.error.HTTPError | None, attempt: int) -> float:
    if error is not None and error.headers is not None:
        retry_after = error.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 30.0)
            except ValueError:
                pass
    return min(2 ** (attempt - 1), 8)


def fetch_citations(
    paper_ids: Iterable[str], api_key: str | None = None
) -> dict[str, int]:
    ids = list(dict.fromkeys(paper_ids))
    if not ids:
        return {}
    if len(ids) > MAX_BATCH_SIZE:
        raise CitationError(
            f"Semantic Scholar accepts at most {MAX_BATCH_SIZE} paper IDs per batch"
        )

    payload = json.dumps({"ids": ids}).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "jacob-zhou.github.io-citation-updater/2.0",
    }
    if api_key:
        headers["x-api-key"] = api_key

    result: object = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            API_URL, data=payload, headers=headers, method="POST"
        )
        http_error: urllib.error.HTTPError | None = None
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            http_error = error
            if error.code not in RETRYABLE_STATUS_CODES or attempt == MAX_ATTEMPTS:
                raise CitationError(
                    f"Semantic Scholar returned HTTP {error.code}"
                ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == MAX_ATTEMPTS:
                raise CitationError(
                    f"Semantic Scholar request failed: {error}"
                ) from error

        delay = _retry_delay(http_error, attempt)
        print(
            f"Citation request failed; retrying in {delay:g}s "
            f"({attempt}/{MAX_ATTEMPTS})",
            file=sys.stderr,
        )
        time.sleep(delay)
    else:  # pragma: no cover - the loop always returns or raises
        raise CitationError("Semantic Scholar request failed")

    if not isinstance(result, list):
        raise CitationError("Semantic Scholar returned an unexpected response")

    requested = set(ids)
    counts: dict[str, int] = {}
    for paper in result:
        if not isinstance(paper, dict):
            continue
        paper_id = paper.get("paperId")
        count = paper.get("citationCount")
        if (
            paper_id in requested
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
        ):
            counts[paper_id] = count

    if not counts:
        raise CitationError("Semantic Scholar returned no usable citation counts")
    return counts


def save_citations(counts: dict[str, int], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(counts, indent=2, sort_keys=True) + "\n"
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(path)


def refresh_citations(
    index_path: Path = INDEX_PATH, cache_path: Path = CACHE_PATH
) -> bool:
    content = index_path.read_text(encoding="utf-8")
    paper_ids = extract_paper_ids(content)
    if not paper_ids:
        raise CitationError(f"No Semantic Scholar paper IDs found in {index_path}")

    try:
        cached = load_cached_citations(cache_path)
    except (CitationError, json.JSONDecodeError, OSError) as error:
        print(f"Warning: ignoring invalid citation cache: {error}", file=sys.stderr)
        cached = {}

    try:
        fetched = fetch_citations(paper_ids, os.environ.get("S2_API_KEY"))
    except CitationError as error:
        if cached:
            print(
                f"Warning: {error}; keeping {len(cached)} cached citation counts",
                file=sys.stderr,
            )
            return False
        raise

    missing = [paper_id for paper_id in paper_ids if paper_id not in fetched]
    if missing:
        print(
            "Warning: no fresh count for "
            + ", ".join(missing)
            + "; using cached values",
            file=sys.stderr,
        )

    updated = {paper_id: cached.get(paper_id, 0) for paper_id in paper_ids}
    updated.update(fetched)

    if updated == cached and cache_path.exists():
        print(f"Citation cache is current ({len(updated)} papers)")
        return False

    save_citations(updated, cache_path)
    print(f"Updated citation cache for {len(updated)} papers")
    return True


def main() -> int:
    try:
        refresh_citations()
    except (CitationError, OSError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
