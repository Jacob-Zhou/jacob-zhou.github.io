import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import citation


PAPER_A = "9a6479bdd7605029d3972dea4c64a2e914f2755a"
PAPER_B = "46fe2ae301aeb75b25ebca0bdc26132ca46f5101"


class CitationUpdaterTests(unittest.TestCase):
    def test_extracts_ids_from_includes_and_semantic_scholar_urls(self):
        content = f"""
{{% include citation.html id=\"{PAPER_A}\" %}}
<a href=\"https://www.semanticscholar.org/paper/title/{PAPER_B}\">paper</a>
{{% include citation.html id=\"{PAPER_A}\" %}}
"""

        self.assertEqual(citation.extract_paper_ids(content), [PAPER_A, PAPER_B])

    def test_refresh_merges_fresh_counts_with_cached_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "index.md"
            cache_path = root / "citations.json"
            index_path.write_text(
                f'{{% include citation.html id="{PAPER_A}" %}}\n'
                f'{{% include citation.html id="{PAPER_B}" %}}\n',
                encoding="utf-8",
            )
            cache_path.write_text(
                json.dumps({PAPER_A: 1, PAPER_B: 10}), encoding="utf-8"
            )

            with patch.object(
                citation, "fetch_citations", return_value={PAPER_A: 7}
            ):
                changed = citation.refresh_citations(index_path, cache_path)

            self.assertTrue(changed)
            self.assertEqual(
                json.loads(cache_path.read_text(encoding="utf-8")),
                {PAPER_A: 7, PAPER_B: 10},
            )

    def test_refresh_keeps_cache_when_api_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "index.md"
            cache_path = root / "citations.json"
            index_path.write_text(
                f'{{% include citation.html id="{PAPER_A}" %}}\n',
                encoding="utf-8",
            )
            cache_path.write_text(json.dumps({PAPER_A: 7}), encoding="utf-8")

            with patch.object(
                citation,
                "fetch_citations",
                side_effect=citation.CitationError("temporarily unavailable"),
            ):
                changed = citation.refresh_citations(index_path, cache_path)

            self.assertFalse(changed)
            self.assertEqual(
                json.loads(cache_path.read_text(encoding="utf-8")),
                {PAPER_A: 7},
            )


if __name__ == "__main__":
    unittest.main()
