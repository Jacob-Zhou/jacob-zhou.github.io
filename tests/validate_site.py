"""Validate machine-readable files emitted by the Jekyll build."""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"


class JsonLdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.documents: list[str] = []
        self._parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._parts is not None:
            self.documents.append("".join(self._parts))
            self._parts = None


def is_absolute_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def load_json(relative_path: str) -> object:
    return json.loads((SITE / relative_path).read_text(encoding="utf-8"))


def validate() -> None:
    feed = load_json("feed.json")
    profile = load_json("data/profile.json")
    projects = load_json("data/projects.json")
    publications = load_json("data/publications.json")

    assert isinstance(feed, dict)
    assert feed["version"] == "https://jsonfeed.org/version/1.1"
    assert all(is_absolute_http_url(item["url"]) for item in feed["items"])
    assert isinstance(profile, dict) and profile["@type"] == "ProfilePage"
    assert isinstance(projects, dict) and projects["schema_version"] == "1.0"
    assert len(projects["projects"]) >= 7
    assert len(projects["press_and_interviews"]) >= 6
    assert isinstance(publications, dict) and publications["schema_version"] == "1.0"
    assert publications["publications"]
    publication_ids = [
        item["semantic_scholar_id"] for item in publications["publications"]
    ]
    assert len(publication_ids) == len(set(publication_ids))
    assert all(len(paper_id) == 40 for paper_id in publication_ids)

    sitemap = ET.parse(SITE / "sitemap.xml")
    sitemap_urls = [element.text for element in sitemap.findall(".//{*}loc")]
    assert sitemap_urls and all(url and is_absolute_http_url(url) for url in sitemap_urls)

    atom = ET.parse(SITE / "feed.xml")
    atom_links = [element.get("href") for element in atom.findall(".//{*}link")]
    assert atom_links and all(url and is_absolute_http_url(url) for url in atom_links)

    for html_path in [SITE / "index.html", *SITE.glob("20*/*/*/*.html")]:
        parser = JsonLdParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        assert parser.documents, f"No JSON-LD found in {html_path}"
        for document in parser.documents:
            json.loads(document)

    llms = (SITE / "llms.txt").read_text(encoding="utf-8")
    assert llms.startswith("# Houquan Zhou")
    assert "/data/publications.json" in llms
    assert "/data/projects.json" in llms
    llms_urls = re.findall(r"https?://[^\s)]+", llms)
    assert llms_urls and all(is_absolute_http_url(url) for url in llms_urls)


if __name__ == "__main__":
    try:
        validate()
    except (AssertionError, json.JSONDecodeError, OSError, ET.ParseError) as error:
        print(f"Generated site validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Generated machine-readable files are valid")
