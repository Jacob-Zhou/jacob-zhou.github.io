# -*- coding: utf-8 -*-

import re
import json
import urllib.request

AUTHOR_ID = 50986473
API_URL = f"https://api.semanticscholar.org/graph/v1/author/{AUTHOR_ID}?fields=papers.citationCount,papers.paperId"

def fetch_citations():
    req = urllib.request.Request(API_URL)
    req.add_header('User-Agent', 'citation-updater/1.0')
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return {p['paperId']: p.get('citationCount', 0) for p in data.get('papers', []) if p.get('paperId')}

def update_cite(match, paper_map):
    full = match.group(0)
    pid_match = re.search(r'/([a-f0-9]{40})', full)
    if not pid_match:
        return full
    pid = pid_match.group(1)
    count = paper_map.get(pid)
    if count is None:
        return full
    return re.sub(r'<span class="cite-count">\d+</span>', f'<span class="cite-count">{count}</span>', full)

paper_map = fetch_citations()
print(f"Fetched citation counts for {len(paper_map)} papers")

with open('index.md', 'r') as f:
    content = f.read()

content = re.sub(
    r'<a href="https://www\.semanticscholar\.org/paper/[^"]*" class="pub-link pub-link--cite">cited <span class="cite-count">\d+</span></a>',
    lambda m: update_cite(m, paper_map),
    content
)

with open('index.md', 'w') as f:
    f.write(content)

print("Done")
