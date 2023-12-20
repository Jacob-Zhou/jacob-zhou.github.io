# -*- coding: utf-8 -*-

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple
import unicodedata
# from urllib.parse import quote

import arxiv

# KEYS = [
#     'adversarial', 'algebraic', 'algebratic', 'auto-encoding', 'autoencoder', 'autoencoding',
#     'autoregressive',
#     'parse', 'parser', 'parsing',
#     'evaluation', 'evaluating', 'evaluations', 'benchmark', 'benchmarks', 
#     'grammar', 'grammatical', 'grammatical error', 'grammar error correction', 
#     'decoding', 
#     'agent', 'agents', 
#     'feedback', 'feedbacks', 
#     'seq2seq', 'sequence', 'sequence to sequence', 'sequence-to-sequence',
#     'struct', 'structural', 'structure', 'structured', 'syntax',
#     'question generation',
#     'compression', 'compressor', 'compressors', 
#     'legal', 'law', 
# ]
KEYS = {
    # Adversarial related
    'adversarial': 'Adversarial',
    # Algebraic related
    'algebraic': 'Algebraic',
    'algebratic': 'Algebraic',
    # Auto-encoding related
    'auto-encoding': 'AutoEncoder',
    'autoencoder': 'AutoEncoder',
    'autoencoding': 'AutoEncoder',
    # Auto-regressive related
    'autoregressive': 'AutoRegressive',
    'auto-regressive': 'AutoRegressive',
    # Parsing related
    'parse': 'Parsing',
    'parser': 'Parsing',
    'parsing': 'Parsing',
    # Evaluation related
    'evaluation': 'Evaluation',
    'evaluating': 'Evaluation',
    'evaluations': 'Evaluation',
    'benchmark': 'Evaluation',
    'benchmarks': 'Evaluation',
    # Grammar related
    'grammar': 'Grammatical',
    'grammatical': 'Grammatical',
    'grammatical error': 'Grammatical',
    'grammar error correction': 'Grammatical',
    # Decoding related
    'decoding': 'Decoding',
    # Agent related
    'agent': 'Agent',
    'agents': 'Agent',
    # Feedback related
    'feedback': 'Feedback',
    'feedbacks': 'Feedback',
    # Seq2Seq related
    'seq2seq': 'Seq2Seq',
    'sequence to sequence': 'Seq2Seq',
    'sequence-to-sequence': 'Seq2Seq',
    # Structural related
    'struct': 'Structural',
    'structural': 'Structural',
    'structure': 'Structural',
    'structured': 'Structural',
    # Syntax related
    'syntax': 'Syntax',
    'syntactic': 'Syntax',
    # Question generation related
    'question generation': 'Question Generation',
    # Compression related
    'compression': 'Compression',
    'compressor': 'Compression',
    'compressors': 'Compression',
    # Legal related
    'legal': 'Legal',
    'law': 'Legal',
    # Rewrite related
    'rewrite': 'Rewrite',
    'rewriting': 'Rewrite',
    # Large language model related
    'large language model': 'Large Language Model',
    'large language models': 'Large Language Model',
    'LLM': 'Large Language Model',
    'LLMs': 'Large Language Model',
}

AUTHORS = [
    'Albert Gu', 'Alexander M. Rush', 'André F. T. Martins',
    'Bailin Wang',
    'Caio Corro', 'Chris Dyer', 'Christopher D. Manning', 'Christopher Ré',
    'Daniel Gildea', 'Daniel Y. Fu', 'David Chiang', 'David M. Blei',
    'Eduard Hovy',
    'Fei Huang',
    'Hao Zhou',
    'Giorgio Satta', 'Graham Neubig',
    'Ivan Titov',
    'Jan Buys', 'Jason Eisner', 'Justin T. Chiu',
    'Kevin Gimpel',
    'Lifu Tu', 'Lingpeng Kong',
    'Mathieu Blondel', 'Michael Collins', 'Mirella Lapata',
    'Noah A. Smith',
    'Percy Liang'
    'Ryan Cotterell',
    'Shay B. Cohen', 'Songlin Yang',
    'Tim Vieira', 'Tri Dao',
    'Vlad Niculae',
    'Xiang Lisa Li', 'Xuezhe Ma',
    'Yao Fu', 'Yang Feng', 'Yoon Kim', 'Yuntian Deng',
    'Luke Zettlemoyer',
    'Diyi Yang',
]

# CONFS = ['ACL', 'EMNLP', 'NAACL', 'COLING', 'ICLR', 'NIPS', 'NEURIPS', 'ICML', 'JMLR']

CONFS = {
    "ACL": "ACL",
    "EMNLP": "EMNLP",
    "NAACL": "NAACL",
    "COLING": "COLING",
    "ICLR": "ICLR",
    "NIPS": "NeurIPS",
    "NEURIPS": "NeurIPS",
    "ICML": "ICML",
    "JMLR": "JMLR"
}

CLASSES = ['cs.CL', 'cs.LG']


def red(t: str) -> str:
    return f'<strong class="highlight"><em>{t}</em></strong>'


def text_title(t: str) -> str:
    return f'<code class="title">{t}</code>'

def texttt(t: str) -> str:
    return f'<code>{t}</code>'

def link(t: str) -> str:
    # return f'[{t}]({t})'
    return f'<a href="{t}">{t}</a>'

def normalize_id(t: str) -> str:
    t = unicodedata.normalize('NFD', t)
    t = ''.join([c for c in t if not unicodedata.combining(c)])
    t = t.lower()
    # remove "." and ","
    t = t.replace('.', '')
    t = t.replace(',', '')
    # space to _
    t = re.sub(r'\s+', '_', t)
    # check if start with number
    if str.isdigit(t[0]):
        t = 'N' + t
    return t

def upper_first(t: str) -> str:
    return t[0].upper() + t[1:]

def match(t: str, keys: Iterable) -> Tuple[str, bool]:
    # raw = t
    matched_keys = []
    for key in keys:
        if re.search(fr'\b{key}\b', t, flags=re.I):
            if isinstance(keys, dict):
                matched_keys.append(keys[key])
            else:
                matched_keys.append(key)
            t = re.sub(fr'\b{key}\b', lambda m: red(m.group()), t, flags=re.I)
    return t, matched_keys

def cover_timezones(date: datetime) -> datetime:
    # to UTF+8
    return date.astimezone(timezone(timedelta(hours=8)))

papers = defaultdict(lambda: defaultdict(dict))
papers_by_date = defaultdict(dict)
max_day = 7
new_day = 2
available_tabs = set()
tabs_info = defaultdict(dict)
new_date = cover_timezones(datetime.now() - timedelta(new_day)).strftime("%Y %b %d, %a")
client = arxiv.Client(num_retries=10, page_size=500)
for name in CLASSES:
    search = arxiv.Search(query=name, sort_by=arxiv.SortCriterion.LastUpdatedDate)
    results = client.results(search)
    # for paper in search.results():
    max_iter = 1000
    while True:
        try:
            paper = next(results)
        except StopIteration:
            break
        except arxiv.arxiv.UnexpectedEmptyPageError:
            continue
        max_iter -= 1
        if max_iter < 0:
            break
        date = datetime.now(paper.updated.tzinfo) - timedelta(max_day)
        print(f"Find paper {paper.entry_id} {paper.title} {paper.updated}")
        if paper.updated.date() < date.date():
            break
        # Convert to UTC+8
        date = cover_timezones(paper.updated).strftime("%Y %b %d, %a")
        any_match = []
        title, matched = match(paper.title, KEYS)
        any_match.extend(matched)
        authors, matched = match(', '.join([f"{author}" for author in paper.authors]), AUTHORS)
        any_match.extend(matched)
        abstract, matched = match(paper.summary, KEYS)
        any_match.extend(matched)
        comments, comment_matched = match(paper.comment or '', CONFS)
        any_match.extend(comment_matched)
        if len(any_match) == 0:
            continue
        available_tabs.update(any_match)
        paper_content = f'<strong>{title}</strong><br>\n'
        paper_content += f'{text_title("[AUTHORS]")}{authors} <br>\n'
        paper_content += f'{text_title("[ABSTRACT]")}{abstract} <br>\n'
        if comments:
            paper_content += f'{text_title("[COMMENTS]")}{comments} <br>\n'
        paper_content += f'{text_title("[LINK]")}{link(paper.entry_id)} <br>\n'
        paper_content += f'{text_title("[DATE]")}{cover_timezones(paper.updated)} <br>\n'
        categories = '    '.join([texttt(c) for c in paper.categories if c in CLASSES])
        paper_content += f'{text_title("[CATEGORIES]")}{categories} <br>\n'
        for key in any_match:
            if date >= new_date:
                tabs_info[key]["new"] = True
            papers[key][date][paper.title] = paper_content
            papers_by_date[date][paper.title] = paper_content

with open('arxiv.md', 'w') as f:
    f.write('---\nlayout: default\n---\n\n')
    f.write('<ul class="tab-nav">\n')
    for i, domain in enumerate([KEYS, AUTHORS, CONFS]):
        if isinstance(domain, dict):
            domain = set(domain.values())
        for i, tab in enumerate(sorted(available_tabs)):
            if tab not in domain:
                continue
            f.write(f'<li><a class="button" href="#{normalize_id(tab)}">{upper_first(tab)}</a>')
            if tabs_info[tab].get("new", False):
                f.write('<span class="new-dot"> </span>')
            f.write('</li>\n')
        f.write('<li style="margin-right: auto;"><div></div></li>\n')
        f.write(f'<hr class="tab-nav-divider {" last" if i == 2 else ""}">\n')
    for i, date in enumerate(sorted(papers_by_date.keys(), reverse=True)):
        f.write(f'<li><a class="button{" active" if i == 0 else ""}" href="#{normalize_id(date)}">{date}</a></li>\n')
    f.write('</ul>\n\n')
    f.write(f'<hr>\n')

    f.write('<div class="tab-content">\n')
    for i, tab in enumerate(sorted(available_tabs)):
        f.write(f'<div class="tab-pane" id="{normalize_id(tab)}">\n')
        for j, date in enumerate(sorted(papers[tab].keys(), reverse=True)):
            f.write(f'<details {"open" if j == 0 else ""}><summary class="date">{date}</summary>\n\n')
            f.write('<ul>\n')
            for title, paper in papers[tab][date].items():
                f.write('<li class="arxiv-paper">\n')
                f.write(paper.replace('{', '\{').replace('}', '\}') + '\n\n')
                f.write('</li>\n')
            f.write('</ul>\n')
            f.write('</details>\n\n')
        f.write('</div>\n')
    for i, date in enumerate(sorted(papers_by_date.keys(), reverse=True)):
        f.write(f'<div class="tab-pane{" active" if i == 0 else ""}" id="{normalize_id(date)}">\n')
        f.write('<ul>\n')
        for title, paper in papers_by_date[date].items():
            f.write('<li class="arxiv-paper">\n')
            f.write(paper.replace('{', '\{').replace('}', '\}') + '\n\n')
            f.write('</li>\n')
        f.write('</ul>\n')
        f.write('</div>\n')
    f.write('</div>\n')
