# -*- coding: utf-8 -*-

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple
import unicodedata

import arxiv

KEYS = [
    'adversarial', 'algebraic', 'algebratic', 'auto-encoding', 'autoencoder', 'autoencoding',
    'autoregressive',
    'parse', 'parser', 'parsing',
    'evaluation', 'evaluating', 'evaluations', 'benchmark', 'benchmarks', 
    'grammar', 'grammatical', 'grammatical error', 'grammar error correction', 
    'decoding', 
    'agent', 'agents', 
    'feedback', 'feedbacks', 
    'seq2seq', 'sequence', 'sequence to sequence', 'sequence-to-sequence',
    'stochasticity', 'struct', 'structural', 'structure', 'structured', 'syntax',
    'question generation',
    'compression', 'compressor', 'compressors'
    # 'transducer', 'transduction', 'transformer', 'translation', 
]

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
    'Luke Zettlemoyer'
]

CONFS = ['ACL', 'EMNLP', 'NAACL', 'COLING', 'ICLR', 'NIPS', 'NEURIPS', 'ICML', 'JMLR']
CLASSES = ['cs.CL', 'cs.LG']


def red(t: str) -> str:
    return f'<strong class="highlight"><em>{t}</em></strong>'


def text_title(t: str) -> str:
    return f'<code class="title">{t}</code>'

def texttt(t: str) -> str:
    return f'<code>{t}</code>'

def link(t: str) -> str:
    return f'[{t}]({t})'

def normalize_id(t: str) -> str:
    t = unicodedata.normalize('NFD', t)
    t = ''.join([c for c in t if not unicodedata.combining(c)])
    t = t.lower()
    # space to _
    t = re.sub(r'\s+', '_', t)
    # escape special characters
    # t = re.sub(r'([\\`*_{}[\]()#+-.!])', r'\\\1', t)
    return t

def upper_first(t: str) -> str:
    return t[0].upper() + t[1:]

def match(t: str, keys: Iterable) -> Tuple[str, bool]:
    # raw = t
    matched_keys = []
    for key in keys:
        if re.search(fr'\b{key}\b', t, flags=re.I):
            matched_keys.append(key)
            t = re.sub(fr'\b{key}\b', lambda m: red(m.group()), t, flags=re.I)
    return t, matched_keys

def cover_timezones(date: datetime) -> datetime:
    # to UTF+8
    return date.astimezone(timezone(timedelta(hours=8)))

# papers = defaultdict(defaultdict(dict))
papers = defaultdict(lambda: defaultdict(dict))
# for day in range(7):
max_day = 7
available_tabs = set()
for name in CLASSES:
    search = arxiv.Search(query=name, sort_by=arxiv.SortCriterion.LastUpdatedDate)
    for paper in search.results():
        # date = datetime.now(paper.updated.tzinfo) - timedelta(day)
        date = datetime.now(paper.updated.tzinfo) - timedelta(max_day)
        if paper.updated.date() < date.date():
            break
        # Convert to UTC+8
        # date = cover_timezones(paper.updated).strftime("%a, %d %b %Y")
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
        for key in any_match:
            papers[key][date][paper.title] = f'<strong>{title}</strong><br>\n'
            papers[key][date][paper.title] += f'{text_title("[AUTHORS]")}{authors} <br>\n'
            papers[key][date][paper.title] += f'{text_title("[ABSTRACT]")}{abstract} <br>\n'
            # papers[key][date][paper.title] += f'{text_title("[ABSTRACT]")}<details><summary>Click to expand</summary>{abstract}</details> <br>\n'
            if comments:
                papers[key][date][paper.title] += f'{text_title("[COMMENTS]")}{comments} <br>\n'
            papers[key][date][paper.title] += f'{text_title("[LINK]")}{link(paper.entry_id)} <br>\n'
            # papers[key][date][paper.title] += f'{text_title("[DATE]")}{paper.updated} <br>\n'
            papers[key][date][paper.title] += f'{text_title("[DATE]")}{cover_timezones(paper.updated)} <br>\n'
            categories = '    '.join([texttt(c) for c in paper.categories if c in CLASSES])
            papers[key][date][paper.title] += f'{text_title("[CATEGORIES]")}{categories} <br>\n'

with open('arxiv.md', 'w') as f:
    f.write('---\nlayout: default\n---\n\n')
    # f.write('<details><summary>Contents</summary><ul>')
    # for date in sorted(papers.keys(), reverse=True):
    #     f.write(f'<li><a href="#{date.replace(" ", "-").replace(",", "").lower()}">{date}</a></li>')
    # f.write('</ul></details><br>\n\n')
    # for date in sorted(papers.keys(), reverse=True):
    #     f.write(f'#### {date}\n\n')
    #     for title, paper in papers[key][date].items():
    #         f.write(paper.replace('{', '\{').replace('}', '\}') + '\n\n')
    f.write('<ul class="tab-nav">\n')
    for i, domain in enumerate([KEYS, AUTHORS, CONFS]):
        for i, tab in enumerate(sorted(available_tabs)):
            if tab not in domain:
                continue
            f.write(f'<li><a class="button{" active" if i == 0 else ""}" href="#{normalize_id(tab)}">{upper_first(tab)}</a></li>\n')
        f.write(f'<hr class="tab-nav-divider {" last" if i == 2 else ""}">\n')
    f.write('</ul>\n\n')

    f.write('<div class="tab-content">\n')
    for i, tab in enumerate(sorted(available_tabs)):
        f.write(f'<div class="tab-pane {" active" if i == 0 else ""}" id="{normalize_id(tab)}">\n')
        for date in sorted(papers[tab].keys(), reverse=True):
            # f.write(f'#### {date}\n\n')
            f.write(f'<details><summary class="date">{date}</summary>\n\n')
            f.write('<ul>\n')
            for title, paper in papers[tab][date].items():
                f.write('<li class="arxiv-paper">\n')
                f.write(paper.replace('{', '\{').replace('}', '\}') + '\n\n')
                f.write('</li>\n')
            f.write('</ul>\n')
            f.write('</details>\n\n')
        f.write('</div>\n')
    f.write('</div>\n')
