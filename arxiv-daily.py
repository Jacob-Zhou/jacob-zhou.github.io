# -*- coding: utf-8 -*-

import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Tuple

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
    'Yao Fu', 'Yang Feng', 'Yoon Kim', 'Yuntian Deng'
]

CONFS = ['ACL', 'EMNLP', 'NAACL', 'COLING', 'ICLR', 'NIPS', 'NEURIPS', 'ICML', 'JMLR']
CLASSES = ['cs.CL', 'cs.LG']


def red(t: str) -> str:
    return f'<strong class="highlight">*{t}*</strong>'


def text_title(t: str) -> str:
    return f'<code class="title">{t}</code>'

def texttt(t: str) -> str:
    return f'<code>{t}</code>'

def link(t: str) -> str:
    return f'[{t}]({t})'


def match(t: str, keys: Iterable) -> Tuple[str, bool]:
    raw = t
    for key in keys:
        t = re.sub(fr'\b{key}\b', lambda m: red(m.group()), t, flags=re.I)
    return t, (raw != t)


papers = defaultdict(dict)
for day in range(7):
    for name in CLASSES:
        search = arxiv.Search(query=name, sort_by=arxiv.SortCriterion.LastUpdatedDate)
        for paper in search.results():
            date = datetime.now(paper.updated.tzinfo) - timedelta(day)
            if paper.updated.date() < date.date():
                break
            if any(paper.title in i for i in papers.values()):
                continue
            date = date.strftime("%a, %d %b %Y")
            any_match = False
            title, matched = match(paper.title, KEYS)
            any_match = any_match or matched
            authors, matched = match(', '.join([f"{author}" for author in paper.authors]), AUTHORS)
            any_match = any_match or matched
            abstract, matched = match(paper.summary, KEYS)
            any_match = any_match or matched
            comments, comment_matched = match(paper.comment or '', CONFS)
            any_match = any_match or comment_matched
            if not any_match:
                continue
            papers[date][paper.title] = f'* **{title}** <br>\n'
            papers[date][paper.title] += f'{text_title("[AUTHORS]")}{authors} <br>\n'
            if matched:
                papers[date][paper.title] += f'{text_title("[ABSTRACT]")}{abstract} <br>\n'
            if comments:
                papers[date][paper.title] += f'{text_title("[COMMENTS]")}{comments} <br>\n'
            papers[date][paper.title] += f'{text_title("[LINK]")}{link(paper.entry_id)} <br>\n'
            papers[date][paper.title] += f'{text_title("[DATE]")}{paper.updated} <br>\n'
            categories = '    '.join([texttt(c) for c in paper.categories if c in CLASSES])
            papers[date][paper.title] += f'{text_title("[CATEGORIES]")}{categories} <br>\n'

with open('arxiv.md', 'w') as f:
    f.write('---\nlayout: default\n---\n\n')
    f.write('<details><summary>Contents</summary><ul>')
    for date in papers:
        f.write(f'<li><a href="#{date.replace(" ", "-").replace(",", "").lower()}">{date}</a></li>')
    f.write('</ul></details><br>\n\n')
    for date in papers:
        f.write(f'#### {date}\n\n')
        for title, paper in papers[date].items():
            f.write(paper.replace('{', '\{').replace('}', '\}') + '\n\n')
