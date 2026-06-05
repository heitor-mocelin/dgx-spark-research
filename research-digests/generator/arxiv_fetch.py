#!/usr/bin/env python3
"""Fetch + parse arXiv Atom API into JSON lines: {id,title,abstract,published,authors}."""
import sys, json, urllib.request, urllib.parse, xml.etree.ElementTree as ET
NS = {'a': 'http://www.w3.org/2005/Atom'}

def fetch(query, max_results=8, sort='relevance'):
    params = urllib.parse.urlencode({
        'search_query': f'all:{query}', 'start': 0,
        'max_results': max_results, 'sortBy': sort, 'sortOrder': 'descending'})
    url = f'http://export.arxiv.org/api/query?{params}'
    req = urllib.request.Request(url, headers={'User-Agent': 'inference-research/0.1'})
    with urllib.request.urlopen(req, timeout=40) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall('a:entry', NS):
        t = e.find('a:title', NS); s = e.find('a:summary', NS); i = e.find('a:id', NS)
        p = e.find('a:published', NS)
        if t is None or s is None or i is None:
            continue
        out.append({
            'id': i.text.strip(),
            'title': ' '.join(t.text.split()),
            'abstract': ' '.join(s.text.split()),
            'published': (p.text[:10] if p is not None else ''),
            'authors': [a.find('a:name', NS).text for a in e.findall('a:author', NS)][:6],
        })
    return out

if __name__ == '__main__':
    q = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for p in fetch(q, n):
        print(json.dumps(p, ensure_ascii=False))
