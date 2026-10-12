"""Helpers for the Cycle 3 opener table. python -m em.doc_addcol <read_doc.json> start|fill <header>
start: print table start index and revision. fill: print requests filling the last (empty) column with header + Pending."""
import json, sys
d = json.load(open(sys.argv[1])); d = d.get('content', d)
def text(c): return ''.join(e.get('textRun', {}).get('content', '') for p in c['content'] for e in p.get('paragraph', {}).get('elements', []))
for el in d['tabs'][0]['documentTab']['body']['content']:
    t = el.get('table')
    if t and text(t['tableRows'][0]['tableCells'][0]).startswith('Opener'):
        if sys.argv[2] == 'start':
            print(json.dumps({"start": el['startIndex'], "ncols": len(t['tableRows'][0]['tableCells']), "rev": d['revisionId']}))
        else:
            out = []
            for i, row in reversed(list(enumerate(t['tableRows']))):
                c = row['tableCells'][-1]; s = c['content'][0]['startIndex']; v = sys.argv[3] if i == 0 else 'Pending'
                assert text(c).strip() == ''
                out.append({"insertText": {"location": {"index": s, "tabId": "t.0"}, "text": v}})
                if i == 0: out.append({"updateTextStyle": {"range": {"startIndex": s, "endIndex": s + len(v), "tabId": "t.0"}, "textStyle": {"bold": True}, "fields": "bold"}})
            print(json.dumps({"rev": d['revisionId'], "requests": out}))
