import os
import re
import json
from datetime import date

INPUT_DIR = 'snopes_cleaned'
OUTPUT_DIR = 'annotated_articles'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex patterns to pull out each header field
HEADER_PATTERNS = {
    'article_title':  r'^Article Title:\s*(.*)$',
    'author':         r'^Author:\s*(.*)$',
    'date_published': r'^Date Published:\s*(.*)$',
    'claim':          r'^Claim:\s*(.*)$',
    'rating':         r'^Rating:\s*(.*)$',
    'url':            r'^URL:\s*(.*)$',
}

def parse_txt_file(path):
    with open(path, encoding='utf-8') as f:
        raw = f.read()

    # Extract headers
    hdr = {}
    for key, pat in HEADER_PATTERNS.items():
        m = re.search(pat, raw, re.MULTILINE)
        hdr[key] = m.group(1).strip() if m else ''

    # Isolate body text 
    parts = re.split(r'\nURL:.*\n+', raw, maxsplit=1)
    body = parts[1] if len(parts) > 1 else ''

    # Drop article text label
    body = re.sub(r'^\s*Article Text:\s*', '', body, flags=re.IGNORECASE)

    # Split into paragraphs for context
    paras = [p.strip() for p in body.strip().split('\n\n') if p.strip()]
    context_text = '\n\n'.join(paras) if paras else ''

    # Build the JSON-ready dict
    return {
        "claim":       hdr['claim'],
        "label":       hdr['rating'],
        "source":      "Snopes",
        "url":         hdr['url'],
        "explanation": "",      # Needs to be manually annotated for correctness
        "evidence":    [],      # 
        "context": {
            "article_title": hdr['article_title'],
            "text":          context_text
        },
        "metadata": {
            "date_published": hdr['date_published'],
            "date_collected": date.today().strftime('%m/%d/%Y'),
            "author":         hdr['author']
        }
    }

def main():
    for fn in os.listdir(INPUT_DIR):
        if not fn.lower().endswith('.txt'):
            continue
        in_path  = os.path.join(INPUT_DIR, fn)
        out_path = os.path.join(OUTPUT_DIR, fn[:-4] + '.json')
        data     = parse_txt_file(in_path)

        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)

        print(f"Wrote {out_path}")

if __name__ == '__main__':
    main()


