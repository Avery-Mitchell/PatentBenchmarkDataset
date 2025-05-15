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

def parse_txt_file(path: str) -> dict[str, any]:
    """
    Parses a Snopes article text file and extracts relevant information

    Arguments:
        path: path to the text file

    Returns:
        A dictionary containing the parsed data in a JSON-ready format
    """
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

    # Build the JSON-ready dictionary
    return {
        "claim":       hdr['claim'],
        "label":       hdr['rating'],
        "source":      "Snopes",
        "url":         hdr['url'],
        "explanation": "",      # Needs to be manually annotated for correctness
        "evidence":    [],      # Needs to be manually annotated for correctness
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

def write_to_json():
    """
    Reads all text files in the input directory, parses them, and writes the output to JSON files
    
    Arguments:
        None

    Returns:
        None
    """
    
    for fn in os.listdir(INPUT_DIR):
        if not fn.lower().endswith('.txt'):
            continue
        in_path  = os.path.join(INPUT_DIR, fn)
        out_path = os.path.join(OUTPUT_DIR, fn[:-4] + '.json')
        data     = parse_txt_file(in_path)

        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)

        print(f"Wrote {out_path}")

def combine_json_files(input_folder: str, output_file: str) -> None:
    """
    Combines multiple JSON files into a single JSON file.

    Arguments:
        input_folder: path to the folder containing JSON files
        output_file: path to the output JSON file

    Returns:
        None
    """

    combined_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            filepath = os.path.join(input_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    combined_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(combined_data, f_out, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    combine_json_files("annotated_articles", "dataset.json")


