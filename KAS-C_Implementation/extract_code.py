"""
Script to add Code_Snippet column to the annotated CSV.
Code_Snippet is derived from a_post_commit with suppression annotations removed.
Comments are preserved. Justifications from annotations are kept as comments.
a_post_commit is kept unchanged.
"""

import csv
import re


def remove_suppression_annotations(code: str) -> str:
    """
    Remove suppression annotations from Java/code snippet, but:
    - Keep all comments intact
    - Extract justification strings from annotations and preserve them as comments
    - If no justification= param, drop the annotation silently
    """
    if not code or not code.strip():
        return code

    def extract_justification_as_comment(match: re.Match) -> str:
        """Pull out the justification= value as a comment, or drop silently if none."""
        annotation_text = match.group(0)

        # Only extract if there's an explicit justification="..." param
        just_match = re.search(r'justification\s*=\s*\{?"([^"]+)"', annotation_text)
        if just_match:
            return f'// {just_match.group(1)}'

        # No justification found — drop the annotation silently
        return ''

    # 1. Remove @SuppressFBWarnings(...) — extract justification as comment if present
    code = re.sub(
        r'@(?:edu\.umd\.cs\.findbugs\.annotations\.)?SuppressFBWarnings\s*\([^)]*(?:\([^)]*\)[^)]*)*\)',
        extract_justification_as_comment,
        code
    )

    # 2. Remove @SuppressWarnings(...) — extract justification as comment if present
    code = re.sub(
        r'@SuppressWarnings\s*\([^)]*(?:\([^)]*\)[^)]*)*\)',
        extract_justification_as_comment,
        code
    )

    # 3. Clean up lines that are now empty/whitespace after annotation removal,
    #    and collapse 3+ consecutive blank lines to 2
    lines = code.split('\n')
    cleaned = []
    prev_empty = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not prev_empty:
                cleaned.append(line.rstrip())
            prev_empty = True
        else:
            cleaned.append(line.rstrip())
            prev_empty = False

    result = '\n'.join(cleaned).strip()
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result


def process_csv(input_path: str, output_path: str | None = None) -> None:
    """
    Read CSV, add Code_Snippet column before Category, write output.
    If output_path is None, overwrites input file.
    """
    output_path = output_path or input_path

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Insert Code_Snippet before Category
        if 'Category' not in fieldnames:
            fieldnames.append('Code_Snippet')
        else:
            idx = fieldnames.index('Category')
            fieldnames.insert(idx, 'Code_Snippet')

        rows = []
        for row in reader:
            a_post_commit = row.get('a_post_commit', '') or ''
            code_snippet = remove_suppression_annotations(a_post_commit)
            row['Code_Snippet'] = code_snippet
            rows.append(row)

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} rows. Output written to {output_path}")


if __name__ == '__main__':
    import sys

    input_file = 'Evaluation_Set/Input.csv'
    output_file = 'Evaluation_Set/Evaluation_Set.csv'

    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    process_csv(input_file, output_file)