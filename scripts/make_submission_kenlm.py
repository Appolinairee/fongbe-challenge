"""kenlm_test.csv (id, transcription brute) -> submission.csv syllabifié."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
from syllabify import syllabify  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'lm' / 'kenlm_test.csv'
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'submission_kenlm.csv'

subs = {r['id']: r['transcription'] for r in csv.DictReader(open(src, encoding='utf-8'))}
rows = list(csv.DictReader(open(ROOT / 'data' / 'Test.csv', encoding='utf-8')))
with open(dst, 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID', 'Target'])
    for r in rows:
        w.writerow([r['id'], syllabify(subs.get(r['id'], ''))])
print('écrit', dst, len(rows), 'lignes')
