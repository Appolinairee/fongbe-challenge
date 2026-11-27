"""Décode KenLM des logits .npy -> kenlm_test.csv (id, transcription brute).

Usage: .venv/bin/python scripts/decode_kenlm.py --logits-dir lm/test_logits --out lm/kenlm_test.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyctcdecode import build_ctcdecoder

ROOT = Path(__file__).resolve().parent.parent
LM = ROOT / 'lm'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--beta', type=float, default=1.5)
    ap.add_argument('--beam-width', type=int, default=100)
    ap.add_argument('--shard', default='0/1', help='i/n : traite le shard i parmi n')
    args = ap.parse_args()

    logits_dir = Path(args.logits_dir)
    labels = json.load(open(LM / 'labels.json', encoding='utf-8'))
    unigrams = set((LM / 'unigrams.txt').read_text(encoding='utf-8').split())

    dec = build_ctcdecoder(labels, kenlm_model_path=str(LM / 'lm.bin'),
                           unigrams=unigrams, alpha=args.alpha, beta=args.beta)
    files = sorted(logits_dir.glob('*.npy'))
    si, sn = (int(x) for x in args.shard.split('/'))
    files = [f for i, f in enumerate(files) if i % sn == si]
    print(f'{len(files)} logits à décoder (shard {si}/{sn})', flush=True)
    out = {}
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            lg = np.load(f).astype(np.float32)
            out[f.stem] = dec.decode(lg, beam_width=args.beam_width)
        except Exception as e:
            print('ERR', f.name, e, flush=True)
            out[f.stem] = ''
        if (i + 1) % 200 == 0 or i == len(files) - 1:
            el = time.time() - t0
            print(f'{i+1}/{len(files)} · {el/60:.1f} min · ETA {(len(files)-i-1)/(el/max(i+1,1))*60/60:.1f} h', flush=True)

    with open(args.out, 'w', encoding='utf-8', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(['id', 'transcription'])
        for rid, h in out.items():
            w.writerow([rid, h])
    print('écrit', args.out)


if __name__ == '__main__':
    main()
