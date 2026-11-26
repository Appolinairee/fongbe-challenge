#!/usr/bin/env python3
"""Transcrit le test set Fongbe et produit submission.csv (format Zindi).

Usage :
    python scripts/predict_test.py --model ckpt_hf/v2_kaggle/outputs/w2vbert/model \
        --csv data/Test.csv --root data/test_audio --out out_test.csv

- fp32, batching dynamique par longueur, checkpoint incremental (reprenable)
- en sortie : out_test.csv avec colonnes id,transcription (brut)
"""
import argparse, csv, os, sys, time
import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from syllabify import syllabify  # noqa: E402


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="out_test.csv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2BertProcessor.from_pretrained(a.model)
    model = Wav2Vec2BertForCTC.from_pretrained(a.model).to(device).eval()

    rows = load_rows(a.csv)
    todo = [r for r in rows if r["id"] not in done_ids(a.out)]
    print(f"{len(rows)} fichiers, {len(todo)} à transcrire", flush=True)

    # tri par durée pour des batches homogènes
    infos = []
    for r in todo:
        p = os.path.join(a.root, r["audio_path"])
        try:
            info = sf.info(p)
            infos.append((info.frames / info.samplerate, p, r["id"]))
        except Exception as e:
            print("skip", r["id"], e, flush=True)
            infos.append((0.0, p, r["id"]))
    infos.sort()

    new = not os.path.exists(a.out)
    fout = open(a.out, "a", encoding="utf-8", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(["id", "transcription"]); fout.flush()

    t0 = time.time(); n = 0
    i = 0
    while i < len(infos):
        # batch = fichiers de durées proches, cap ~40 s total
        batch, total = [], 0.0
        while i < len(infos) and len(batch) < a.batch and total < 40.0:
            d, p, rid = infos[i]; i += 1
            if d <= 0:
                continue
            batch.append((p, rid)); total += d
        if not batch:
            continue
        waves = []
        for p, _ in batch:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            waves.append(x)
        feats = processor(waves, sampling_rate=16000, return_tensors="pt",
                          padding=True)
        with torch.no_grad():
            logits = model(input_features=feats.input_features.to(device),
                           attention_mask=feats.attention_mask.to(device)).logits
        ids = logits.argmax(-1).cpu()
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        for (_, rid), h in zip(batch, hyps):
            w.writerow([rid, h]); n += 1
        fout.flush()
        el = time.time() - t0
        print(f"{n} ok · {el/60:.1f} min · {n/max(el,1)*3600:.0f} utt/h", flush=True)
    fout.close()

    # submission Zindi
    subs = {r["id"]: r["transcription"] for r in load_rows(a.out)}
    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["ID", "Target"])
        for r in rows:
            w.writerow([r["id"], syllabify(subs.get(r["id"], ""))])
    print("submission.csv écrite", flush=True)


if __name__ == "__main__":
    main()
