#!/usr/bin/env python3
"""Évalue des modèles ASR Fongbe pré-entraînés sur les voix de validation.

Usage :
    python eval_fongbe_models.py --data-dir /content/fongbe --n 100

- Échantillonne N utterances `recommended_validation == 1` de Train_reliable.csv
- Transcrit avec chaque modèle du registre (zero-shot, sans fine-tuning)
- Note avec le score officiel (score.py) et affiche le classement.
"""

import argparse
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# name -> (type, hf_id)
# types : ctc (Wav2Vec2/Wav2Vec2Bert ForCTC), whisper, speechbrain,
#         auto (essaie ctc puis speechbrain), unsupported (skip)
MODELS = {
    # --- Apache-2.0 ---
    "dvoice_speechbrain":      ("speechbrain", "speechbrain/asr-wav2vec2-dvoice-fongbe"),
    "aioxlabs_dvoice":         ("speechbrain", "aioxlabs/dvoice-fongbe"),
    "fonxlsr":                 ("ctc", "chrisjay/fonxlsr"),
    "whettenr_diacritics":     ("speechbrain", "whettenr/asr-fon-with-diacritics"),
    "whettenr_no_diacritics":  ("speechbrain", "whettenr/asr-fon-without-diacritics"),
    "whettenr_bpe256":         ("speechbrain", "whettenr/asr-fon-with-diacritics-bpe-256"),
    # --- autres licences (informatif) ---
    "professor_mms_fongbe":    ("ctc", "Professor/mms-300m-fongbe"),
    "whisper_fongbe":          ("whisper", "misterkissi/whisper-small-fongbe"),
    "whettenr_streaming":      ("unsupported", "whettenr/asr-fon-streaming-conformer-without-diacritics (transducer)"),
    "whettenr_fr_streaming":   ("unsupported", "whettenr/asr-fon-french-streaming-conformer-without-diacritics (transducer)"),
    "billfass":                ("ctc", "billfass/fongbe_asr_model"),
    "finanfa_v6":              ("whisper", "finanfa-tech/finanfa-fon-asr-v6"),
    # --- sans licence ---
    "octaspace_w2vbert":       ("ctc", "OctaSpace/wav2vec2-bert-fongbe"),
    "afrihubert_fongbe":       ("speechbrain", "fortunedavis/afrihubert_fongbe_asr"),
    "junxwinx_v2":             ("ctc", "JunxWinx/fon-asr-fongbe-v2"),
    "toutestnwaar":            ("speechbrain", "toutestnwaar/asr_fongbe"),
    "ronaldodev_v4":           ("unsupported", "Ronaldodev/fon-asr-v4 (GGUF llama.cpp)"),
    # --- zéro-shot de référence ---
    "mms_1b_zeroshot":         ("ctc", "facebook/mms-1b-all"),
    "whisper_small_zeroshot":  ("whisper", "openai/whisper-small"),
    "xlsr53_zeroshot":         ("ctc", "facebook/wav2vec2-large-xlsr-53"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--models", default="all", help="liste csv de noms, ou 'all'")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default=None, help="CSV de résultats")
    return p.parse_args()


def get_eval_df(data_dir: Path, n: int, seed: int):
    df = pd.read_csv(data_dir / "Train_reliable.csv")
    df["wav"] = df.audio_path.map(lambda p: str(data_dir / p))
    val = df[(df.recommended_validation == 1) & df.wav.map(lambda p: Path(p).exists())]
    return val.sample(min(n, len(val)), random_state=seed).reset_index(drop=True)


def load_audio(p):
    import soundfile as sf
    audio, sr = sf.read(p, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def progress(i, n, tag):
    if (i + 1) % 10 == 0 or i == 0:
        print(f"    [{i+1}/{n}] {tag}", flush=True)


# ---------------- handlers ----------------

def make_ctc(model_id, device):
    """Wav2Vec2ForCTC / Wav2Vec2BertForCTC via classes Auto."""
    from transformers import (AutoConfig, AutoFeatureExtractor, AutoModelForCTC,
                              AutoTokenizer)
    cfg = AutoConfig.from_pretrained(model_id)
    feat = AutoFeatureExtractor.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
    blank = tok.pad_token_id if tok.pad_token_id is not None else 0

    @torch.no_grad()
    def transcribe(paths):
        out = []
        for i, p in enumerate(paths):
            inp = feat(load_audio(p), sampling_rate=16000,
                       return_tensors="pt").input_features.to(device)
            logits = model(inp).logits
            ids = logits.argmax(-1)[0].tolist()
            collapsed = [ids[0]] + [b for a, b in zip(ids, ids[1:]) if b != a]
            collapsed = [j for j in collapsed if j != blank]
            out.append(tok.decode(collapsed, skip_special_tokens=True).strip())
            progress(i, len(paths), model_id.split("/")[-1])
        return out
    return transcribe


def make_whisper(model_id, device):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def transcribe(paths):
        out = []
        for i, p in enumerate(paths):
            inp = proc(load_audio(p), sampling_rate=16000,
                       return_tensors="pt").input_features.to(device)
            gen = model.generate(inp, max_new_tokens=128)
            out.append(proc.batch_decode(gen, skip_special_tokens=True)[0].strip())
            progress(i, len(paths), model_id.split("/")[-1])
        return out
    return transcribe


def make_speechbrain(model_id):
    """Choisit EncoderASR (CTC-only) ou EncoderDecoderASR selon hyperparams.yaml."""
    from huggingface_hub import hf_hub_download
    from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR

    try:
        hp = Path(hf_hub_download(model_id, "hyperparams.yaml")).read_text()
    except Exception:
        hp = ""
    is_transducer = "transducer" in hp.lower()
    is_ctc_only = "ctc_greedy_decode" in hp and not re_search_decoder(hp)

    if is_transducer:
        raise RuntimeError("modèle transducer non supporté par ce script")

    cls = EncoderASR if is_ctc_only else EncoderDecoderASR
    print(f"    speechbrain: {cls.__name__}", flush=True)
    savedir = Path(tempfile.mkdtemp(prefix="sb_"))
    try:
        asr = cls.from_hparams(source=model_id, savedir=str(savedir),
                               run_opts={"device": "cpu"})
    except TypeError:
        # certaines versions n'acceptent pas run_opts en kwargs
        asr = cls.from_hparams(source=model_id, savedir=str(savedir))

    def transcribe(paths):
        out = []
        for i, p in enumerate(paths):
            out.append(asr.transcribe_file(str(p)).strip())
            progress(i, len(paths), model_id.split("/")[-1])
        return out
    return transcribe


def re_search_decoder(hp: str) -> bool:
    import re
    # 'decoder:' dans la section modules: → modèle seq2seq
    m = re.search(r"modules:\s*\n((?:\s+\w+:.*\n)+)", hp)
    return bool(m and re.search(r"^\s*decoder:", m.group(1), re.M))


def score_predictions(data_dir: Path, df: pd.DataFrame, preds: list[str], tag: str):
    """Écrit un CSV temporaire et appelle score.py officiel."""
    sub = pd.DataFrame({"id": df["id"], "transcription": preds})
    ref = df[["id", "transcription"]]
    with tempfile.TemporaryDirectory() as td:
        rp, sp = Path(td) / "ref.csv", Path(td) / "sub.csv"
        ref.to_csv(rp, index=False)
        sub.to_csv(sp, index=False)
        r = subprocess.run([sys.executable, str(data_dir / "score.py"),
                            "--reference", str(rp), "--submission", str(sp)],
                           capture_output=True, text=True)
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    print(f"  → {tag}: {line}", flush=True)
    import re
    m = re.search(r"score (\d+\.\d+)", r.stdout)
    return float(m.group(1)) if m else float("nan")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = get_eval_df(data_dir, args.n, args.seed)
    print(f"évaluation sur {len(df)} utterances de validation", flush=True)

    names = list(MODELS) if args.models == "all" else args.models.split(",")
    results = []
    for name in names:
        if name not in MODELS:
            print(f"⚠️  {name} inconnu, skip")
            continue
        mtype, model_id = MODELS[name]
        t0 = time.time()
        print(f"\n=== {name} ({model_id}) [{mtype}] ===", flush=True)
        try:
            if mtype == "unsupported":
                raise RuntimeError("non supporté (format/licence)")
            if mtype == "ctc":
                fn = make_ctc(model_id, device)
            elif mtype == "whisper":
                fn = make_whisper(model_id, device)
            elif mtype == "speechbrain":
                fn = make_speechbrain(model_id)
            else:  # auto : ctc puis speechbrain
                try:
                    fn = make_ctc(model_id, device)
                except Exception as e:
                    print(f"  ctc a échoué ({type(e).__name__}), essai speechbrain", flush=True)
                    fn = make_speechbrain(model_id)
            preds = fn(df.wav.tolist())
            score = score_predictions(data_dir, df, preds, name)
            results.append({"model": name, "hf_id": model_id, "type": mtype,
                            "score": score, "minutes": round((time.time()-t0)/60, 1)})
        except Exception as e:
            print(f"  ❌ {name} a échoué : {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=3)
            results.append({"model": name, "hf_id": model_id, "type": mtype,
                            "score": float("nan"), "minutes": round((time.time()-t0)/60, 1)})

    res = pd.DataFrame(results).sort_values("score")
    print("\n===== CLASSEMENT =====")
    print(res.to_string(index=False))
    out = args.out or str(data_dir / "eval_models_results.csv")
    res.to_csv(out, index=False)
    print(f"résultats → {out}")


if __name__ == "__main__":
    main()
