#!/usr/bin/env python3
"""Fine-tuning CTC de facebook/w2v-bert-2.0 pour l'ASR Fongbe (challenge Zindi).

Usage :
    python train_w2vbert.py --data-dir /content/fongbe \
        --work-dir /content/fongbe_run \
        --sync-dir /content/drive/MyDrive/fongbe-challenge/outputs/w2vbert_shared \
        --checkpoint-every 500

Architecture robuste :
- Les checkpoints sont écrits sur le **disque local** (--work-dir/checkpoints/),
  jamais directement sur Drive : pas de blocage FUSE pendant le training.
- Chaque checkpoint est ensuite **copié en arrière-plan** vers --sync-dir
  (Drive partagé) : un autre compte peut reprendre là où ce run s'est arrêté.
- À la reprise, on prend le plus grand step parmi les checkpoints locaux ET
  ceux du sync-dir (téléchargés en local si nécessaire).
- Toutes les écritures de checkpoints sont atomiques (tmp + os.replace).
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    SeamlessM4TFeatureExtractor,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    get_linear_schedule_with_warmup,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Dossier contenant CSV + audio/")
    p.add_argument("--work-dir", required=True,
                   help="Dossier de travail LOCAL (checkpoints + modèle final)")
    p.add_argument("--sync-dir", default=None,
                   help="Dossier distant PARTAGÉ (Drive) : checkpoints + artefacts y sont copiés")
    p.add_argument("--base-model", default="facebook/w2v-bert-2.0")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--max-seconds", type=float, default=15.0)
    p.add_argument("--n-validation", type=int, default=300)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def _step_of(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        return -1


def latest_checkpoint(*dirs: Path):
    """Retourne (step, path) du plus grand checkpoint parmi les dossiers donnés."""
    best = (-1, None)
    for d in dirs:
        if d and d.exists():
            for p in d.glob("step_*.pt"):
                if _step_of(p) > best[0]:
                    best = (_step_of(p), p)
    return best[1]


def atomic_copy(src: Path, dst: Path):
    """Copie atomique : dst_tmp puis rename (évite les fichiers partiels visibles)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def save_checkpoint(path: Path, step, model, optimizer, scheduler, scaler, rng_states):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_states,
    }, tmp)
    os.replace(tmp, path)


def prune_checkpoints(ckpt_dir: Path, keep: int):
    cands = sorted(ckpt_dir.glob("step_*.pt"), key=_step_of)
    for old in cands[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)


class Syncer:
    """Copie en arrière-plan des checkpoints/artefacts vers le dossier partagé."""

    def __init__(self, sync_dir: Path):
        self.sync_dir = sync_dir
        self._threads = []

    def submit(self, src: Path, rel: str):
        dst = self.sync_dir / rel

        def work():
            try:
                atomic_copy(src, dst)
                print(f"☁️  sync → {rel}", flush=True)
            except Exception as e:
                print(f"⚠️  sync {rel} échoué ({e})", flush=True)

        t = threading.Thread(target=work, daemon=True)
        self._threads.append(t)
        t.start()

    def wait(self, timeout=600):
        for t in self._threads:
            t.join(timeout=max(1, timeout // max(1, len(self._threads))))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    local_ckpt = work_dir / "checkpoints"
    sync_dir = Path(args.sync_dir) if args.sync_dir else None
    sync_ckpt = sync_dir / "checkpoints" if sync_dir else None
    work_dir.mkdir(parents=True, exist_ok=True)

    syncer = Syncer(sync_dir) if sync_dir else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}", flush=True)

    sys.path.insert(0, str(data_dir))
    from normalize import tonal_target  # noqa: E402  (script fourni par l'orga)

    # ---------- Données ----------
    reliable = pd.read_csv(data_dir / "Train_reliable.csv")
    reliable["target"] = reliable.transcription.map(tonal_target).str.strip()
    reliable["wav"] = reliable.audio_path.map(lambda p: str(data_dir / p))
    available = reliable[reliable.wav.map(lambda p: Path(p).exists()) & (reliable.target != "")]

    train_df = available[(available.recommended_validation == 0)
                         & (available.duration <= args.max_seconds)]
    valid_df = available[available.recommended_validation == 1]
    valid_df = valid_df.sample(min(args.n_validation, len(valid_df)), random_state=args.seed)
    print(f"train: {len(train_df):,} utts, {train_df.duration.sum()/3600:.1f} h, "
          f"{train_df.speaker_id.nunique()} voices", flush=True)
    print(f"valid: {len(valid_df):,} utts, {valid_df.speaker_id.nunique()} voices (unseen)", flush=True)

    # ---------- Tokenizer (déterministe) ----------
    chars = sorted(set("".join(train_df.target)) - {" "})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    vocab_path = work_dir / "vocab.json"
    if not vocab_path.exists():
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False))
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), unk_token="[UNK]", pad_token="[PAD]",
                                     word_delimiter_token="|", bos_token=None, eos_token=None)

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.base_model)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def load_audio(path):
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        assert sr == 16000, path
        return audio if audio.ndim == 1 else audio.mean(axis=1)

    class SpeechSet(Dataset):
        def __init__(self, df, with_labels=True):
            self.paths = df.wav.tolist()
            self.texts = df.target.tolist() if with_labels else None

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            item = {"input_features": feature_extractor(
                load_audio(self.paths[i]), sampling_rate=16000).input_features[0]}
            if self.texts is not None:
                item["labels"] = tokenizer(self.texts[i]).input_ids
            return item

    def collate(items):
        batch = processor.pad([{"input_features": x["input_features"]} for x in items],
                              padding=True, return_tensors="pt")
        if "labels" in items[0]:
            labels = processor.pad(labels=[{"input_ids": x["labels"]} for x in items],
                                   padding=True, return_tensors="pt")
            batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        return batch

    def length_batches(durations, batch_size, shuffle):
        order = np.argsort(durations)
        batches = [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]
        if shuffle:
            random.shuffle(batches)
        return batches

    # ---------- Modèle ----------
    model = Wav2Vec2BertForCTC.from_pretrained(
        args.base_model, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0,
        mask_time_prob=0.0, layerdrop=0.0, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        add_adapter=True, pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer)).to(device)
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.0f} M parameters", flush=True)

    train_set = SpeechSet(train_df)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    # ---------- Reprise (local d'abord, puis sync distant) ----------
    step, start_time = 0, time.time()
    if not args.no_resume:
        ckpt = latest_checkpoint(local_ckpt, sync_ckpt)
        if ckpt is not None:
            if ckpt.parent == sync_ckpt:
                # checkpoint distant → rapatriement local avant chargement
                print(f"⬇️  rapatriement {ckpt.name} depuis le sync-dir...", flush=True)
                local_ckpt.mkdir(parents=True, exist_ok=True)
                atomic_copy(ckpt, local_ckpt / ckpt.name)
                ckpt = local_ckpt / ckpt.name
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            random.setstate(state["rng"]["python"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"])
            if device == "cuda" and state["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state(state["rng"]["cuda"])
            step = state["step"]
            print(f"↻ reprise depuis {ckpt.name} (step {step})", flush=True)

    # ---------- Boucle d'entraînement ----------
    model.train()
    micro, running = 0, []
    while step < args.max_steps:
        loader = DataLoader(train_set,
                            batch_sampler=length_batches(train_df.duration.values,
                                                         args.batch_size, True),
                            collate_fn=collate, num_workers=2)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running.append(loss.item() * args.grad_accum)
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step:5d}/{args.max_steps}  loss {np.mean(running):.3f}  "
                      f"{(time.time()-start_time)/60:5.1f} min", flush=True)
                running = []
            if step % args.checkpoint_every == 0:
                try:
                    save_checkpoint(local_ckpt / f"step_{step}.pt", step, model, optimizer,
                                    scheduler, scaler,
                                    {"python": random.getstate(), "numpy": np.random.get_state(),
                                     "torch": torch.get_rng_state(),
                                     "cuda": torch.cuda.get_rng_state() if device == "cuda" else None})
                    prune_checkpoints(local_ckpt, args.keep_checkpoints)
                    if syncer:
                        syncer.submit(local_ckpt / f"step_{step}.pt",
                                      f"checkpoints/step_{step}.pt")
                        syncer.submit(work_dir / "vocab.json", "vocab.json")
                    print(f"💾 checkpoint step_{step} (local" +
                          (" + sync Drive" if syncer else "") + ")", flush=True)
                except Exception as e:
                    print(f"⚠️ checkpoint step_{step} échoué ({e}) — training continue", flush=True)
            if step >= args.max_steps:
                break

    # ---------- Évaluation ----------
    @torch.no_grad()
    def transcribe(df, batch_size=8):
        model.eval()
        data = SpeechSet(df, with_labels=False)
        batches = length_batches(df.duration.values, batch_size, False)
        out = [""] * len(df)
        for idx in batches:
            batch = collate([data[i] for i in idx])
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                logits = model(input_features=batch["input_features"].to(device),
                               attention_mask=batch["attention_mask"].to(device)).logits
            for i, text in zip(idx,
                               processor.batch_decode(logits.argmax(-1).cpu(),
                                                      skip_special_tokens=True)):
                out[i] = text
        return out

    valid_df = valid_df.copy()
    valid_df["prediction"] = transcribe(valid_df)
    valid_df[["id", "transcription"]].to_csv(work_dir / "valid_reference.csv", index=False)
    valid_df[["id", "prediction"]].rename(
        columns={"prediction": "transcription"}).to_csv(work_dir / "valid_prediction.csv",
                                                      index=False)

    score_proc = subprocess.run(
        [sys.executable, str(data_dir / "score.py"),
         "--reference", str(work_dir / "valid_reference.csv"),
         "--submission", str(work_dir / "valid_prediction.csv")],
        capture_output=True, text=True)
    print(score_proc.stdout, flush=True)
    (work_dir / "score_output.txt").write_text(score_proc.stdout + score_proc.stderr)

    for _, r in valid_df.head(5).iterrows():
        print("REF:", r.transcription, "\nHYP:", r.prediction, "\n")

    # ---------- Sauvegarde finale ----------
    model_dir = work_dir / "model"
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    print(f"✅ terminé en {(time.time()-start_time)/60:.0f} min — modèle dans {model_dir}",
          flush=True)

    if syncer:
        # Copie des artefacts finaux vers le dossier partagé
        for name in ["valid_reference.csv", "valid_prediction.csv", "score_output.txt"]:
            f = work_dir / name
            if f.exists():
                syncer.submit(f, name)
        # Le dossier model/ peut être volumineux : on le copie fichier par fichier
        for f in model_dir.iterdir():
            if f.is_file():
                syncer.submit(f, f"model/{f.name}")
        syncer.wait(timeout=1800)
        if sync_ckpt:
            try:
                prune_checkpoints(sync_ckpt, args.keep_checkpoints)
            except Exception:
                pass
        print(f"☁️  artefacts synchronisés vers {sync_dir}", flush=True)


if __name__ == "__main__":
    main()
