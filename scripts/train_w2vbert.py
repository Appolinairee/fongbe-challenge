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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    p.add_argument("--hf-repo", default=None,
                   help="Repo Hugging Face (ex: user/fongbe-ckpt) : 2e destination de sync, "
                        "token via variable d'env HF_TOKEN")
    p.add_argument("--sync-remote", default=None,
                   help="Remote rclone (ex: fonvoice-drive:dossier) : checkpoints écrits "
                        "PAR le compte propriétaire → quota propriétaire, pas de FUSE")
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
    p.add_argument("--include-religious", action="store_true",
                   help="Ajoute Train_religious.csv au train (pas de split val : tout en train)")
    p.add_argument("--include-uncertain", action="store_true",
                   help="Ajoute TOUTES les lignes de Train_uncertain.csv au train")
    p.add_argument("--uncertain-weight", type=float, default=1.0,
                   help="Poids d'échantillonnage des lignes uncertain (défaut 1.0)")
    p.add_argument("--mask-time-prob", type=float, default=0.0)
    p.add_argument("--mask-feature-prob", type=float, default=0.0)
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


def prune_checkpoints(ckpt_dir: Path, keep: int, protect=None):
    cands = sorted(ckpt_dir.glob("step_*.pt"), key=_step_of)
    for old in cands[:-keep] if keep > 0 else []:
        if protect and protect(old):
            continue  # encore en file de sync — ne pas supprimer la source
        old.unlink(missing_ok=True)
    # Nettoie les .tmp orphelins de copies interrompues (>60 min)
    cutoff = time.time() - 3600
    for tmp in ckpt_dir.glob("*.tmp"):
        try:
            if tmp.stat().st_mtime < cutoff:
                tmp.unlink(missing_ok=True)
        except OSError:
            pass


class Syncer:
    """Copie en arrière-plan des checkpoints/artefacts vers le dossier partagé.

    File SÉQUENTIELLE : une seule copie à la fois — les checkpoints font ~7 GB
    et les copies parallèles saturaient le mount FUSE (d'où les .tmp à 0 octet).
    """

    def __init__(self, sync_dir: Path):
        import queue
        self.sync_dir = sync_dir
        self._q = queue.Queue()
        self._pending = set()  # src encore en file — ne pas élager localement
        self.failed = False  # True si un artefact critique (.pt) n'a pas pu être synchronisé
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, src: Path, rel: str):
        self._pending.add(str(src))
        self._q.put((src, rel))

    def is_pending(self, path: Path) -> bool:
        return str(path) in self._pending

    def _run(self):
        while True:
            src, rel = self._q.get()
            self._pending.discard(str(src))
            dst = self.sync_dir / rel
            ok = False
            for attempt in range(3):
                try:
                    atomic_copy(src, dst)
                    # fsync best-effort : force le flush des données vers Drive
                    try:
                        fd = os.open(dst, os.O_RDONLY)
                        os.fsync(fd)
                        os.close(fd)
                    except OSError:
                        pass
                    # Vérifie que le fichier distant a la bonne taille
                    # (le mount FUSE peut buffériser : copy2 peut retourner avant flush)
                    if dst.exists() and dst.stat().st_size == src.stat().st_size:
                        ok = True
                        break
                    print(f"⚠️  sync {rel} : taille mismatch, retry {attempt+1}/3", flush=True)
                except Exception as e:
                    print(f"⚠️  sync {rel} échoué ({e}), retry {attempt+1}/3", flush=True)
                time.sleep(5)
            if ok:
                self.failed = False
                print(f"☁️  sync → {rel}", flush=True)
            else:
                print(f"❌ sync {rel} PERDU après 3 essais", flush=True)
                if rel.endswith(".pt"):
                    self.failed = True

    def wait(self, timeout=600):
        self._q.join()


class HfSyncer:
    """Upload séquentiel vers un repo Hugging Face — alternative robuste à Drive.

    Même file unique que Syncer : les gros checkpoints ne se marchent pas dessus.
    Token via la variable d'environnement HF_TOKEN (huggingface_hub la lit seule).
    """

    def __init__(self, repo_id: str):
        import queue
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.api = HfApi()
        self._q = queue.Queue()
        self._pending = set()
        self.failed = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, src: Path, rel: str):
        self._pending.add(str(src))
        self._q.put((src, rel))

    def is_pending(self, path: Path) -> bool:
        return str(path) in self._pending

    def _remote_size(self, rel: str):
        from huggingface_hub import get_hf_file_metadata
        url = f"https://huggingface.co/{self.repo_id}/resolve/main/{rel}"
        return get_hf_file_metadata(url).size

    def _run(self):
        while True:
            src, rel = self._q.get()
            self._pending.discard(str(src))
            ok = False
            for attempt in range(3):
                try:
                    self.api.upload_file(path_or_fileobj=str(src),
                                         path_in_repo=rel, repo_id=self.repo_id)
                    if self._remote_size(rel) == src.stat().st_size:
                        ok = True
                        break
                    print(f"⚠️  hf sync {rel} : taille mismatch, retry {attempt+1}/3", flush=True)
                except Exception as e:
                    print(f"⚠️  hf sync {rel} échoué ({e}), retry {attempt+1}/3", flush=True)
                time.sleep(5)
            if ok:
                self.failed = False
                print(f"🤗 sync → {self.repo_id}/{rel}", flush=True)
            else:
                print(f"❌ hf sync {rel} PERDU après 3 essais", flush=True)
                if rel.endswith(".pt"):
                    self.failed = True

    def wait(self, timeout=600):
        self._q.join()


def _rclone(*args, check=True):
    return subprocess.run(["rclone", *args], capture_output=True, text=True, check=check)


class RcloneSyncer:
    """Sync séquentielle via rclone vers un remote du COMPTE PROPRIÉTAIRE.

    - Les fichiers sont possédés par le compte du remote → quota du propriétaire.
    - Upload HTTP résumable : pas de perte liée au cache FUSE.
    - rclone doit être installé et configuré (RCLONE_CONFIG_* ou rclone.conf).
    """

    def __init__(self, remote: str):
        import queue
        self.remote = remote.rstrip("/")
        self._q = queue.Queue()
        self._pending = set()
        self.failed = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, src: Path, rel: str):
        self._pending.add(str(src))
        self._q.put((src, rel))

    def is_pending(self, path: Path) -> bool:
        return str(path) in self._pending

    def _remote_size(self, rel: str):
        r = _rclone("lsjson", f"{self.remote}/{rel}", check=False)
        if r.returncode != 0 or not r.stdout.strip():
            return -1
        try:
            return int(json.loads(r.stdout)[0]["Size"])
        except Exception:
            return -1

    def _run(self):
        while True:
            src, rel = self._q.get()
            self._pending.discard(str(src))
            dst = f"{self.remote}/{rel}"
            ok = False
            for attempt in range(3):
                try:
                    r = _rclone("copyto", str(src), dst, "--stats-one-line", "--retries", "3",
                                check=False)
                    if r.returncode == 0 and self._remote_size(rel) == src.stat().st_size:
                        ok = True
                        break
                    print(f"⚠️  rclone sync {rel} : rc={r.returncode} ou taille mismatch, "
                          f"retry {attempt+1}/3", flush=True)
                except Exception as e:
                    print(f"⚠️  rclone sync {rel} échoué ({e}), retry {attempt+1}/3", flush=True)
                time.sleep(5)
            if ok:
                self.failed = False
                print(f"🪣 sync → {dst}", flush=True)
            else:
                print(f"❌ rclone sync {rel} PERDU après 3 essais", flush=True)
                if rel.endswith(".pt"):
                    self.failed = True

    def wait(self, timeout=600):
        self._q.join()

    def prune(self, keep: int):
        """Garde les `keep` plus grands step_*.pt distants, supprime le reste."""
        try:
            r = _rclone("lsf", f"{self.remote}/checkpoints/", "--files-only", check=False)
            names = sorted((n for n in r.stdout.split() if n.startswith("step_") and n.endswith(".pt")),
                           key=_step_of)
            for old in names[:-keep] if keep > 0 else []:
                _rclone("deletefile", f"{self.remote}/checkpoints/{old}", check=False)
        except Exception as e:
            print(f"⚠️  prune remote échoué ({e})", flush=True)


def latest_rclone_checkpoint(remote: str, dest_dir: Path):
    """Télécharge le plus grand step_*.pt du remote rclone. Retourne le path local."""
    try:
        r = _rclone("lsf", f"{remote}/checkpoints/", "--files-only", check=False)
        names = [n for n in r.stdout.split() if n.startswith("step_") and n.endswith(".pt")]
        if not names:
            return None
        best = max(names, key=_step_of)
        dest = dest_dir / best
        _rclone("copyto", f"{remote}/checkpoints/{best}", str(dest), check=True)
        print(f"⬇️  checkpoint {remote} {best} (step {_step_of(Path(best))}) rapatrié", flush=True)
        return dest
    except Exception as e:
        print(f"⚠️  lookup rclone échoué ({type(e).__name__}: {e})", flush=True)
        return None


def latest_hf_checkpoint(repo_id: str, dest_dir: Path):
    """Télécharge le plus gros step_*.pt du repo HF dans dest_dir. Retourne le path local."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        best_step, best_path = -1, None
        for f in api.list_repo_tree(repo_id, repo_type="model", recursive=True):
            p = getattr(f, "path", "") or ""
            if "/step_" in p and p.endswith(".pt") and _step_of(Path(p)) > best_step:
                best_step, best_path = _step_of(Path(p)), p
        if best_path is None:
            return None
        local = hf_hub_download(repo_id, best_path, repo_type="model",
                                local_dir=str(dest_dir))
        print(f"⬇️  checkpoint HF {best_path} (step {best_step}) rapatrié", flush=True)
        return Path(local)
    except Exception as e:
        print(f"⚠️  lookup HF échoué ({type(e).__name__}: {e})", flush=True)
        return None


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    local_ckpt = work_dir / "checkpoints"
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Syncers (Drive +/ou Hugging Face) ----------
    syncers = []
    sync_dir = Path(args.sync_dir) if args.sync_dir else None
    sync_ckpt = sync_dir / "checkpoints" if sync_dir else None
    if sync_dir:
        syncers.append(Syncer(sync_dir))
    if args.hf_repo:
        try:
            syncers.append(HfSyncer(args.hf_repo))
            print(f"sync HF → {args.hf_repo}", flush=True)
        except Exception as e:
            print(f"⚠️  HF sync indisponible ({e}) — Drive seul", flush=True)
    if args.sync_remote:
        try:
            _rclone("listremotes", check=True)
            syncers.append(RcloneSyncer(args.sync_remote))
            print(f"sync rclone → {args.sync_remote} (compte propriétaire)", flush=True)
        except Exception as e:
            print(f"⚠️  rclone indisponible ({e}) — ignoré", flush=True)
    syncer = syncers[0] if syncers else None  # compat affichage

    if sync_dir:
        # Le vrai dossier partagé contient data/ et audio/ à côté d'outputs/
        shared_root = sync_dir.parent.parent
        markers = [p.name for p in shared_root.iterdir()] if shared_root.exists() else []
        if "data" not in markers:
            print(f"⚠️  ATTENTION : {shared_root} ne ressemble pas au dossier partagé "
                  f"(contenu: {markers}). Checkpoints possiblement perdus — "
                  f"vérifier que le compte Colab a bien le raccourci Drive.", flush=True)

    if not torch.cuda.is_available():
        sys.exit("⛔ CUDA indisponible — ce runtime n'a PAS de GPU "
                 "(quota épuisé ?). Change de runtime (Runtime → Change runtime "
                 "type → GPU) ou de compte, puis relance. Entraîner sur CPU "
                 "tuerait la VM (exit -9).")
    device = "cuda"
    print(f"device: {device} {torch.cuda.get_device_name(0)}", flush=True)

    sys.path.insert(0, str(data_dir))
    from normalize import tonal_target  # noqa: E402  (script fourni par l'orga)

    # ---------- Données ----------
    reliable = pd.read_csv(data_dir / "Train_reliable.csv")
    if args.include_religious:
        rel_path = data_dir / "Train_religious.csv"
        if rel_path.exists():
            religious = pd.read_csv(rel_path)
            religious["recommended_validation"] = 0
            reliable = pd.concat([reliable, religious], ignore_index=True)
            print(f"+ Train_religious: {len(religious):,} utts "
                  f"({religious.duration.sum()/3600:.1f} h)", flush=True)
    if args.include_uncertain:
        unc_path = data_dir / "Train_uncertain.csv"
        if unc_path.exists():
            unc = pd.read_csv(unc_path)
            unc["recommended_validation"] = 0
            unc["_w"] = args.uncertain_weight
            reliable = pd.concat([reliable.assign(_w=1.0), unc], ignore_index=True)
            print(f"+ Train_uncertain: {len(unc):,} utts "
                  f"({unc.duration.sum()/3600:.1f} h, poids {args.uncertain_weight})",
                  flush=True)
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
        mask_time_prob=args.mask_time_prob, mask_feature_prob=args.mask_feature_prob,
        apply_spec_augment=(args.mask_time_prob > 0 or args.mask_feature_prob > 0),
        layerdrop=0.0, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        add_adapter=True, pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer)).to(device)
    model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.0f} M parameters", flush=True)

    train_set = None  # SpeechSet reconstruit par epoch (échantillonnage pondéré)
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        print("optimiseur: AdamW8bit (bitsandbytes, ~4 GB GPU libérés)", flush=True)
    except Exception as e:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        print(f"optimiseur: AdamW fp32 (bitsandbytes indisponible: {e})", flush=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    # ---------- Reprise (local d'abord, puis sync distant) ----------
    step, start_time = 0, time.time()
    if not args.no_resume:
        ckpt = latest_checkpoint(local_ckpt, sync_ckpt)
        if args.hf_repo:
            hf_ckpt = latest_hf_checkpoint(args.hf_repo, work_dir)
            if hf_ckpt and (ckpt is None or _step_of(hf_ckpt) > _step_of(ckpt)):
                ckpt = hf_ckpt
        if args.sync_remote:
            rc_ckpt = latest_rclone_checkpoint(args.sync_remote, local_ckpt)
            if rc_ckpt and (ckpt is None or _step_of(rc_ckpt) > _step_of(ckpt)):
                ckpt = rc_ckpt
        if ckpt is not None:
            if ckpt.parent == sync_ckpt:
                # checkpoint distant → rapatriement local avant chargement
                print(f"⬇️  rapatriement {ckpt.name} depuis le sync-dir...", flush=True)
                local_ckpt.mkdir(parents=True, exist_ok=True)
                atomic_copy(ckpt, local_ckpt / ckpt.name)
                ckpt = local_ckpt / ckpt.name
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception as e:
                print(f"⚠️ état optimiseur non restauré ({e}) — optimiseur frais", flush=True)
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            try:
                random.setstate(state["rng"]["python"])
                np.random.set_state(state["rng"]["numpy"])
                t = state["rng"]["torch"]
                if not torch.is_tensor(t):
                    t = torch.tensor(t, dtype=torch.uint8)
                torch.set_rng_state(t)
                if device == "cuda" and state["rng"].get("cuda") is not None:
                    c = state["rng"]["cuda"]
                    torch.cuda.set_rng_state(c if torch.is_tensor(c)
                                             else torch.tensor(c, dtype=torch.uint8))
            except Exception as e:
                print(f"⚠️ état RNG non restauré ({e}) — reprise continue", flush=True)
            step = state["step"]
            del state
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"↻ reprise depuis {ckpt.name} (step {step})", flush=True)

    # ---------- Boucle d'entraînement ----------
    print(f"=== TRAINING {step} → {args.max_steps} steps "
          f"(checkpoint /{args.checkpoint_every}, "
          f"sync Drive: {'oui' if syncer else 'non'}) ===", flush=True)
    model.train()
    micro, running = 0, []
    all_failed = lambda: bool(syncers) and all(s.failed for s in syncers)
    any_pending = lambda p: any(s.is_pending(p) for s in syncers)
    total_q = lambda: sum(s._q.qsize() for s in syncers)

    while step < args.max_steps and not all_failed():
        epoch_df = train_df
        if "_w" in train_df.columns:
            epoch_df = train_df.sample(frac=1.0, replace=True,
                                       weights="_w", random_state=step)
        loader = DataLoader(SpeechSet(epoch_df),
                            batch_sampler=length_batches(epoch_df.duration.values,
                                                         args.batch_size, True),
                            collate_fn=collate, num_workers=2)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=device == "cuda"):
                    loss = model(**batch).loss / args.grad_accum
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                print(f"⚠️  OOM — batch ignoré (step ~{step}), cache vidé", flush=True)
                del batch
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
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
            if all_failed():
                print("⛔ sync en échec sur TOUTES les destinations — arrêt propre "
                      "(checkpoints locaux conservés, relançable)", flush=True)
                break
            if step % 20 == 0:
                elapsed = time.time() - start_time
                sec_per_step = elapsed / max(step, 1)
                eta_min = (args.max_steps - step) * sec_per_step / 60
                gpu = (torch.cuda.memory_allocated() / 1e9
                       if device == "cuda" else 0.0)
                print(f"step {step:5d}/{args.max_steps}  loss {np.mean(running):.3f}  "
                      f"GPU {gpu:4.1f} Go  {elapsed/60:5.1f} min  ETA {eta_min:5.1f} min",
                      flush=True)
                running = []
            if step % args.checkpoint_every == 0:
                try:
                    save_checkpoint(local_ckpt / f"step_{step}.pt", step, model, optimizer,
                                    scheduler, scaler,
                                    {"python": random.getstate(), "numpy": np.random.get_state(),
                                     "torch": torch.get_rng_state(),
                                     "cuda": torch.cuda.get_rng_state() if device == "cuda" else None})
                    prune_checkpoints(local_ckpt, args.keep_checkpoints,
                                      protect=any_pending if syncers else None)
                    if sync_ckpt:
                        try:
                            prune_checkpoints(sync_ckpt, args.keep_checkpoints)
                        except Exception:
                            pass
                    for s in syncers:
                        if isinstance(s, RcloneSyncer):
                            s.prune(args.keep_checkpoints)
                    for s in syncers:
                        s.submit(local_ckpt / f"step_{step}.pt",
                                 f"checkpoints/step_{step}.pt")
                        s.submit(work_dir / "vocab.json", "vocab.json")
                    print(f"💾 checkpoint step_{step} écrit en local"
                          + (f" — sync en cours (file: {total_q()})" if syncers else ""), flush=True)
                except Exception as e:
                    print(f"⚠️ checkpoint step_{step} échoué ({e}) — training continue", flush=True)
            if step >= args.max_steps:
                break

    # ---------- Évaluation ----------
    print(f"=== EVALUATION sur {len(valid_df)} utterances tenues ===", flush=True)

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

    if syncers:
        # Copie des artefacts finaux vers les destinations de sync
        for name in ["valid_reference.csv", "valid_prediction.csv", "score_output.txt"]:
            f = work_dir / name
            if f.exists():
                for s in syncers:
                    s.submit(f, name)
        # Le dossier model/ peut être volumineux : on le copie fichier par fichier
        for f in model_dir.iterdir():
            if f.is_file():
                for s in syncers:
                    s.submit(f, f"model/{f.name}")
        for s in syncers:
            s.wait(timeout=1800)
            if isinstance(s, RcloneSyncer):
                s.prune(args.keep_checkpoints)
        if sync_ckpt:
            try:
                prune_checkpoints(sync_ckpt, args.keep_checkpoints)
            except Exception:
                pass
        print(f"☁️  artefacts synchronisés ({len(syncers)} destination(s))", flush=True)


if __name__ == "__main__":
    main()
