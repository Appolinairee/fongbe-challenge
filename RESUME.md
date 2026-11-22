# Fongbe Challenge — Résumé de toutes les informations

## 1. Le challenge

- **Objectif** : construire un ASR (reconnaissance vocale) pour le **fongbe**, langue tonale du sud du Bénin (~2,3 M locuteurs).
- **Sortie attendue** : transcription orthographiée **avec toutes les marques de tons** (accent aigu = haut, grave = bas, caron = montant ; chiffres 1-5 après `syllabify()`).
- **Pourquoi** : les ASR actuels ratent les tons → transcriptions plausibles mais fausses. Modèle **frugal** : doit tourner sur CPU modeste.
- **Organisation** : Ministère de la Transformation Digitale et de l'Innovation (MTDI) du Bénin + ASIN + IIDiA, projet "J'aime Ma Langue", financé par la Gates Foundation. Plateforme : Zindi.
- **Dates** : clôture ~13 déc. 2026, reveal ~14 déc. 2026.
- **Prix** : 1er 2 500 $, 2e 1 500 $, 3e 1 000 $. Points Zindi.
- **Audio** : mono, 16 kHz, 16 bits WAV. ~370 h au total.

## 2. Les données (dossier `data/`)

| Fichier | Contenu |
|---|---|
| `Train_reliable.csv` | ~106 h, transcriptions validées par linguistes. Colonne `recommended_validation` = voix réservées pour la validation locale (inédites au test) |
| `Train_uncertain.csv` | ~198 h, tons de qualité incertaine ("as is"). Colonnes `protocol` (p1/p2/p3) |
| `Train_religious.csv` | ~42 h, lectures bibliques d'**un seul lecteur**, livres entiers tenus hors test |
| `Test.csv` | 9 407 utterances (~22 h), audio seul |
| `SampleSubmission.csv` | format : `ID`, `Target` |
| `audio_*.tar` + `SHA256SUMS` | archives ~2 GB, téléchargement : `https://storage.googleapis.com/fongbe/` |

**Colonnes CSV** : `id`, `audio_path`, `transcription`, `speaker_id` (anonyme), `duration`, + métadonnées (`recommended_validation`, `protocol`, `book`, `chapter`).

**Test en 2 parties** :
- *Général* : 6 382 utterances, 60 voix absentes de l'entraînement.
- *Religieux* : 3 025 utterances, même lecteur que `Train_religious`, livres bibliques tenus dehors (2 Chroniques, Proverbes, Sophonie, Aggée, 2 Rois, 1 Chroniques).
- Chaque partie est split public/privé avec des voix et livres différents.

## 3. Format d'écriture (conventions)

- Tons sur **chaque mot** : `á` haut, `à` bas, `ǎ` montant.
- Alphabet fon : `ɖ`, `ɛ`, `ɔ`… conservés.
- Emprunts français/anglais : orthographe d'origine, **sans tons**.
- Contractions (`ɖ'é`) et chiffres présents dans les données "as is".

## 4. Métrique et scripts fournis

- **Score** : WER calculé sur **syllabes** (pas mots). Chaque erreur de ton = 1 syllabe. Plus bas = mieux, 0 = parfait.
- **`syllabify.py`** : **obligatoire** avant soumission. Normalise (NFD, minuscules, retire ponctuation/chiffres/apostrophes), découpe en syllabes (onset + voyelle + "n" final), écrit le ton en chiffre (1 haut, 2 bas, 3 montant, 4 descendant, 5 milieu).
  - Ex : `"Un nɔ́ ɖu nǔ"` → `un nɔ1 ɖu nu3`
  - Usage : `python syllabify.py my_transcriptions.csv submission.csv`
- **`normalize.py`** : normalisation commune références + soumissions.
- **`score.py`** : évaluation locale + diagnostics tonals (TER conditionnel, TER par ton).
- **`test_score.py`** : `python test_score.py` doit afficher "11/11 tests passed".
- **`measure_frugality.py`** : mesure taille / RTF / RAM / offline (utilisé pour la vérification finale).
- **`starter_notebook.ipynb`** : baseline w2v-BERT 2.0, fine-tuning sur ~18 h de `Train_reliable`, Kaggle T4 gratuit en 2 h 32 → ~21 % syllable WER (20,6 % voix tenues, 21,8 % public, 21,3 % privé). **Point de départ, pas la solution gagnante ; pas frugal tel quel.**

## 5. Contraintes d'inférence (éliminatoires pour les prix)

- **Taille** : ≤ 1 000 MB tous poids confondus (pré-entraînés + LM), après quantification.
- **Vitesse** : RTF ≤ 1 sur **un seul thread CPU** sur tout le test.
- **Mémoire** : ≤ 2 GB RAM pic.
- **Environnement** : offline, pas de GPU, pas d'appel réseau/API.
- **Entraînement** : non contraint (GPU autorisés, mais **pas de GPU fourni** → Colab/Kaggle/locaux).
- **Vérification** : top teams → code, poids, Docker ou script reproductible, liste des ressources externes + licences, encodeur isolé (pour analyse de tons dans le paper de synthèse). Machine de référence : x86-64 récent, 1 thread, réseau coupé. `measure_frugality.py` rejoue tout : au-dessus d'une limite ou score non reproduit = exclu des prix.
- **Soumissions** : 5/jour, 100 total. Choisir **2 soumissions** pour le leaderboard privé (sinon les 2 meilleures publiques).

## 6. Règles clés

- Outils open-source uniquement ; données confidentielles (pas de partage/republication) ; suppression en fin de challenge.
- Pré-entraînés autorisés si **librement accessibles + licence commerciale** + déclarés avant la fin + comptés dans les 1 000 MB.
- **Texte biblique** : seul celui de `Train_religious.csv` est autorisé ; tout autre texte biblique en fongbe interdit.
- Données J'aime Ma Langue hors fichiers fournis : interdites.
- **Annotation/correction manuelle interdite** ; filtrage/pondération automatique des données incertaines autorisés.
- Pas d'automl ; pas de seuil bidon pour gonfler le leaderboard (probas brutes si métrique probabiliste).
- Top 10 privé → email, **48 h** pour soumettre le code ; tout user peut être sommé de fournir son code (24 h sinon).
- Gagnant → droits d'auteur du code cédés à Zindi ; paiement après code review.
- Équipes : max 4, leader reçoit le prix.

## 7. Check-list pour démarrer

1. Télécharger audio : `https://storage.googleapis.com/fongbe/` → `sha256sum -c SHA256SUMS` → `tar -xf audio_*.tar` (crée `audio/train_*/`, `audio/test/`).
2. `pip install -r requirements.txt` (pour `score.py` ; `syllabify.py` = stdlib seule).
3. Lancer `starter_notebook.ipynb` sur Colab/Kaggle T4 pour reproduire la baseline (~21 % WER).
4. Validation locale : voix `recommended_validation=1` de `Train_reliable` (~5,5 h).
5. Avant chaque soumission : `syllabify()` + vérifier le format `ID,Target`.
6. Avant de jouer les prix : `measure_frugality.py` doit passer (≤1 GB, RTF<1, ≤2 GB RAM).
