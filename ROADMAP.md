# Roadmap — ASR Fongbe (Frugal Tonal ASR)

## Catalogue des approches ASR

### Niveau 1 — Approches fondamentales : modèles et apprentissage
1. Transfer Learning
2. Fine-tuning complet (Full Fine-tuning)
3. Partial Fine-tuning
4. Layer Freezing / Progressive Unfreezing
5. Parameter-Efficient Fine-tuning (PEFT)
6. LoRA / QLoRA
7. Adapters
8. Training from Scratch
9. Self-supervised Pretraining
10. Continued Pretraining
11. Semi-supervised Learning
12. Self-training
13. Pseudo-labeling
14. Weakly Supervised Learning
15. Contrastive Learning
16. Multi-task Learning
17. Joint Audio-Text Learning
18. Knowledge Distillation

### Niveau 2 — Optimisation des données
19. Data Augmentation
20. Speed Perturbation
21. Pitch Perturbation
22. Noise Augmentation
23. SpecAugment
24. Data Filtering
25. Confidence Filtering
26. Curriculum Learning
27. Active Learning
28. Speaker-aware Training
29. Data Rebalancing
30. Weighted Sampling
31. Hard Example Mining

### Niveau 3 — Architectures ASR alternatives
32. CTC
33. RNN-T
34. Attention Encoder-Decoder
35. Transformer ASR
36. Conformer
37. Zipformer
38. Emformer
39. Hybrid CTC-Attention
40. Encoder-Decoder Transducer

### Niveau 4 — Modélisation linguistique et décodage
41. Character Language Model
42. Subword Language Model
43. Phoneme Language Model
44. N-gram Language Model
45. Neural Language Model
46. CTC Greedy Decoding
47. CTC Beam Search
48. Shallow Fusion
49. Lexicon-based Decoding
50. Reranking

### Niveau 5 — Spécialisation tonale fongbe
51. Tone-aware CTC
52. Multi-head Tone Prediction
53. Joint Phoneme-Tone Modeling
54. Tone Classification
55. Prosody-aware Modeling
56. Multi-task Phonetic-Tonal Learning
57. Tone-conditioned Decoding
58. Syllable-level Modeling
59. Explicit Tone Embeddings
60. Tone Error Correction

### Niveau 6 — Compression et frugalité
61. Dynamic Quantization
62. Static Quantization
63. Quantization-aware Training
64. Pruning
65. Structured Pruning
66. Knowledge Distillation
67. Low-rank Factorization
68. Weight Sharing
69. Encoder Compression
70. Efficient Attention

### Niveau 7 — Combinaisons et optimisation finale
71. Model Ensembling
72. Checkpoint Averaging
73. Multi-model Distillation
74. Hybrid Acoustic Modeling
75. Two-stage ASR
76. Error-driven Training
77. Architecture Search
78. Joint Acoustic-Linguistic Optimization

## Méthode de travail

- Implémenter chaque technique.
- Entraîner et évaluer séparément.
- Mesurer le WER syllabique et les erreurs tonales.
- Mesurer taille, RTF et RAM.
- Conserver les meilleurs checkpoints.
- Combiner progressivement les techniques gagnantes.
