# Amazon ML Challenge 2026 — Advanced Business Entity Resolution Pipeline

This repository implements the complete end-to-end Business Entity Resolution pipeline in a single unified Python file (`src/pipeline.py`) as specified in `ml challenge plan.md` and grounded in **Vladimir Gurevich's 2024 survey**: *"Blocking Techniques for Large-Scale Entity Resolution: A Comprehensive Survey and Architecture Analysis"*.

## Unified Architecture (`src/pipeline.py`)

1. **Phase 1: Ingestion & Multilingual Normalization**
   - Indic transliteration (`sanscript.ITRANS`) + Romanization phonetic vowel/consonant folding (`shree`/`sri`, `v`/`w`, `aa->a`, `ee->i`, `oo->u`).
   - Diacritic accent stripping (`unidecode` for French `é, è, ê, ç, ô -> e, e, e, c, o`).
   - Dedicated legal form extraction (`legal_form` across US, India, and France) and street abbreviation expansion.

2. **Phase 2: JedAI 3-Tier Blocking Architecture (Survey §9.1 Fig 9, §9.3, §10.1)**
   - **Stage 2A (Block Building Ensemble)**:
     1. Strict Country Partitioning (`US`, `India`, `France`, `UNKNOWN`)
     2. Calibrated MinHash LSH ($b=16$ bands, $r=4$ rows, $s^* \approx 0.50$)
     3. Sparkly-Style BM25 Rare-Token Inverted Index ($\text{IDF} > 2.5$)
     4. Multi-Pass Sorted Neighborhood (Forward & Reversed, sliding window $w=4$)
     5. Numeric PIN/ZIP Conjunction Index ($\ge 5$ digits + 3-char name prefix) & Domain Index
     6. Multilingual Dense Vector ANN (`multilingual-e5-small` + FAISS IVF-Flat top-15)
   - **Stage 2B (Block Cleaning)**:
     - **Block Purging**: Discards hyper-frequent blocks ($|B_k| > 0.005 \times N$).
     - **Block Filtering**: Retains each entity in its smallest $80\%$ blocks ($\rho = 0.80$).
   - **Stage 2C (Comparison Cleaning / Meta-Blocking)**:
     - **ARCS Edge Weighting**: $W(s_1, t) = \sum_{B_k} \frac{1}{\log |B_k|}$.
     - **Reciprocal Cardinality Node Pruning (CNP)** & per-source budgeting ($K_2 \le 15$ for S2, $K_3 \le 15$ for S3, $C_{\text{avg}} \le 25$).

3. **Phase 3: Pairwise Multimodal Feature Engineering (12 Features)**
   - `name_ratio`, `name_token_sort_ratio`, `name_token_set_ratio`, `name_jaro_winkler`, `name_jaccard_3gram`, `name_semantic_sim`, `addr_token_set_ratio`, `addr_number_iou`, `domain_exact_match`, `legal_form_compatible`, `blocking_arcs_weight`, `blocking_shared_count`.

4. **Phase 4: LightGBM GBDT Classifier, LOCO CV & Dual-Threshold Sweep**
   - Leave-One-Country-Out (LOCO) Cross-Validation ($\text{US} \leftrightarrow \text{India}$) + SGER training augmentation.
   - Brier Score (MSE), Binary Log-Loss, PR-AUC, and $R^2$ calibration metrics.
   - Dual-threshold optimization ($\tau_{S2}, \tau_{S3} \in [0.45, 0.90]$) + sparsity-binned threshold refinement.

5. **Phase 5: JedAI Unique Mapping Clustering & Walmart L2 Hard Veto**
   - Enforces Walmart L2 Hard Veto on actively conflicting populated domains and postal codes.
   - Global confidence sort with 1-to-1 target exclusivity and source clamping (at most 1 S2 and 1 S3 match per S1).

---

## How to Execute in Google Colab

```bash
# Step 1: Install dependencies
!pip install -q lightgbm rapidfuzz unidecode indic-transliteration pyarrow sentence-transformers faiss-cpu

# Step 2: Run unified single-file pipeline
!python code/business_entity_resolution/src/pipeline.py

# Step 3: Run submission validator
!python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# Step 4: Package final submission zip
!zip -r team_submission.zip output/ code/ Documentation_template.md
```
