# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

This repository contains the end-to-end solution for the Amazon ML Challenge 2026 (Business Entity Resolution). The solution is heavily optimized for a high-RAM environment (e.g., Google Colab with 51 GB RAM) and heavily relies on multi-threaded data processing, MinHash Locality-Sensitive Hashing (LSH), and a LightGBM classifier.

## Hardware & Environment
- **Memory Requirements**: ~50 GB RAM recommended for full training and execution.
- **Optimized for**: Google Colab High-RAM instance.

## Pipeline Structure (Single Script)

All logic lives in a single, highly-optimized script (`846code.py`), organized logically into execution phases (cells):

- **Cell 1** — Config & dataset extraction (`ZIP_PATH`, `OUTPUT_DIR`, blocking/model hyperparameters, etc.).
- **Cell 2** — Text normalization: single-pass vectorised regex operations for legal-suffix canonicalization (US/India/France suffixes), noise-prefix/suffix stripping, street-abbreviation expansion, domain extraction, DBA detection, and `name_core` (suffix+stopword-stripped) construction. Uses fast unicode transliteration.
- **Cell 3** — Blocking / candidate generation: Strict declarative blocking (partitioning by country). Employs a Dual-View MinHash LSH on character 3-grams (64-permutation name index + 16-permutation address index) mapped into 64-bit integer bitmasks, augmented by exact-prefix, sorted-token-prefix, distinctive-word, domain, and address-number lookup boosts.
- **Cell 4** — Feature engineering: 13 similarity features per `(S1, candidate)` pair using RapidFuzz (Levenshtein, Jaro-Winkler, token sort, token set) and custom metrics (Jaccard, length ratio, numeric overlap). Includes multi-threaded negative sampling (keeps all positives and samples hard negatives) to accelerate feature extraction.
- **Cell 5** — Metric & threshold optimization: Computes per-entity macro F0.5 (with exact singleton credit handling). Performs global threshold+margin sweeps and per-country threshold sweeps.
- **Cell 6** — Training pipeline: Leak-free S1 entity split (train/val/local-test split applied *before* candidate generation), building blocking index, feature extraction, LightGBM classifier training (600 estimators, early stopping), and threshold selection.
- **Cell 7** — Test inference: Streamed in chunks of 150k S1 entities to constrain RAM usage. Incorporates candidate generation, feature building, LightGBM inference, and concludes with a global 1-to-1 greedy bipartite assignment to enforce uniqueness constraints.
- **Cell 8** — Row-count integrity checks (guarantees exactly one row per S1 entity) and optional automated Google Drive backup.
- **Cell 9** — Automatically executes the official `utils/validate_submission.py` against the generated output files to ensure zero formatting or schema errors before submission.

## Reproduction

1. Place the competition dataset zip at the path configured in `ZIP_PATH` (or ensure `DATASET_DIR` already contains `train/` and `test/`), and update `ZIP_PATH` / `DATASET_DIR` in `846code.py` to match your local environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python 846code.py
   ```
4. Outputs will be written to `OUTPUT_DIR` (`/content/output` by default):
   - `output/matching_results.tsv` — Final matches, ready for leaderboard submission.
   - `output/candidate_pairs.tsv` — Candidate pairs from the blocking stage.
   - `output/lgb_model.txt` — Saved LightGBM model booster.
   - `output/audit_candidates.tsv` — Candidate pairs with a predicted probability in `[0.50, 0.65)`, provided for manual auditing.
5. Upon completion, Cell 9 automatically runs `utils/validate_submission.py` against the two output TSV files and prints PASS/FAIL with any identified issues.

## Key Notes

- **Data Sampling:** `TRAIN_SAMPLE_SIZE = 200_000` S1 rows are randomly sampled from `train_source1.tsv` for training efficiency.
- **Zero Cross-Country Matching:** The country partition is strictly enforced. Candidates are only ever generated within the identical country partition (US↔US, India↔India, France↔France), cutting the search space by 60%+ with 0 recall loss.
- **Leak-Free Validation:** The dataset is split into training, validation, and local test sets *before* building the blocking LSH indexes. This prevents information bleed between splits.
- **Greedy Global Assignment:** The final matching step applies a global greedy 1-to-1 bipartite assignment across all predicted pairs (each S2/S3 candidate is claimed by at most one S1 entity, highest-probability claim wins), mathematically based on the ground-truth property that every S2/S3 entity maps to at most one S1 entity.
