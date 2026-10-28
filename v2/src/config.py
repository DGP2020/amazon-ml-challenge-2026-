"""
Configuration & Tunable Hyperparameters for the Amazon ML Challenge 2026
Advanced Business Entity Resolution Pipeline.

Strictly aligned with Vladimir Gurevich's 2024 survey:
"Blocking Techniques for Large-Scale Entity Resolution: A Comprehensive Survey
and Architecture Analysis".

Engineered for 0.70+ Macro F0.5 performance with strict < 12 GB RAM footprint
(preventing OOM in Google Colab CPU and High-RAM runtimes).
"""

import os
import sys

# ============================================================================
# 1. File Paths & Directory Resolution (Colab & Local Dynamic Root Discovery)
# ============================================================================
def _get_current_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.abspath(os.getcwd())


_CURRENT_DIR = _get_current_dir()


def _find_project_root():
    p = _CURRENT_DIR
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "dataset")) and (
            os.path.isdir(os.path.join(p, "code")) or os.path.isdir(os.path.join(p, "output")) or os.path.isdir(os.path.join(p, "utils"))
        ):
            return os.path.abspath(p)
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent

    cwd = os.path.abspath(os.getcwd())
    if os.path.isdir(os.path.join(cwd, "dataset")):
        return cwd

    for base in [cwd, "/content", "/content/drive/MyDrive"]:
        if os.path.exists(base):
            if os.path.isdir(os.path.join(base, "dataset")):
                return os.path.abspath(base)
            try:
                for entry in os.listdir(base):
                    low = entry.lower()
                    if "amazon" in low or "challenge" in low or "challan" in low:
                        cand = os.path.join(base, entry)
                        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "dataset")):
                            return os.path.abspath(cand)
            except Exception:
                pass

    return cwd


PROJECT_ROOT     = _find_project_root()
ZIP_PATH         = os.environ.get("ER_ZIP_PATH", "")
EXTRACT_DIR      = os.environ.get("ER_EXTRACT_DIR", os.path.join(PROJECT_ROOT, "extracted_data"))
DATASET_DIR      = os.environ.get("ER_DATASET_DIR", os.path.join(PROJECT_ROOT, "dataset"))
OUTPUT_DIR       = os.environ.get("ER_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
DRIVE_BACKUP_DIR = os.environ.get("ER_DRIVE_BACKUP_DIR", "")

TEAM_NAME        = "Freiren"
RANDOM_SEED      = 42

# ============================================================================
# 2. Sampling & Split Ratios (Optimized for Peak Memory Safety)
# ============================================================================
TRAIN_SAMPLE_SIZE    = 200_000   # Stratified S1 entities sampled for training (~2M candidate pairs)
VAL_SPLIT_RATIO      = 0.15      # Held-out validation fraction for dual-threshold sweep
TEST_SPLIT_RATIO     = 0.10      # Completely held-out local test fraction for unbiased evaluation
CHUNK_SIZE_INFERENCE = 35_000    # Streamed chunk size: keeps peak feature extraction RAM strictly < 2.5 GB

# ============================================================================
# 3. Phase 2: JedAI 3-Tier Blocking Architecture (Survey §9.1, Figure 9 & §10.1)
# ============================================================================
# Stage 2A: Block Building Ensemble
# Blocker 1: Calibrated MinHash LSH (b=16 bands, r=4 rows -> s* = (1/16)^(1/4) = 0.50)
NUM_NAME_PERM        = 64        # 64 hash permutations on char 3-grams
LSH_NAME_BANDS       = 16        # b = 16 bands
LSH_NAME_ROWS        = 4         # r = 4 rows per band
NUM_ADDR_PERM        = 16        # 16 hash permutations for address 3-grams
LSH_ADDR_BANDS       = 4         # b = 4 bands
LSH_ADDR_ROWS        = 4         # r = 4 rows per band
MAX_LSH_BUCKET_SIZE  = 600       # Bucket size ceiling before pruning

# Blocker 2: Sparkly-Style BM25 Rare-Token Inverted Index (Survey §3.2, Algorithm 3)
BM25_MIN_IDF         = 2.5       # Only index rare distinguishing tokens with IDF > 2.5
BM25_K1              = 1.5
BM25_B               = 0.75
BM25_MAX_DF          = 600       # Cap document frequency to ignore generic stopwords
BM25_TOP_K_PER_TOKEN = 35        # Max candidates retrieved per rare token

# Blocker 3: Multi-Pass Sorted Neighborhood (Survey §2.1.2, Algorithm 2)
SN_WINDOW_SIZE       = 4         # Sliding window w=4 on Forward & Reversed normalized names

# Blocker 4: Numeric PIN/ZIP Conjunction Index
MIN_PIN_DIGITS       = 5         # PIN >= 5 digits + 3-char name prefix conjunction

# Blocker 5: Dense Vector ANN (Disabled on CPU to eliminate PyTorch tensor OOM)
ENABLE_DENSE_ANN     = os.environ.get("ER_ENABLE_DENSE_ANN", "false").lower()

# Stage 2B: Block Cleaning (Survey §9.1 & Figure 9)
BLOCK_PURGE_RATIO    = 0.005     # Drop blocks containing > 0.5% of partition entities
BLOCK_PURGE_MIN_CAP  = 150       # Minimum absolute block size before purging
BLOCK_FILTER_RHO     = 0.80      # Retain query in its smallest 80% matching blocks

# Stage 2C: Comparison Cleaning / Meta-Blocking (Survey §3.2.2 & §9.3)
USE_ARCS_WEIGHTING   = True      # Adamic-Adar edge weighting: sum(1 / log(|B_k| + e))
CNP_TOP_K_FACTOR     = 30        # Reciprocal Cardinality Node Pruning neighborhood
MAX_CANDIDATES_K2    = 15        # Per-source candidate budget: K2 <= 15 for Source 2
MAX_CANDIDATES_K3    = 15        # Per-source candidate budget: K3 <= 15 for Source 3
MAX_CANDIDATES_K     = 30        # Total Top-K candidates per S1 entity
MAX_NEGATIVES_TRAIN  = 15        # Hard negatives sampled per S1 entity for GBDT training

# Co-occurrence weight constants for fast scoring
WEIGHT_NAME_LSH      = 3.5
WEIGHT_ADDR_LSH      = 2.0
WEIGHT_BM25_RARE     = 4.0
WEIGHT_SORTED_NEIGH  = 2.5
WEIGHT_PIN_MATCH     = 4.5
WEIGHT_DOMAIN_MATCH  = 6.5
WEIGHT_PREFIX_4      = 3.0
WEIGHT_PREFIX_3_FALL = 1.0

# ============================================================================
# 4. Phase 3: 12 Pairwise Multimodal Feature Columns
# ============================================================================
FEATURE_COLS = [
    "name_ratio",              # 1. RapidFuzz Levenshtein similarity
    "name_token_sort_ratio",   # 2. Word-order invariant token similarity
    "name_token_set_ratio",    # 3. Subset/acronym invariant token similarity
    "name_jaro_winkler",       # 4. Prefix-boosted similarity
    "name_jaccard_3gram",      # 5. Character 3-gram IoU
    "name_semantic_sim",       # 6. Dense subword n-gram TF-cosine similarity
    "addr_token_set_ratio",    # 7. Token set similarity on normalized addresses
    "addr_number_iou",         # 8. Intersection over Union of street numbers & PIN codes
    "domain_exact_match",      # 9. Exact match indicator for extracted domains
    "legal_form_compatible",   # 10. Boolean indicator of legal form consistency
    "blocking_arcs_weight",    # 11. ARCS meta-blocking edge weight
    "blocking_shared_count",   # 12. Total distinct blocker count sharing the pair
]

# ============================================================================
# 5. Phase 4: LightGBM GBDT Hyperparameters & Dual-Threshold Sweep
# ============================================================================
LGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 800,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}

# Dual-threshold sweep grid: tau_S2, tau_S3 in [0.45, 0.90]
SWEEP_TAU_MIN  = 0.45
SWEEP_TAU_MAX  = 0.90
SWEEP_TAU_STEP = 0.03

# ============================================================================
# 6. Phase 5: JedAI Unique Mapping Clustering & Walmart L2 Hard Veto
# ============================================================================
MAX_S2_MATCHES_PER_S1 = 1        # Source clamping: max 1 S2 match per S1 entity
MAX_S3_MATCHES_PER_S1 = 1        # Source clamping: max 1 S3 match per S1 entity
ENABLE_HARD_VETO      = True     # Walmart L2 Hard Veto on conflicting domain AND PIN
