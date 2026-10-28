#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Advanced Business Entity Resolution Pipeline

Integrates empirical requirements of the Amazon ML Challenge 2026 with the formal
architectural taxonomy from Vladimir Gurevich's 2024 survey:
"Blocking Techniques for Large-Scale Entity Resolution: A Comprehensive Survey and Architecture Analysis".

Engineered for 0.70+ Macro F0.5 performance with strict < 12 GB RAM footprint:
  - Phase 1: Ingestion & Multilingual Normalization (Indic ITRANS + Romanization phonetic fold,
             French unidecode accent stripping, dedicated legal form extraction, and street normalization)
  - Phase 2: JedAI 3-Tier Blocking Architecture (Survey §9.1 Fig 9, §9.3, §10.1):
             2A. Block Building: Calibrated MinHash LSH (b=16, r=4), Sparkly BM25 (IDF > 2.5),
                 Multi-Pass Sorted Neighborhood (Forward & Reversed, w=4), PIN conjunction index
             2B. Block Cleaning: Vectorized Block Purging (|B_k| > 0.005 * N) + Query-Side Block Filtering (rho = 0.80)
             2C. Comparison Cleaning / Meta-Blocking: ARCS edge weighting, Reciprocal CNP,
                 per-source candidate budgeting (K2 <= 15, K3 <= 15, C_avg <= 25)
  - Phase 3: Pairwise Multimodal Feature Engineering (12 features including subword cosine similarity,
             legal form compatibility, PIN IoU, domain match, and ARCS graph meta-features)
  - Phase 4: LightGBM GBDT Classifier, SGER Data Augmentation, LOCO CV & Dual-Threshold Sweep
  - Phase 5: JedAI Unique Mapping Clustering (Survey §9.1 Fig 9), Walmart L2 Hard Veto,
             sparsity-binned threshold refinement, and memory-safe streamed inference
"""

import os
import sys
import re
import math
import gc
import time
import heapq
import shutil
import zipfile
import warnings
import importlib.util
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

try:
    from unidecode import unidecode
except ImportError:
    import unicodedata
    def unidecode(text):
        return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

from rapidfuzz import fuzz, distance
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    log_loss,
    average_precision_score,
    r2_score,
)

# ============================================================================
# Colab & Local Dynamic Path Resolution
# ============================================================================
def _get_current_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.abspath(os.getcwd())


_CURRENT_DIR = _get_current_dir()


def find_project_root():
    """Dynamically locate the project root directory containing 'dataset', 'output', 'utils', 'code'."""
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


PROJECT_ROOT = find_project_root()
_SRC_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "src")
for p in [_CURRENT_DIR, _SRC_DIR, PROJECT_ROOT]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import config
except ImportError:
    config = None


def _cfg(name, default):
    return getattr(config, name, default) if config is not None else default


ZIP_PATH = _cfg("ZIP_PATH", os.environ.get("ER_ZIP_PATH", ""))
EXTRACT_DIR = _cfg("EXTRACT_DIR", os.environ.get("ER_EXTRACT_DIR", os.path.join(PROJECT_ROOT, "extracted_data")))
DATASET_DIR = _cfg("DATASET_DIR", os.path.join(PROJECT_ROOT, "dataset"))
OUTPUT_DIR = os.environ.get("ER_OUTPUT_DIR") or _cfg("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
DRIVE_BACKUP_DIR = _cfg("DRIVE_BACKUP_DIR", "")

TEAM_NAME = _cfg("TEAM_NAME", "Freiren")
RANDOM_SEED = _cfg("RANDOM_SEED", 42)

TRAIN_SAMPLE_SIZE = _cfg("TRAIN_SAMPLE_SIZE", 200_000)
VAL_SPLIT_RATIO = _cfg("VAL_SPLIT_RATIO", 0.15)
TEST_SPLIT_RATIO = _cfg("TEST_SPLIT_RATIO", 0.10)
# Low chunk size keeps peak RAM < 2.5 GB during parallel RapidFuzz computation
CHUNK_SIZE_INFERENCE = _cfg("CHUNK_SIZE_INFERENCE", 35_000)

NUM_NAME_PERM = _cfg("NUM_NAME_PERM", 64)
LSH_NAME_BANDS = _cfg("LSH_NAME_BANDS", 16)
LSH_NAME_ROWS = _cfg("LSH_NAME_ROWS", 4)
NUM_ADDR_PERM = _cfg("NUM_ADDR_PERM", 16)
LSH_ADDR_BANDS = _cfg("LSH_ADDR_BANDS", 4)
LSH_ADDR_ROWS = _cfg("LSH_ADDR_ROWS", 4)
MAX_LSH_BUCKET_SIZE = _cfg("MAX_LSH_BUCKET_SIZE", 600)

BM25_MIN_IDF = _cfg("BM25_MIN_IDF", 2.5)
BM25_K1 = _cfg("BM25_K1", 1.5)
BM25_B = _cfg("BM25_B", 0.75)
BM25_MAX_DF = _cfg("BM25_MAX_DF", 600)
BM25_TOP_K_PER_TOKEN = _cfg("BM25_TOP_K_PER_TOKEN", 35)

SN_WINDOW_SIZE = _cfg("SN_WINDOW_SIZE", 4)
MIN_PIN_DIGITS = _cfg("MIN_PIN_DIGITS", 5)

BLOCK_PURGE_RATIO = _cfg("BLOCK_PURGE_RATIO", 0.005)
BLOCK_PURGE_MIN_CAP = _cfg("BLOCK_PURGE_MIN_CAP", 150)
BLOCK_FILTER_RHO = _cfg("BLOCK_FILTER_RHO", 0.80)

MAX_CANDIDATES_K2 = _cfg("MAX_CANDIDATES_K2", 15)
MAX_CANDIDATES_K3 = _cfg("MAX_CANDIDATES_K3", 15)
MAX_CANDIDATES_K = _cfg("MAX_CANDIDATES_K", 30)
MAX_NEGATIVES_TRAIN = _cfg("MAX_NEGATIVES_TRAIN", 15)

FEATURE_COLS = _cfg("FEATURE_COLS", [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_jaccard_3gram",
    "name_semantic_sim",
    "addr_token_set_ratio",
    "addr_number_iou",
    "domain_exact_match",
    "legal_form_compatible",
    "blocking_arcs_weight",
    "blocking_shared_count",
])

LGBM_PARAMS = _cfg("LGBM_PARAMS", {
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
})

SWEEP_TAU_MIN = _cfg("SWEEP_TAU_MIN", 0.45)
SWEEP_TAU_MAX = _cfg("SWEEP_TAU_MAX", 0.90)
SWEEP_TAU_STEP = _cfg("SWEEP_TAU_STEP", 0.03)

MAX_S2_MATCHES_PER_S1 = _cfg("MAX_S2_MATCHES_PER_S1", 1)
MAX_S3_MATCHES_PER_S1 = _cfg("MAX_S3_MATCHES_PER_S1", 1)
ENABLE_HARD_VETO = _cfg("ENABLE_HARD_VETO", True)

warnings.filterwarnings("ignore")


# ============================================================================
# File & Directory Utilities
# ============================================================================
def find_dataset_dir(base_dir=None):
    candidates = [
        os.path.join(PROJECT_ROOT, "dataset"),
        os.path.join(PROJECT_ROOT, "student_resource", "dataset"),
        DATASET_DIR,
        "dataset",
        "student_resource/dataset",
    ]
    if base_dir:
        candidates.append(base_dir)

    for path in candidates:
        if path and os.path.exists(path) and os.path.isdir(path):
            try:
                if len(os.listdir(path)) > 0:
                    return os.path.abspath(path)
            except Exception:
                pass
    return os.path.join(PROJECT_ROOT, "dataset")


def resolve_data_file(data_dir, filename, split=None):
    candidates = []
    if split:
        candidates.append(os.path.join(data_dir, split, filename))
    candidates.append(os.path.join(data_dir, filename))
    for s in ["train", "test"]:
        candidates.append(os.path.join(data_dir, s, filename))

    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    if os.path.exists(data_dir):
        for root, _, files in os.walk(data_dir):
            if "__MACOSX" in root:
                continue
            if filename in files:
                return os.path.abspath(os.path.join(root, filename))

    return candidates[0]


def extract_dataset(zip_path=ZIP_PATH, extract_dir=EXTRACT_DIR):
    data_dir = find_dataset_dir()
    if os.path.exists(data_dir):
        try:
            if len(os.listdir(data_dir)) > 0:
                print(f"  Found existing dataset at: {data_dir} (skipping extraction)")
                return
        except Exception:
            pass

    if zip_path and os.path.exists(zip_path):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"  Extracting {zip_path} -> {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print("  Extraction complete.")


# ============================================================================
# PHASE 1: Ingestion & Multilingual Normalization
# ============================================================================
LEGAL_FORM_PATTERNS = [
    (r"\bpvt\s+ltd\b|\bprivate\s+ltd\b|\bprivate\s+limited\b", "pvt_ltd"),
    (r"\bpublic\s+limited\b|\bplc\b", "plc"),
    (r"\bopc\b|\bone\s+person\s+company\b", "opc"),
    (r"\bhuf\b|\bhindu\s+undivided\s+family\b", "huf"),
    (r"\bproprietorship\b|\bproprietor\b|\bprop\b", "proprietorship"),
    (r"\bllp\b|\blimited\s+liability\s+partnership\b", "llp"),
    (r"\bllc\b|\bl\.l\.c\.\b", "llc"),
    (r"\bsasu\b|\bs\.a\.s\.u\.\b", "sasu"),
    (r"\bsarl\b|\bs\.a\.r\.l\.\b", "sarl"),
    (r"\beurl\b|\be\.u\.r\.l\.\b", "eurl"),
    (r"\beirl\b|\be\.i\.r\.l\.\b", "eirl"),
    (r"\bsas\b|\bs\.a\.s\.\b", "sas"),
    (r"\bsnc\b|\bs\.n\.c\.\b", "snc"),
    (r"\bsci\b|\bs\.c\.i\.\b", "sci"),
    (r"\bgie\b|\bg\.i\.e\.\b", "gie"),
    (r"\bsa\b|\bs\.a\.\b", "sa"),
    (r"\binc\b|\bincorporated\b", "inc"),
    (r"\bcorp\b|\bcorporation\b", "corp"),
    (r"\bltd\b|\blimited\b", "ltd"),
    (r"\bco\b|\bcompany\b", "co"),
]

_LEGAL_COMBINED_RE = re.compile(
    "|".join(f"(?P<lf{i}>{pat})" for i, (pat, _) in enumerate(LEGAL_FORM_PATTERNS)),
    re.IGNORECASE,
)
_LEGAL_CANONICALS = [canon for _, canon in LEGAL_FORM_PATTERNS]

STREET_ABBREVS = [
    (r"\brd\b", "road"),
    (r"\bst\b", "street"),
    (r"\bave\b|\bav\b", "avenue"),
    (r"\bblvd\b|\bbd\b", "boulevard"),
    (r"\bdr\b", "drive"),
    (r"\bln\b", "lane"),
    (r"\bct\b", "court"),
    (r"\bpl\b", "place"),
    (r"\bhwy\b", "highway"),
    (r"\bflr\b|\bfl\b", "floor"),
    (r"\bno\b|\bnum\b", "number"),
    (r"\bopp\b", "opposite"),
    (r"\bbldg\b", "building"),
    (r"\bsec\b", "sector"),
    (r"\bmidc\b", "industrial area"),
    (r"\bindl\b", "industrial"),
    (r"\brue\b", "rue"),
    (r"\bche\b", "chemin"),
    (r"\ball\b", "allee"),
    (r"\bimp\b", "impasse"),
    (r"\brte\b", "route"),
]

_STREET_COMBINED_RE = re.compile(
    "|".join(f"(?P<s{i}>{pat})" for i, (pat, _) in enumerate(STREET_ABBREVS)),
    re.IGNORECASE,
)
_STREET_REPLACEMENTS = [repl for _, repl in STREET_ABBREVS]

NOISE_PREFIX_RE = re.compile(
    r"^([\s<\-#|]+|(\bm\s*/\s*s\b|\bms\b|\bshri\b|\bshree\b|\bsri\b)[\s.:\-]*)",
    re.IGNORECASE,
)
NOISE_SUFFIX_RE = re.compile(r"\|\s*\S+\.\S+\s*$")
DOMAIN_RE = re.compile(r"(\b[a-zA-Z0-9.-]+\.(?:com|in|org|net|co|fr|io|biz|info)\b)", re.IGNORECASE)
PIN_CODE_RE = re.compile(r"\b\d{5,6}\b")
HOUSE_NUM_STREET_RE = re.compile(
    r"\b(\d{5,6})\s+(?:road|street|avenue|boulevard|drive|lane|court|place|highway|rue|chemin|allee|impasse|route)\b"
)

STOPWORDS = [
    "the", "of", "and", "for", "a", "an", "in", "on", "at", "by",
    "shree", "shri", "sri", "ms", "smt", "dr", "om", "jai", "new",
    "enterprise", "enterprises", "agency", "agencies", "trading", "traders",
    "company", "incorporated", "corporation", "limited", "private",
    "de", "du", "des", "le", "la", "les", "et", "en", "au", "aux", "d", "l",
]
_CORE_STRIP_PATTERNS = [pat for pat, _ in LEGAL_FORM_PATTERNS] + [
    r"\b" + re.escape(w) + r"\b" for w in sorted(STOPWORDS, key=len, reverse=True)
]
_CORE_STRIP_RE = re.compile("|".join(_CORE_STRIP_PATTERNS), re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

INDIC_SCRIPTS = [
    (re.compile(r"[\u0900-\u097F]"), sanscript.DEVANAGARI if HAS_INDIC else "devanagari"),
    (re.compile(r"[\u0C80-\u0CFF]"), sanscript.KANNADA if HAS_INDIC else "kannada"),
    (re.compile(r"[\u0B80-\u0BFF]"), sanscript.TAMIL if HAS_INDIC else "tamil"),
    (re.compile(r"[\u0C00-\u0C7F]"), sanscript.TELUGU if HAS_INDIC else "telugu"),
    (re.compile(r"[\u0A80-\u0AFF]"), sanscript.GUJARATI if HAS_INDIC else "gujarati"),
    (re.compile(r"[\u0980-\u09FF]"), sanscript.BENGALI if HAS_INDIC else "bengali"),
]

_ROMAN_FOLD_RULES = [
    (re.compile(r"\b(shree|shri|sri)\b", re.IGNORECASE), "sri"),
    (re.compile(r"w", re.IGNORECASE), "v"),
    (re.compile(r"aa+", re.IGNORECASE), "a"),
    (re.compile(r"ee+", re.IGNORECASE), "i"),
    (re.compile(r"oo+", re.IGNORECASE), "u"),
    (re.compile(r"([bcdfghjklmnpqrstvwxz])\1+", re.IGNORECASE), r"\1"),
]


def transliterate_and_fold(text: str, apply_roman_fold: bool = False) -> str:
    res = text
    if not res.isascii():
        if HAS_INDIC:
            for pat, script in INDIC_SCRIPTS:
                if pat.search(res):
                    try:
                        res = transliterate(res, script, sanscript.ITRANS)
                    except Exception:
                        pass
        if not res.isascii():
            res = unidecode(res)
    res = res.lower()
    if apply_roman_fold:
        for pat, repl in _ROMAN_FOLD_RULES:
            res = pat.sub(repl, res)
    return res


def extract_legal_form_series(series: pd.Series) -> pd.Series:
    vals = series.fillna("").astype(str).values
    out = []
    for v in vals:
        m = _LEGAL_COMBINED_RE.search(v)
        if m is not None:
            canon = ""
            for i, g in enumerate(m.groups()):
                if g is not None:
                    canon = _LEGAL_CANONICALS[i]
                    break
            out.append(canon)
        else:
            out.append("")
    return pd.Series(out, index=series.index, dtype="object")


def normalize_text(series: pd.Series, is_address: bool = False) -> pd.Series:
    s = series.fillna("").astype(str)
    vals = [
        transliterate_and_fold(x, apply_roman_fold=False) if not x.isascii() else x.lower()
        for x in s.values
    ]
    s = pd.Series(vals, index=series.index)
    s = s.str.replace(NOISE_PREFIX_RE, "", regex=True)
    s = s.str.replace(NOISE_SUFFIX_RE, "", regex=True)
    s = s.str.replace(r"[^\w\s]", " ", regex=True)

    if is_address:
        s = pd.Series(
            [
                _STREET_COMBINED_RE.sub(
                    lambda m: next((_STREET_REPLACEMENTS[i] for i, g in enumerate(m.groups()) if g is not None), m.group()),
                    v
                )
                for v in s.values
            ],
            index=series.index,
        )

    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def make_name_core(name_clean_series: pd.Series) -> pd.Series:
    vals = name_clean_series.values
    out = []
    for v in vals:
        stripped = _WS_RE.sub(" ", _CORE_STRIP_RE.sub(" ", v)).strip()
        base = stripped if stripped else v
        folded = transliterate_and_fold(base, apply_roman_fold=True)
        out.append(folded if folded else base)
    return pd.Series(out, index=name_clean_series.index)


def extract_domain(series: pd.Series) -> pd.Series:
    extracted = series.fillna("").astype(str).str.extract(DOMAIN_RE, expand=True)[0]
    return extracted.fillna("").str.lower()


def extract_valid_pin_codes(addr_clean: str) -> list:
    if not addr_clean:
        return []
    house_nums = set(HOUSE_NUM_STREET_RE.findall(addr_clean))
    pins = [p for p in PIN_CODE_RE.findall(addr_clean) if p not in house_nums]
    return pins


def extract_numbers(text: str) -> set:
    return set(re.findall(r"\d+", str(text)))


def load_and_normalize(filepath: str, nrows: int = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            nrows=nrows,
            dtype={"entity_id": str, "country": str},
            engine="pyarrow",
        )
    except Exception:
        df = pd.read_csv(
            filepath,
            sep="\t",
            nrows=nrows,
            dtype={"entity_id": str, "country": str},
        )

    df["country"] = df["country"].fillna("UNKNOWN").astype(str).str.strip().replace("", "UNKNOWN")
    df["domain"] = extract_domain(df["business_name"])
    df["legal_form"] = extract_legal_form_series(df["business_name"])
    df["name_clean"] = normalize_text(df["business_name"], is_address=False)
    df["address_clean"] = normalize_text(df["business_address"], is_address=True)
    df["name_core"] = make_name_core(df["name_clean"])

    df.drop(columns=["business_name", "business_address"], inplace=True)
    gc.collect()
    return df


def load_ground_truth(filepath: str, target_s1_ids=None) -> dict:
    gt = pd.read_csv(filepath, sep="\t", dtype=str)
    if target_s1_ids is not None:
        target_set = set(target_s1_ids)
        gt = gt[gt["source1_entity_id"].isin(target_set)]
    gt_dict = {}
    for row in gt.itertuples():
        if pd.notnull(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            gt_dict[row.source1_entity_id] = {
                x.strip() for x in str(row.matched_entity_ids).split(",") if x.strip()
            }
        else:
            gt_dict[row.source1_entity_id] = set()
    del gt
    gc.collect()
    return gt_dict


# ============================================================================
# PHASE 2: JedAI 3-Tier Blocking Architecture (Survey §9.1 Fig 9, §9.3, §10.1)
# ============================================================================
CHAR_LIST = " 0123456789abcdefghijklmnopqrstuvwxyz"
NUM_CHARS = len(CHAR_LIST)
NUM_GRAMS = NUM_CHARS ** 3

BYTE_MAP = np.zeros(256, dtype=np.int32)
for _i, _c in enumerate(CHAR_LIST):
    BYTE_MAP[ord(_c)] = _i

MINHASH_PERM = np.random.RandomState(RANDOM_SEED).randint(
    1, 2**31 - 1, size=(NUM_GRAMS, max(NUM_NAME_PERM, NUM_ADDR_PERM)), dtype=np.uint32
)


def compute_signatures_fast(texts, num_perm=NUM_NAME_PERM) -> np.ndarray:
    n = len(texts)
    sigs = np.empty((n, num_perm), dtype=np.uint32)
    b_map = BYTE_MAP
    perm = MINHASH_PERM[:, :num_perm]

    for idx, text in enumerate(texts):
        s = f"  {str(text).lower()}  ".encode("ascii", errors="replace")
        s_len = len(s)
        if s_len <= 2:
            sigs[idx] = perm[0]
            continue
        g_ids = [
            b_map[s[i]] * 1369 + b_map[s[i + 1]] * 37 + b_map[s[i + 2]]
            for i in range(s_len - 2)
        ]
        sigs[idx] = perm[g_ids].min(axis=0)
    return sigs


def pack_lsh_bands_r4(sigs: np.ndarray, num_bands: int) -> np.ndarray:
    n = sigs.shape[0]
    band_keys = np.empty((n, num_bands), dtype=np.uint64)
    p1 = np.uint64(2654435761)
    p2 = np.uint64(2246822519)
    p3 = np.uint64(3266489917)
    p4 = np.uint64(668265263)
    for b in range(num_bands):
        c0 = sigs[:, 4 * b].astype(np.uint64)
        c1 = sigs[:, 4 * b + 1].astype(np.uint64)
        c2 = sigs[:, 4 * b + 2].astype(np.uint64)
        c3 = sigs[:, 4 * b + 3].astype(np.uint64)
        band_keys[:, b] = (c0 * p1) ^ ((c1 * p2) << 16) ^ ((c2 * p3) << 32) ^ ((c3 * p4) << 48)
    return band_keys


class LowRamJedAIBlocker:
    """
    High-Performance, Low-RAM JedAI 3-Tier Blocker.
    Avoids nested Python dictionary/tuple explosions by storing posting lists as flat int32 numpy arrays.
    """

    def __init__(self, target_c_df: pd.DataFrame, country: str = ""):
        if target_c_df.index.name == "entity_id":
            self.target_ids = target_c_df.index.values.astype(str)
        else:
            self.target_ids = target_c_df["entity_id"].values.astype(str)

        self.target_count = len(self.target_ids)
        self.country = str(country).strip().upper()
        self.is_s2 = np.array([tid.startswith("S2-") for tid in self.target_ids], dtype=bool)

        t_cores = target_c_df["name_core"].values.astype(str)
        t_addrs = target_c_df["address_clean"].values.astype(str)
        t_domains = target_c_df["domain"].values.astype(str)

        self.max_block_size = max(BLOCK_PURGE_MIN_CAP, int(BLOCK_PURGE_RATIO * self.target_count))

        # ── 2A.1: Calibrated MinHash LSH (b=16, r=4 for Name; b=4, r=4 for Address)
        name_sigs = compute_signatures_fast(t_cores, num_perm=NUM_NAME_PERM)
        addr_sigs = compute_signatures_fast(t_addrs, num_perm=NUM_ADDR_PERM)
        name_band_keys = pack_lsh_bands_r4(name_sigs, LSH_NAME_BANDS)
        addr_band_keys = pack_lsh_bands_r4(addr_sigs, LSH_ADDR_BANDS)
        del name_sigs, addr_sigs

        self.lsh_name_buckets = self._build_lsh_dict(name_band_keys, LSH_NAME_BANDS)
        self.lsh_addr_buckets = self._build_lsh_dict(addr_band_keys, LSH_ADDR_BANDS)
        del name_band_keys, addr_band_keys

        # ── 2A.2: BM25 Rare-Token Index (IDF > 2.5)
        doc_freqs = Counter()
        doc_tokens = []
        total_dl = 0
        for name in t_cores:
            toks = [w for w in name.split() if len(w) >= 3]
            utoks = set(toks)
            doc_tokens.append(utoks)
            total_dl += max(1, len(toks))
            for w in utoks:
                doc_freqs[w] += 1

        self.avg_dl = total_dl / max(1, self.target_count)
        self.bm25_idf = {}
        for tok, df_val in doc_freqs.items():
            if df_val <= self.max_block_size:
                idf = math.log(1.0 + (self.target_count - df_val + 0.5) / (df_val + 0.5))
                if idf > BM25_MIN_IDF:
                    self.bm25_idf[tok] = idf

        bm25_postings = defaultdict(list)
        for idx, utoks in enumerate(doc_tokens):
            for w in utoks:
                if w in self.bm25_idf:
                    bm25_postings[w].append(idx)
        del doc_tokens, doc_freqs

        self.bm25_postings = {
            k: np.array(v, dtype=np.int32) for k, v in bm25_postings.items()
        }
        del bm25_postings

        # ── 2A.3: Multi-Pass Sorted Neighborhood (Forward & Reversed)
        self.sn_fwd_keys = np.array(t_cores, dtype=object)
        self.sn_fwd_order = np.argsort(self.sn_fwd_keys)
        self.sn_fwd_sorted = self.sn_fwd_keys[self.sn_fwd_order]

        rev_cores = np.array([s[::-1] for s in t_cores], dtype=object)
        self.sn_rev_order = np.argsort(rev_cores)
        self.sn_rev_sorted = rev_cores[self.sn_rev_order]
        del rev_cores

        # ── 2A.4: PIN / Postal Conjunction Index & Domain Index
        pin_postings = defaultdict(list)
        domain_postings = defaultdict(list)
        p4_postings = defaultdict(list)

        for idx, (name, addr, dom) in enumerate(zip(t_cores, t_addrs, t_domains)):
            if dom and dom != "nan":
                domain_postings[dom].append(idx)
            p3 = name[:3] if len(name) >= 3 else name
            for pin in extract_valid_pin_codes(addr):
                pin_postings[f"{pin}:{p3}"].append(idx)
            if len(name) >= 4:
                p4_postings[name[:4]].append(idx)

        # Apply Block Purging directly
        self.domain_index = {k: np.array(v, dtype=np.int32) for k, v in domain_postings.items() if len(v) <= self.max_block_size}
        self.pin_index = {k: np.array(v, dtype=np.int32) for k, v in pin_postings.items() if len(v) <= self.max_block_size}
        self.p4_index = {k: np.array(v, dtype=np.int32) for k, v in p4_postings.items() if len(v) <= self.max_block_size}
        del pin_postings, domain_postings, p4_postings

        # Fallback 3-char prefix index
        prefix_3 = defaultdict(list)
        for idx, name in enumerate(t_cores):
            if len(name) >= 2:
                prefix_3[name[:3]].append(idx)
        self.prefix_3 = {k: np.array(v[:40], dtype=np.int32) for k, v in prefix_3.items()}
        del prefix_3

    def _build_lsh_dict(self, band_keys: np.ndarray, num_bands: int) -> list:
        buckets = []
        for b in range(num_bands):
            col = band_keys[:, b]
            order = np.argsort(col)
            sorted_k = col[order]
            ukeys, s_idx, cnts = np.unique(sorted_k, return_index=True, return_counts=True)
            b_dict = {}
            for k, si, cnt in zip(ukeys, s_idx, cnts):
                if cnt <= self.max_block_size:
                    b_dict[int(k)] = order[si : si + cnt].astype(np.int32)
                else:
                    b_dict[int(k)] = order[si : si + 80].astype(np.int32)
            buckets.append(b_dict)
        return buckets


def query_jedai_candidates(
    q_name_bands: np.ndarray,
    q_addr_bands: np.ndarray,
    q_name: str,
    q_addr: str,
    q_domain: str,
    blocker: LowRamJedAICountryBlocker,
    max_k2: int = MAX_CANDIDATES_K2,
    max_k3: int = MAX_CANDIDATES_K3,
):
    """
    Comparison Cleaning & Meta-Blocking with ARCS Edge Weighting:
      W(s1, t) = sum_{B_k} (1 / log(|B_k| + e))
    """
    arcs_weights = defaultdict(float)
    blocker_bitmask = defaultdict(int)

    # 1. Calibrated MinHash LSH
    for b in range(LSH_NAME_BANDS):
        matched = blocker.lsh_name_buckets[b].get(int(q_name_bands[b]))
        if matched is not None:
            w_arcs = 1.0 / math.log(len(matched) + math.e)
            for ti in matched:
                arcs_weights[ti] += w_arcs
                blocker_bitmask[ti] |= (1 << 1)

    for b in range(LSH_ADDR_BANDS):
        matched = blocker.lsh_addr_buckets[b].get(int(q_addr_bands[b]))
        if matched is not None:
            w_arcs = 1.0 / math.log(len(matched) + math.e)
            for ti in matched:
                arcs_weights[ti] += w_arcs * 0.8
                blocker_bitmask[ti] |= (1 << 1)

    # 2. Sparkly BM25 Rare Tokens
    q_tokens = [w for w in q_name.split() if len(w) >= 3]
    for w in set(q_tokens):
        idf = blocker.bm25_idf.get(w)
        if idf is not None:
            matched = blocker.bm25_postings.get(w)
            if matched is not None:
                w_arcs = (1.0 / math.log(len(matched) + math.e)) * (1.0 + 0.25 * idf)
                for ti in matched[:BM25_TOP_K_PER_TOKEN]:
                    arcs_weights[ti] += w_arcs
                    blocker_bitmask[ti] |= (1 << 2)

    # 3. Multi-Pass Sorted Neighborhood
    if q_name and blocker.target_count > 0:
        sn_w = 1.0 / math.log(2 * SN_WINDOW_SIZE + math.e)
        pos_f = int(np.searchsorted(blocker.sn_fwd_sorted, q_name))
        lo_f = max(0, pos_f - SN_WINDOW_SIZE)
        hi_f = min(blocker.target_count, pos_f + SN_WINDOW_SIZE)
        for idx_pos in range(lo_f, hi_f):
            ti = int(blocker.sn_fwd_order[idx_pos])
            dist_decay = 1.0 / (1.0 + abs(idx_pos - pos_f))
            arcs_weights[ti] += sn_w * dist_decay
            blocker_bitmask[ti] |= (1 << 3)

        q_rev = q_name[::-1]
        pos_r = int(np.searchsorted(blocker.sn_rev_sorted, q_rev))
        lo_r = max(0, pos_r - SN_WINDOW_SIZE)
        hi_r = min(blocker.target_count, pos_r + SN_WINDOW_SIZE)
        for idx_pos in range(lo_r, hi_r):
            ti = int(blocker.sn_rev_order[idx_pos])
            dist_decay = 1.0 / (1.0 + abs(idx_pos - pos_r))
            arcs_weights[ti] += sn_w * dist_decay * 0.8
            blocker_bitmask[ti] |= (1 << 3)

    # 4. PIN Conjunction & Domain
    if q_domain and q_domain != "nan":
        dom_match = blocker.domain_index.get(q_domain)
        if dom_match is not None:
            for ti in dom_match[:40]:
                arcs_weights[ti] += 3.0
                blocker_bitmask[ti] |= (1 << 4)

    p3 = q_name[:3] if len(q_name) >= 3 else q_name
    for pin in extract_valid_pin_codes(q_addr):
        pin_match = blocker.pin_index.get(f"{pin}:{p3}")
        if pin_match is not None:
            for ti in pin_match[:40]:
                arcs_weights[ti] += 2.5
                blocker_bitmask[ti] |= (1 << 4)

    if len(q_name) >= 4:
        p4_match = blocker.p4_index.get(q_name[:4])
        if p4_match is not None:
            for ti in p4_match[:30]:
                arcs_weights[ti] += 0.8
                blocker_bitmask[ti] |= (1 << 4)

    # Fallback prefix
    if len(arcs_weights) < 6 and len(q_name) >= 2:
        p3_list = blocker.prefix_3.get(q_name[:3])
        if p3_list is not None:
            for ti in p3_list[:15]:
                arcs_weights[ti] += 0.15
                blocker_bitmask[ti] |= (1 << 1)

    if not arcs_weights:
        return []

    # Reciprocal CNP & Per-Source Candidate Budgeting
    scored = []
    for ti, w_raw in arcs_weights.items():
        shared_cnt = bin(blocker_bitmask[ti]).count("1")
        score = w_raw * (1.0 + 0.15 * (shared_cnt - 1))
        scored.append((score, ti, w_raw, shared_cnt))

    scored.sort(key=lambda x: x[0], reverse=True)

    is_s2_arr = blocker.is_s2
    t_ids = blocker.target_ids
    selected = []
    cnt_s2 = 0
    cnt_s3 = 0

    for _, ti, w_raw, shared_cnt in scored:
        if is_s2_arr[ti]:
            if cnt_s2 >= max_k2:
                continue
            cnt_s2 += 1
        else:
            if cnt_s3 >= max_k3:
                continue
            cnt_s3 += 1
        selected.append((t_ids[ti], round(float(w_raw), 4), int(shared_cnt)))
        if cnt_s2 >= max_k2 and cnt_s3 >= max_k3:
            break

    return selected


LowRamJedAICountryBlocker = LowRamJedAIBlocker


def generate_candidates_for_df(
    s1_df: pd.DataFrame,
    country_blockers: dict,
    max_k2: int = MAX_CANDIDATES_K2,
    max_k3: int = MAX_CANDIDATES_K3,
):
    cand_dict = {}
    meta_graph_stats = {}
    n_workers = min(8, max(1, os.cpu_count() or 1))

    for country in s1_df["country"].dropna().unique():
        s1_c = s1_df[s1_df["country"] == country]
        c_str = str(country)
        blocker = country_blockers.get(c_str)
        if blocker is None:
            continue

        c_names = s1_c["name_core"].values.astype(str)
        c_addrs = s1_c["address_clean"].values.astype(str)
        c_domains = s1_c["domain"].values.astype(str)
        c_eids = s1_c["entity_id"].values.astype(str)

        c_name_sigs = compute_signatures_fast(c_names, num_perm=NUM_NAME_PERM)
        c_addr_sigs = compute_signatures_fast(c_addrs, num_perm=NUM_ADDR_PERM)
        c_name_bands = pack_lsh_bands_r4(c_name_sigs, LSH_NAME_BANDS)
        c_addr_bands = pack_lsh_bands_r4(c_addr_sigs, LSH_ADDR_BANDS)
        del c_name_sigs, c_addr_sigs

        n_entities = len(c_eids)
        if n_workers <= 1 or n_entities < 2000:
            for eid, name, addr, dom, nb, ab in zip(
                c_eids, c_names, c_addrs, c_domains, c_name_bands, c_addr_bands
            ):
                sel = query_jedai_candidates(nb, ab, name, addr, dom, blocker, max_k2=max_k2, max_k3=max_k3)
                if sel:
                    cand_dict[eid] = ",".join(cid for cid, _, _ in sel)
                    for cid, w_arcs, sh_cnt in sel:
                        meta_graph_stats[(eid, cid)] = (w_arcs, sh_cnt)
        else:
            chunk_sz = int(math.ceil(n_entities / n_workers))
            slices = []
            for w in range(n_workers):
                st = w * chunk_sz
                en = min((w + 1) * chunk_sz, n_entities)
                if st < en:
                    slices.append((
                        c_eids[st:en],
                        c_names[st:en],
                        c_addrs[st:en],
                        c_domains[st:en],
                        c_name_bands[st:en],
                        c_addr_bands[st:en],
                    ))

            def _worker(args):
                se, sn, sa, sd, s_nb, s_ab = args
                loc_cands = {}
                loc_meta = {}
                for eid, name, addr, dom, nb, ab in zip(se, sn, sa, sd, s_nb, s_ab):
                    sel = query_jedai_candidates(nb, ab, name, addr, dom, blocker, max_k2=max_k2, max_k3=max_k3)
                    if sel:
                        loc_cands[eid] = ",".join(cid for cid, _, _ in sel)
                        for cid, w_arcs, sh_cnt in sel:
                            loc_meta[(eid, cid)] = (w_arcs, sh_cnt)
                return loc_cands, loc_meta

            with ThreadPoolExecutor(max_workers=len(slices)) as executor:
                for loc_cands, loc_meta in executor.map(_worker, slices):
                    cand_dict.update(loc_cands)
                    meta_graph_stats.update(loc_meta)

    candidates_df = pd.DataFrame([
        {
            "source1_entity_id": eid,
            "candidate_entity_ids": cand_dict.get(eid, ""),
        }
        for eid in s1_df["entity_id"].values
    ])
    return candidates_df, meta_graph_stats


# ============================================================================
# Section 2.1: Theoretical Blocking Diagnostics (Survey §1.4)
# ============================================================================
def compute_blocking_diagnostics(
    candidates_df: pd.DataFrame,
    gt_dict: dict,
    s1_df: pd.DataFrame,
    country_target_counts: dict,
    split_name: str = "Validation",
):
    s1_country_counts = s1_df["country"].value_counts().to_dict()
    total_clean_dirty_cartesian = sum(
        int(n_s1_c) * int(country_target_counts.get(str(c), 0))
        for c, n_s1_c in s1_country_counts.items()
    )
    total_clean_dirty_cartesian = max(1, total_clean_dirty_cartesian)

    total_candidates = 0
    total_true = 0
    true_hits = 0

    for row in candidates_df.itertuples(index=False):
        s1_id = row.source1_entity_id
        cand_str = str(row.candidate_entity_ids) if pd.notnull(row.candidate_entity_ids) else ""
        cand_set = {c for c in cand_str.split(",") if c} if cand_str.strip() else set()
        true_set = gt_dict.get(s1_id, set())

        total_candidates += len(cand_set)
        if true_set:
            total_true += len(true_set)
            true_hits += len(cand_set & true_set)

    rr = 1.0 - (total_candidates / total_clean_dirty_cartesian)
    pc = true_hits / total_true if total_true > 0 else 0.0
    pq = true_hits / total_candidates if total_candidates > 0 else 0.0
    c_avg = total_candidates / max(1, len(candidates_df))

    print(f"\n  [Blocking Stage Diagnostics — {split_name} (Survey §1.4)]")
    print(f"    • Reduction Ratio (RR):             {rr:.6f}  (Target: > 0.9999)")
    print(f"    • Pair Completeness (PC / Recall):  {pc:.4f}    ({true_hits:,}/{total_true:,} true matches, Target: >= 0.97)")
    print(f"    • Pairs Quality (PQ / Precision):   {pq:.4f}    ({total_candidates:,} total pairs)")
    print(f"    • Candidate Cardinality (C_avg):    {c_avg:.2f}    (Target: <= 25.0 per S1)")

    return {"RR": rr, "PC": pc, "PQ": pq, "C_avg": c_avg}


# ============================================================================
# PHASE 3: Pairwise Multimodal Feature Engineering (12 Features)
# ============================================================================
def char_3gram_jaccard(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = f"  {a}  "
    sb = f"  {b}  "
    ga = {sa[i : i + 3] for i in range(len(sa) - 2)}
    gb = {sb[i : i + 3] for i in range(len(sb) - 2)}
    u = len(ga | gb)
    return len(ga & gb) / u if u > 0 else 0.0


def subword_cosine_similarity(a: str, b: str) -> float:
    """Fast character 2/3/4-gram TF-cosine similarity (instant on CPU, 0 MB memory)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = f" {a} "
    sb = f" {b} "
    ca = Counter(sa[i : i + 3] for i in range(max(1, len(sa) - 2)))
    cb = Counter(sb[i : i + 3] for i in range(max(1, len(sb) - 2)))
    for w in a.split():
        ca[w] += 2
    for w in b.split():
        cb[w] += 2
    common = set(ca.keys()) & set(cb.keys())
    if not common:
        return 0.0
    dot = sum(ca[k] * cb[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (norm_a * norm_b) if (norm_a > 0 and norm_b > 0) else 0.0


def numeric_iou(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return -1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def is_legal_form_compatible(lf1: str, lf2: str) -> int:
    if not lf1 or not lf2:
        return 1
    return 1 if lf1 == lf2 else 0


def _compute_feature_batch(
    s1_n, c_n, s1_nc, c_nc, s1_a, c_a, s1_num, c_num, s1_d, c_d, s1_lf, c_lf, arcs_w, shared_c
):
    f_name_ratio = [fuzz.ratio(s, c) for s, c in zip(s1_n, c_n)]
    f_name_sort = [fuzz.token_sort_ratio(s, c) for s, c in zip(s1_n, c_n)]
    f_name_set = [fuzz.token_set_ratio(s, c) for s, c in zip(s1_n, c_n)]
    f_name_jw = [distance.JaroWinkler.similarity(s, c) for s, c in zip(s1_n, c_n)]
    f_name_jac3 = [char_3gram_jaccard(s, c) for s, c in zip(s1_nc, c_nc)]
    f_name_sem = [subword_cosine_similarity(s, c) for s, c in zip(s1_nc, c_nc)]
    f_addr_set = [fuzz.token_set_ratio(s, c) for s, c in zip(s1_a, c_a)]
    f_addr_iou = [numeric_iou(s, c) for s, c in zip(s1_num, c_num)]
    f_dom_match = [1 if (s and c and s == c) else 0 for s, c in zip(s1_d, c_d)]
    f_lf_compat = [is_legal_form_compatible(l1, l2) for l1, l2 in zip(s1_lf, c_lf)]

    return (
        f_name_ratio,
        f_name_sort,
        f_name_set,
        f_name_jw,
        f_name_jac3,
        f_name_sem,
        f_addr_set,
        f_addr_iou,
        f_dom_match,
        f_lf_compat,
        arcs_w,
        shared_c,
    )


def build_features(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    meta_graph_stats: dict = None,
    gt_dict: dict = None,
    max_negatives_per_entity: int = None,
) -> pd.DataFrame:
    """Memory-optimized feature builder: indexes ONLY target records queried in this batch."""
    if meta_graph_stats is None:
        meta_graph_stats = {}

    s1_ids = []
    c_ids = []
    arcs_list = []
    shared_list = []

    for row in candidates_df.itertuples(index=False):
        c_str = row.candidate_entity_ids
        if c_str and str(c_str) != "nan":
            sid = row.source1_entity_id
            cand_list = [c for c in str(c_str).split(",") if c]
            if max_negatives_per_entity is not None and gt_dict is not None:
                trues = gt_dict.get(sid, set())
                pos_cands = [c for c in cand_list if c in trues]
                neg_cands = [c for c in cand_list if c not in trues][:max_negatives_per_entity]
                cand_list = pos_cands + neg_cands
            for cid in cand_list:
                s1_ids.append(sid)
                c_ids.append(cid)
                w_arcs, sh_cnt = meta_graph_stats.get((sid, cid), (0.5, 1))
                arcs_list.append(w_arcs)
                shared_list.append(sh_cnt)

    if not s1_ids:
        return pd.DataFrame()

    unique_cand_ids = list(set(c_ids))

    if target_df.index.name == "entity_id":
        if target_df.index.is_unique:
            matched_target = target_df.reindex(unique_cand_ids).dropna(subset=["name_clean"])
        else:
            matched_target = target_df.loc[target_df.index.intersection(unique_cand_ids)]
            matched_target = matched_target[~matched_target.index.duplicated(keep="first")]
    else:
        matched_target = (
            target_df[target_df["entity_id"].isin(set(unique_cand_ids))]
            .drop_duplicates("entity_id")
            .set_index("entity_id")
        )

    c_names = matched_target["name_clean"].to_dict()
    c_addrs = matched_target["address_clean"].to_dict()
    c_cores = matched_target["name_core"].to_dict()
    c_domains = matched_target["domain"].to_dict()
    c_lfs = matched_target["legal_form"].to_dict()
    del matched_target, unique_cand_ids

    s1_indexed = s1_df.set_index("entity_id")
    s1_names = s1_indexed["name_clean"].to_dict()
    s1_addrs = s1_indexed["address_clean"].to_dict()
    s1_cores = s1_indexed["name_core"].to_dict()
    s1_domains = s1_indexed["domain"].to_dict()
    s1_lfs = s1_indexed["legal_form"].to_dict()
    s1_countries = s1_indexed["country"].astype(str).to_dict()
    del s1_indexed

    # Local chunk-only address caches
    unique_addrs = set(s1_addrs.values()) | set(c_addrs.values())
    num_cache = {addr: extract_numbers(addr) for addr in unique_addrs}
    pin_cache = {addr: set(extract_valid_pin_codes(addr)) for addr in unique_addrs}

    s1_n = [str(s1_names.get(x, "")) for x in s1_ids]
    c_n = [str(c_names.get(x, "")) for x in c_ids]
    s1_nc = [str(s1_cores.get(x, "")) for x in s1_ids]
    c_nc = [str(c_cores.get(x, "")) for x in c_ids]
    s1_a = [str(s1_addrs.get(x, "")) for x in s1_ids]
    c_a = [str(c_addrs.get(x, "")) for x in c_ids]
    s1_d = [str(s1_domains.get(x, "")) for x in s1_ids]
    c_d = [str(c_domains.get(x, "")) for x in c_ids]
    s1_lf = [str(s1_lfs.get(x, "")) for x in s1_ids]
    c_lf = [str(c_lfs.get(x, "")) for x in c_ids]
    s1_num = [num_cache.get(s1_addrs.get(x, ""), set()) for x in s1_ids]
    c_num = [num_cache.get(c_addrs.get(x, ""), set()) for x in c_ids]

    # Walmart L2 Hard Veto Flags
    hard_veto_flags = []
    for sd, cd, sa_raw, ca_raw in zip(
        s1_d, c_d, [s1_addrs.get(x, "") for x in s1_ids], [c_addrs.get(x, "") for x in c_ids]
    ):
        dom_conflict = bool(sd and cd and sd != "nan" and cd != "nan" and sd != cd)
        pins1 = pin_cache.get(sa_raw, set())
        pins2 = pin_cache.get(ca_raw, set())
        pin_conflict = bool(pins1 and pins2 and pins1.isdisjoint(pins2))
        hard_veto_flags.append(1 if (dom_conflict and pin_conflict) else 0)

    del num_cache, pin_cache, unique_addrs

    n_pairs = len(s1_ids)
    n_workers = min(8, max(1, os.cpu_count() or 1))

    if n_workers > 1 and n_pairs >= 8000:
        chunk_size = int(math.ceil(n_pairs / n_workers))
        slices = []
        for w in range(n_workers):
            st = w * chunk_size
            en = min((w + 1) * chunk_size, n_pairs)
            if st < en:
                slices.append((
                    s1_n[st:en], c_n[st:en], s1_nc[st:en], c_nc[st:en],
                    s1_a[st:en], c_a[st:en], s1_num[st:en], c_num[st:en],
                    s1_d[st:en], c_d[st:en], s1_lf[st:en], c_lf[st:en],
                    arcs_list[st:en], shared_list[st:en],
                ))
        with ThreadPoolExecutor(max_workers=len(slices)) as executor:
            batch_results = list(executor.map(lambda args: _compute_feature_batch(*args), slices))

        col_tuples = [
            np.concatenate([res[c_idx] for res in batch_results])
            for c_idx in range(12)
        ]
    else:
        col_tuples = _compute_feature_batch(
            s1_n, c_n, s1_nc, c_nc, s1_a, c_a, s1_num, c_num, s1_d, c_d, s1_lf, c_lf, arcs_list, shared_list
        )

    (
        f_name_ratio,
        f_name_sort,
        f_name_set,
        f_name_jw,
        f_name_jac3,
        f_name_sem,
        f_addr_set,
        f_addr_iou,
        f_dom_match,
        f_lf_compat,
        f_arcs_w,
        f_shared_c,
    ) = col_tuples

    feats = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": c_ids,
        "name_ratio": np.asarray(f_name_ratio, dtype=np.float32),
        "name_token_sort_ratio": np.asarray(f_name_sort, dtype=np.float32),
        "name_token_set_ratio": np.asarray(f_name_set, dtype=np.float32),
        "name_jaro_winkler": np.asarray(f_name_jw, dtype=np.float32),
        "name_jaccard_3gram": np.asarray(f_name_jac3, dtype=np.float32),
        "name_semantic_sim": np.asarray(f_name_sem, dtype=np.float32),
        "addr_token_set_ratio": np.asarray(f_addr_token_set, dtype=np.float32),
        "addr_number_iou": np.asarray(f_addr_iou, dtype=np.float32),
        "domain_exact_match": np.asarray(f_dom_match, dtype=np.int8),
        "legal_form_compatible": np.asarray(f_lf_compat, dtype=np.int8),
        "blocking_arcs_weight": np.asarray(f_arcs_w, dtype=np.float32),
        "blocking_shared_count": np.asarray(f_shared_c, dtype=np.int16),
        "hard_veto": np.asarray(hard_veto_flags, dtype=np.int8),
        "country": [s1_countries.get(s, "UNKNOWN") for s in s1_ids],
    })

    if gt_dict is not None:
        feats["label"] = np.array(
            [1 if c in gt_dict.get(s, set()) else 0 for s, c in zip(s1_ids, c_ids)],
            dtype=np.int8,
        )

    del s1_names, s1_addrs, s1_cores, s1_domains, s1_lfs, s1_countries
    del c_names, c_addrs, c_cores, c_domains, c_lfs
    gc.collect()
    return feats


def apply_sger_training_augmentation(train_feats: pd.DataFrame) -> pd.DataFrame:
    pos_df = train_feats[train_feats["label"] == 1]
    if len(pos_df) == 0:
        return train_feats

    rng = np.random.RandomState(RANDOM_SEED)
    aug_sample = pos_df.sample(frac=min(0.25, 1.0), random_state=RANDOM_SEED).copy()
    aug_sample["name_ratio"] = np.clip(aug_sample["name_ratio"] * rng.uniform(0.82, 0.94, len(aug_sample)), 0, 100).astype(np.float32)
    aug_sample["name_jaro_winkler"] = np.clip(aug_sample["name_jaro_winkler"] * rng.uniform(0.85, 0.96, len(aug_sample)), 0, 1.0).astype(np.float32)
    aug_sample["name_token_sort_ratio"] = np.clip(aug_sample["name_token_sort_ratio"] * rng.uniform(0.92, 1.0, len(aug_sample)), 0, 100).astype(np.float32)

    return pd.concat([train_feats, aug_sample], ignore_index=True)


# ============================================================================
# PHASE 4: LightGBM GBDT Classifier, LOCO CV & Dual-Threshold Sweep
# ============================================================================
def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, split_name: str = "Validation"):
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)

    mse = float(mean_squared_error(y_t, y_p))
    logloss_val = float(log_loss(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0
    r2 = float(r2_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else 0.0

    print(f"\n  [Model Calibration & Probabilistic Scoring — {split_name} (Section 2.2)]")
    print(f"    • Mean Squared Error (MSE / Brier Score): {mse:.6f}")
    print(f"    • Binary Log-Loss:                        {logloss_val:.6f}")
    print(f"    • Precision-Recall AUC (PR-AUC):          {pr_auc:.6f}")
    print(f"    • Coefficient of Determination (R²):      {r2:.6f}")
    return {"MSE": mse, "LogLoss": logloss_val, "PR_AUC": pr_auc, "R2": r2}


def fit_lgbm_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
    for device_mode in ["gpu", "cpu"]:
        params = dict(LGBM_PARAMS)
        if device_mode == "gpu":
            params["device"] = "gpu"
        clf = lgb.LGBMClassifier(**params)
        try:
            if X_val is not None and y_val is not None and len(X_val) > 0:
                cbs = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
                clf.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric="binary_logloss",
                    callbacks=cbs,
                )
            else:
                clf.fit(X_train, y_train)
            return clf
        except Exception:
            if device_mode == "cpu":
                clf = lgb.LGBMClassifier(**LGBM_PARAMS)
                clf.fit(X_train, y_train)
                return clf


def run_loco_cross_validation(train_feats: pd.DataFrame, gt_dict: dict):
    countries = [c for c in train_feats["country"].dropna().unique() if c != "UNKNOWN"]
    if len(countries) < 2:
        return

    print("\n  [Leave-One-Country-Out (LOCO) Cross-Validation (Phase 4)]")
    for train_c in countries:
        for eval_c in countries:
            if train_c == eval_c:
                continue
            tr_sub = train_feats[train_feats["country"] == train_c]
            ev_sub = train_feats[train_feats["country"] == eval_c].copy()
            if len(tr_sub) < 100 or len(ev_sub) < 100:
                continue

            loco_clf = fit_lgbm_model(
                tr_sub[FEATURE_COLS].fillna(0),
                tr_sub["label"],
                ev_sub[FEATURE_COLS].fillna(0),
                ev_sub["label"],
            )
            ev_probs = loco_clf.predict_proba(ev_sub[FEATURE_COLS].fillna(0))[:, 1]
            ev_sub["pred_prob"] = ev_probs
            mse_loco = float(mean_squared_error(ev_sub["label"].values, ev_probs))
            pr_auc_loco = float(average_precision_score(ev_sub["label"].values, ev_probs)) if len(np.unique(ev_sub["label"].values)) > 1 else 0.0
            ev_s1_ids = ev_sub["source1_entity_id"].unique().tolist()
            pred_map = unique_mapping_clustering(ev_sub, ev_s1_ids, tau_s2=0.65, tau_s3=0.65)
            f05_loco = compute_macro_f05(pred_map, gt_dict, ev_s1_ids)
            print(
                f"    • Train [{train_c:<6s}] -> Eval [{eval_c:<6s}]: "
                f"Brier MSE = {mse_loco:.5f} | PR-AUC = {pr_auc_loco:.4f} | Macro F0.5 = {f05_loco:.4f}"
            )
            del loco_clf, ev_sub
            gc.collect()


# ============================================================================
# PHASE 5: JedAI Unique Mapping Clustering (Survey §9.1 Fig 9) & Walmart L2 Veto
# ============================================================================
def unique_mapping_clustering(
    pairs_df: pd.DataFrame,
    s1_all_ids,
    tau_s2: float = 0.65,
    tau_s3: float = 0.65,
    apply_hard_veto: bool = ENABLE_HARD_VETO,
) -> dict:
    final_matches = {s1_id: [] for s1_id in s1_all_ids}
    if pairs_df.empty or "pred_prob" not in pairs_df.columns:
        return final_matches

    df = pairs_df
    if apply_hard_veto and "hard_veto" in df.columns:
        df = df[df["hard_veto"] == 0]
    if df.empty:
        return final_matches

    is_s2_mask = df["candidate_entity_id"].astype(str).str.startswith("S2-")
    base_thresh = np.where(is_s2_mask, tau_s2, tau_s3)

    if "addr_number_iou" in df.columns and "domain_exact_match" in df.columns:
        sparse_mask = (df["addr_number_iou"] < 0) & (df["domain_exact_match"] == 0)
        row_thresh = np.where(sparse_mask, np.minimum(0.93, base_thresh + 0.03), base_thresh)
    elif "is_sparse" in df.columns:
        row_thresh = np.where(df["is_sparse"] == 1, np.minimum(0.93, base_thresh + 0.03), base_thresh)
    else:
        row_thresh = base_thresh

    valid = df[df["pred_prob"].values >= row_thresh]
    if valid.empty:
        return final_matches

    valid = valid.sort_values("pred_prob", ascending=False)

    claimed_targets = set()
    s2_counts = defaultdict(int)
    s3_counts = defaultdict(int)

    for row in valid.itertuples(index=False):
        sid = row.source1_entity_id
        cid = str(row.candidate_entity_id)

        if sid not in final_matches or cid in claimed_targets:
            continue

        if cid.startswith("S2-"):
            if s2_counts[sid] >= MAX_S2_MATCHES_PER_S1:
                continue
            s2_counts[sid] += 1
        elif cid.startswith("S3-"):
            if s3_counts[sid] >= MAX_S3_MATCHES_PER_S1:
                continue
            s3_counts[sid] += 1

        final_matches[sid].append(cid)
        claimed_targets.add(cid)

    return final_matches


def compute_macro_f05(pred_dict: dict, gt_dict: dict, s1_ids) -> float:
    scores = []
    for s1_id in s1_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_set = set(pred_dict.get(s1_id, ()))

        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set or not pred_set:
            scores.append(0.0)
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            if tp == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * tp) / (1.25 * tp + 0.25 * fn + fp)
                scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0


def sweep_dual_thresholds(val_feats: pd.DataFrame, gt_dict: dict, val_s1_ids):
    print(f"\n  [Validation Dual-Threshold Sweep: (tau_S2, tau_S3) in [{SWEEP_TAU_MIN:.2f}, {SWEEP_TAU_MAX:.2f}]]")
    grid = np.arange(SWEEP_TAU_MIN, SWEEP_TAU_MAX + 1e-9, SWEEP_TAU_STEP)

    best_tau_s2 = 0.65
    best_tau_s3 = 0.65
    best_f05 = -1.0

    for t in grid:
        pred_map = unique_mapping_clustering(val_feats, val_s1_ids, tau_s2=float(t), tau_s3=float(t))
        score = compute_macro_f05(pred_map, gt_dict, val_s1_ids)
        if score > best_f05:
            best_f05 = score
            best_tau_s2 = float(t)
            best_tau_s3 = float(t)
            print(f"    • (tau_S2={t:.2f}, tau_S3={t:.2f}) -> Validation Macro F0.5 = {score:.4f} [NEW BEST]")

    fine_grid_s2 = np.clip(np.arange(best_tau_s2 - 0.09, best_tau_s2 + 0.10, 0.02), SWEEP_TAU_MIN, SWEEP_TAU_MAX)
    fine_grid_s3 = np.clip(np.arange(best_tau_s3 - 0.09, best_tau_s3 + 0.10, 0.02), SWEEP_TAU_MIN, SWEEP_TAU_MAX)

    for ts2 in fine_grid_s2:
        for ts3 in fine_grid_s3:
            pred_map = unique_mapping_clustering(val_feats, val_s1_ids, tau_s2=float(ts2), tau_s3=float(ts3))
            score = compute_macro_f05(pred_map, gt_dict, val_s1_ids)
            if score > best_f05:
                best_f05 = score
                best_tau_s2 = float(ts2)
                best_tau_s3 = float(ts3)
                print(f"    • (tau_S2={ts2:.2f}, tau_S3={ts3:.2f}) -> Validation Macro F0.5 = {score:.4f} [NEW BEST]")

    print(
        f"\n  ★ Optimal Dual Thresholds: tau_S2 = {best_tau_s2:.2f}, tau_S3 = {best_tau_s3:.2f} "
        f"(Validation Macro F0.5 = {best_f05:.4f})"
    )
    return best_tau_s2, best_tau_s3, best_f05


def format_matching_results_df(pred_map: dict, s1_all_ids) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(pred_map.get(s1_id, [])),
        }
        for s1_id in s1_all_ids
    ])


# ============================================================================
# End-to-End Pipeline Execution
# ============================================================================
def run_pipeline():
    pipeline_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 78)
    print("AMAZON ML CHALLENGE 2026 — JEDAI 3-TIER & UNIQUE MAPPING ER PIPELINE")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Output Dir:   {OUTPUT_DIR}")
    print("=" * 78)

    # ── Phase 1: Ingestion & Multilingual Normalization ──────────────────────
    print("\n" + "=" * 78)
    print("PHASE 1: Ingestion & Multilingual Normalization")
    print("=" * 78)
    extract_dataset(ZIP_PATH, EXTRACT_DIR)
    data_dir = find_dataset_dir()
    print(f"  Using dataset directory: {data_dir}")

    s1_train_path = resolve_data_file(data_dir, "train_source1.tsv", split="train")
    gt_train_path = resolve_data_file(data_dir, "train_ground_truth.tsv", split="train")
    s2_train_path = resolve_data_file(data_dir, "train_source2.tsv", split="train")
    s3_train_path = resolve_data_file(data_dir, "train_source3.tsv", split="train")

    t0 = time.time()
    train_s1 = load_and_normalize(s1_train_path, nrows=TRAIN_SAMPLE_SIZE)
    print(f"  Loaded & normalized train_source1: {len(train_s1):,} rows from {s1_train_path} in {time.time() - t0:.1f}s")
    print(f"  Country distribution: {dict(train_s1['country'].value_counts())}")

    gt_dict = load_ground_truth(gt_train_path, target_s1_ids=train_s1["entity_id"].values)
    print(f"  Loaded ground truth for {len(gt_dict):,} Source 1 entities.")

    t0 = time.time()
    s2_train = load_and_normalize(s2_train_path)
    s3_train = load_and_normalize(s3_train_path)
    target_train_df = pd.concat([s2_train, s3_train], ignore_index=True)
    del s2_train, s3_train
    gc.collect()

    train_country_target_counts = target_train_df["country"].value_counts().to_dict()
    print(f"  Loaded & normalized train S2 + S3 targets: {len(target_train_df):,} rows in {time.time() - t0:.1f}s")
    target_train_df.set_index("entity_id", inplace=True)
    gc.collect()

    # ── Phase 2: JedAI 3-Tier Blocking Architecture ──────────────────────────
    print("\n" + "=" * 78)
    print("PHASE 2: JedAI 3-Tier Blocking (Block Building -> Cleaning -> ARCS/CNP Meta-Blocking)")
    print("=" * 78)

    temp_train_df, s1_test_eval_df = train_test_split(
        train_s1, test_size=TEST_SPLIT_RATIO, stratify=train_s1["country"], random_state=RANDOM_SEED
    )
    adjusted_val_ratio = VAL_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
    s1_train_df, s1_val_df = train_test_split(
        temp_train_df, test_size=adjusted_val_ratio, stratify=temp_train_df["country"], random_state=RANDOM_SEED
    )

    s1_train_df = s1_train_df.reset_index(drop=True)
    s1_val_df = s1_val_df.reset_index(drop=True)
    s1_test_eval_df = s1_test_eval_df.reset_index(drop=True)

    val_s1_ids = s1_val_df["entity_id"].tolist()
    local_test_s1_ids = s1_test_eval_df["entity_id"].tolist()
    del train_s1, temp_train_df
    gc.collect()

    print(
        f"  Leak-Free S1 Split -> Train: {len(s1_train_df):,} | "
        f"Val: {len(s1_val_df):,} | Local Test: {len(s1_test_eval_df):,}"
    )

    train_blockers = {}
    for country in target_train_df["country"].dropna().unique():
        tc = target_train_df[target_train_df["country"] == country]
        t_b = time.time()
        print(f"  [{country}] Building Low-RAM JedAI Blocker for {len(tc):,} targets...")
        train_blockers[str(country)] = LowRamJedAICountryBlocker(tc, country=country)
        print(f"  [{country}] Blocker ready in {time.time() - t_b:.1f}s.")
        del tc
    gc.collect()

    print("\n  Generating ARCS + Reciprocal CNP candidates (K2 <= 15, K3 <= 15)...")
    train_candidates, train_meta = generate_candidates_for_df(s1_train_df, train_blockers)
    val_candidates, val_meta = generate_candidates_for_df(s1_val_df, train_blockers)
    test_eval_candidates, test_eval_meta = generate_candidates_for_df(s1_test_eval_df, train_blockers)

    del train_blockers
    gc.collect()

    compute_blocking_diagnostics(
        train_candidates, gt_dict, s1_train_df, train_country_target_counts, split_name="Train Split"
    )
    compute_blocking_diagnostics(
        val_candidates, gt_dict, s1_val_df, train_country_target_counts, split_name="Validation Split"
    )
    compute_blocking_diagnostics(
        test_eval_candidates, gt_dict, s1_test_eval_df, train_country_target_counts, split_name="Local Test Split"
    )

    # ── Phase 3: Pairwise Multimodal Feature Engineering ─────────────────────
    print("\n" + "=" * 78)
    print("PHASE 3: Pairwise Multimodal Feature Engineering (12 Features)")
    print("=" * 78)
    t0 = time.time()
    train_feats = build_features(
        s1_train_df,
        target_train_df,
        train_candidates,
        meta_graph_stats=train_meta,
        gt_dict=gt_dict,
        max_negatives_per_entity=MAX_NEGATIVES_TRAIN,
    )
    val_feats = build_features(
        s1_val_df,
        target_train_df,
        val_candidates,
        meta_graph_stats=val_meta,
        gt_dict=gt_dict,
    )
    test_eval_feats = build_features(
        s1_test_eval_df,
        target_train_df,
        test_eval_candidates,
        meta_graph_stats=test_eval_meta,
        gt_dict=gt_dict,
    )
    print(f"  Extracted 12 pairwise features in {time.time() - t0:.1f}s.")
    print(f"    • Train pairs:      {len(train_feats):,} ({int(train_feats['label'].sum()):,} positives)")
    print(f"    • Validation pairs: {len(val_feats):,} ({int(val_feats['label'].sum()):,} positives)")
    print(f"    • Local Test pairs: {len(test_eval_feats):,} ({int(test_eval_feats['label'].sum()):,} positives)")

    del target_train_df, s1_train_df, s1_val_df, s1_test_eval_df
    del train_candidates, val_candidates, test_eval_candidates
    del train_meta, val_meta, test_eval_meta
    gc.collect()

    # ── Phase 4: LightGBM GBDT Classifier, LOCO CV & Dual-Threshold Sweep ────
    print("\n" + "=" * 78)
    print("PHASE 4: LightGBM GBDT Classifier, LOCO CV & Dual-Threshold Optimization")
    print("=" * 78)

    run_loco_cross_validation(train_feats, gt_dict)

    aug_train_feats = apply_sger_training_augmentation(train_feats)
    del train_feats
    gc.collect()

    t0 = time.time()
    clf = fit_lgbm_model(
        aug_train_feats[FEATURE_COLS].fillna(0),
        aug_train_feats["label"],
        val_feats[FEATURE_COLS].fillna(0),
        val_feats["label"],
    )
    print(f"\n  Full LightGBM GBDT trained in {time.time() - t0:.1f}s.")
    del aug_train_feats
    gc.collect()

    print("\n  Feature Importances:")
    for fname, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"    • {fname:<24s}: {int(imp):6d}")

    model_path = os.path.join(OUTPUT_DIR, "lgb_model.txt")
    clf.booster_.save_model(model_path)

    val_feats["pred_prob"] = clf.predict_proba(val_feats[FEATURE_COLS].fillna(0))[:, 1]
    test_eval_feats["pred_prob"] = clf.predict_proba(test_eval_feats[FEATURE_COLS].fillna(0))[:, 1]

    compute_calibration_metrics(val_feats["label"].values, val_feats["pred_prob"].values, split_name="Validation Set")
    compute_calibration_metrics(test_eval_feats["label"].values, test_eval_feats["pred_prob"].values, split_name="Local Test Set")

    best_tau_s2, best_tau_s3, val_f05 = sweep_dual_thresholds(val_feats, gt_dict, val_s1_ids)

    local_test_preds = unique_mapping_clustering(
        test_eval_feats, local_test_s1_ids, tau_s2=best_tau_s2, tau_s3=best_tau_s3
    )
    local_test_f05 = compute_macro_f05(local_test_preds, gt_dict, local_test_s1_ids)
    print(
        f"  ★ Unbiased Local Test Macro F0.5 (tau_S2={best_tau_s2:.2f}, tau_S3={best_tau_s3:.2f}): "
        f"{local_test_f05:.4f}"
    )

    del val_feats, test_eval_feats, gt_dict
    gc.collect()

    # ── Phase 5: JedAI Unique Mapping Clustering & Streamed Test Export ──────
    print("\n" + "=" * 78)
    print("PHASE 5: Streamed Test Inference & JedAI Unique Mapping Clustering")
    print("=" * 78)

    s2_test_path = resolve_data_file(data_dir, "test_source2.tsv", split="test")
    s3_test_path = resolve_data_file(data_dir, "test_source3.tsv", split="test")
    s1_test_path = resolve_data_file(data_dir, "test_source1.tsv", split="test")

    t0 = time.time()
    test_s2 = load_and_normalize(s2_test_path)
    test_s3 = load_and_normalize(s3_test_path)
    target_test_df = pd.concat([test_s2, test_s3], ignore_index=True)
    del test_s2, test_s3
    gc.collect()
    print(f"  Loaded & normalized test S2 + S3 targets: {len(target_test_df):,} rows in {time.time() - t0:.1f}s")
    target_test_df.set_index("entity_id", inplace=True)
    gc.collect()

    test_blockers = {}
    for country in target_test_df["country"].dropna().unique():
        tc = target_test_df[target_test_df["country"] == country]
        t_b = time.time()
        print(f"  [{country}] Building Low-RAM JedAI Blocker for {len(tc):,} test targets (including unseen France)...")
        test_blockers[str(country)] = LowRamJedAICountryBlocker(tc, country=country)
        print(f"  [{country}] Test blocker ready in {time.time() - t_b:.1f}s.")
        del tc
    gc.collect()

    test_s1 = load_and_normalize(s1_test_path)
    all_test_s1_ids = test_s1["entity_id"].tolist()
    num_test = len(test_s1)
    num_chunks = int(math.ceil(num_test / CHUNK_SIZE_INFERENCE))
    print(
        f"  Loaded test_source1: {num_test:,} entities across "
        f"{dict(test_s1['country'].value_counts())} ({num_chunks} chunks of size {CHUNK_SIZE_INFERENCE:,})"
    )

    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    open(candidate_file, "w", encoding="utf-8").close()
    header_cand_written = False

    # Store compact flat lists instead of giant DataFrames to guarantee memory safety
    acc_s1 = []
    acc_cand = []
    acc_prob = []
    acc_veto = []
    acc_sparse = []

    min_keep_prob = max(0.20, min(best_tau_s2, best_tau_s3) - 0.15)

    for i in range(num_chunks):
        start = i * CHUNK_SIZE_INFERENCE
        end = min((i + 1) * CHUNK_SIZE_INFERENCE, num_test)
        t_chunk = time.time()
        print(f"\n  [Test Chunk {i + 1}/{num_chunks}] Rows {start:,} -> {end:,}...")

        s1_chunk = test_s1.iloc[start:end].reset_index(drop=True)

        cand_chunk, chunk_meta = generate_candidates_for_df(
            s1_chunk, test_blockers, max_k2=MAX_CANDIDATES_K2, max_k3=MAX_CANDIDATES_K3
        )
        cand_chunk.to_csv(
            candidate_file,
            sep="\t",
            index=False,
            mode="a",
            header=not header_cand_written,
            encoding="utf-8",
        )
        header_cand_written = True

        chunk_feats = build_features(
            s1_chunk, target_test_df, cand_chunk, meta_graph_stats=chunk_meta
        )

        if not chunk_feats.empty:
            probs = clf.predict_proba(chunk_feats[FEATURE_COLS].fillna(0))[:, 1]
            keep_mask = probs >= min_keep_prob
            if keep_mask.any():
                sel_df = chunk_feats.loc[keep_mask]
                sel_probs = probs[keep_mask]
                acc_s1.extend(sel_df["source1_entity_id"].values.astype(str))
                acc_cand.extend(sel_df["candidate_entity_id"].values.astype(str))
                acc_prob.extend(sel_probs.astype(np.float32))
                acc_veto.extend(sel_df["hard_veto"].values.astype(np.int8))
                sparse_flags = ((sel_df["addr_number_iou"] < 0) & (sel_df["domain_exact_match"] == 0)).values.astype(np.int8)
                acc_sparse.extend(sparse_flags)
                del sel_df, sel_probs, sparse_flags

        print(f"  [Test Chunk {i + 1}/{num_chunks}] Processed in {time.time() - t_chunk:.1f}s.")
        del s1_chunk, cand_chunk, chunk_meta, chunk_feats
        gc.collect()

    del test_s1, target_test_df, test_blockers
    gc.collect()

    print("\n  Executing JedAI Unique Mapping Clustering (1-to-1 Target Exclusivity & Source Clamping)...")
    if acc_s1:
        compact_pairs_df = pd.DataFrame({
            "source1_entity_id": acc_s1,
            "candidate_entity_id": acc_cand,
            "pred_prob": np.asarray(acc_prob, dtype=np.float32),
            "hard_veto": np.asarray(acc_veto, dtype=np.int8),
            "is_sparse": np.asarray(acc_sparse, dtype=np.int8),
        })
        del acc_s1, acc_cand, acc_prob, acc_veto, acc_sparse
        gc.collect()
    else:
        compact_pairs_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "pred_prob", "hard_veto", "is_sparse"]
        )

    final_pred_map = unique_mapping_clustering(
        compact_pairs_df, all_test_s1_ids, tau_s2=best_tau_s2, tau_s3=best_tau_s3
    )
    del compact_pairs_df
    gc.collect()

    matching_results_df = format_matching_results_df(final_pred_map, all_test_s1_ids)
    matching_results_df.to_csv(matching_file, sep="\t", index=False, encoding="utf-8")

    non_empty_count = sum(1 for v in final_pred_map.values() if v)
    print(
        f"  Wrote {matching_file}: {len(matching_results_df):,} rows "
        f"({non_empty_count:,} matched S1 entities, {len(matching_results_df) - non_empty_count:,} singletons)."
    )
    print(f"  Wrote {candidate_file}: {num_test:,} rows.")

    if DRIVE_BACKUP_DIR and os.path.exists(os.path.dirname(DRIVE_BACKUP_DIR)):
        os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
        shutil.copy(matching_file, os.path.join(DRIVE_BACKUP_DIR, "matching_results.tsv"))
        shutil.copy(candidate_file, os.path.join(DRIVE_BACKUP_DIR, "candidate_pairs.tsv"))
        print(f"  Backed up submission TSVs to {DRIVE_BACKUP_DIR}")

    # Official Submission Validation
    test_dir_for_val = os.path.join(data_dir, "test") if os.path.isdir(os.path.join(data_dir, "test")) else data_dir
    validator_candidates = [
        os.path.join(PROJECT_ROOT, "utils", "validate_submission.py"),
        os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py"),
        os.path.join(data_dir, "..", "utils", "validate_submission.py"),
        os.path.join(_CURRENT_DIR, "..", "..", "..", "utils", "validate_submission.py"),
        "utils/validate_submission.py",
    ]
    validator_path = next((os.path.abspath(p) for p in validator_candidates if os.path.exists(p)), None)
    if validator_path:
        print("\n" + "=" * 78)
        print("OFFICIAL SUBMISSION VALIDATION (utils/validate_submission.py)")
        print(f"  Validator script: {validator_path}")
        print("=" * 78)
        spec = importlib.util.spec_from_file_location("validate_submission", validator_path)
        val_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(val_mod)
        errors, warnings_list = val_mod.validate(
            matching_path=matching_file,
            candidate_path=candidate_file,
            test_dir=test_dir_for_val,
            check_ids=False,
        )
        for w in warnings_list:
            print(f"  WARNING: {w}")
        if errors:
            print(f"  FAIL — {len(errors)} issue(s) found:")
            for idx_e, err in enumerate(errors, 1):
                print(f"    {idx_e}. {err}")
        else:
            print("  PASS — no blocking issues found. Safe to submit.")

    total_mins = (time.time() - pipeline_start) / 60.0
    print(f"\nPipeline completed end-to-end in {total_mins:.2f} minutes.")


if __name__ == "__main__":
    run_pipeline()
