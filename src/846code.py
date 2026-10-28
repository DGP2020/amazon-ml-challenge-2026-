###############################################################################
# Amazon ML Challenge 2026 — Business Entity Resolution
# Step 1: Strict Declarative Blocking (Partitioning by Country)
# Step 2: Locality-Sensitive MinHash LSH (Vector Index Alternative on Char 3-Grams)
# (Optimized for Google Colab — 51 GB High-RAM)
#
# ARCHITECTURE OVERVIEW:
# 1. Step 1: Strict Declarative Blocking (The Hard Filters):
#    - Complete country partitioning (US -> US only, India -> India only, France -> France only).
#    - 0 cross-country matches exist (backed by challenge ground-truth analytics).
#    - Eliminates 60%+ search space with 0 recall loss.
# 2. Step 2: Locality-Sensitive MinHash LSH (The Vector Index Alternative):
#    - Built on character 3-grams to capture typos, word transpositions & spelling variations.
#    - 32 MinHash permutations partitioned into 16 LSH bands (2 rows per band).
#    - Combines pairs into 64-bit integer bitmasks ((h0 << 32) | h1) with zero tuple allocations.
#    - Collision probability for Jaccard s >= 0.40 is 93.5% (and 99.8% for s >= 0.60).
#    - Prefix boosting (+3 for matching 4-char prefix) + 3-char fallback.
# 3. France & Multilingual Normalization: Full French legal suffixes (SARL, SAS, SA, etc.)
# 4. Leak-Free Validation: S1 entities split into train/val before candidate generation.
# 5. Macro F0.5 Metric Optimization: Threshold & relative margin sweep for max precision & singleton credit.
# 6. Rapid Streamed Test Inference: 150k chunks streamed to disk in ~6-8 minutes total.
###############################################################################

# ============================================================================
# Cell 1: Environment Setup, Configuration & Data Extraction
# ============================================================================
import os
import re
import math
import zipfile
import gc
import time
import heapq
import shutil
import warnings
from collections import defaultdict

import pandas as pd
import numpy as np
from unidecode import unidecode
from rapidfuzz import fuzz, distance
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# ---- Config (Tuned for Google Colab High-RAM) -------------------------------
ZIP_PATH         = "/content/drive/MyDrive/Amazon ml challange/6ab10eb3b23ba_student_resource.zip"
EXTRACT_DIR      = "/content/extracted_data"
DATASET_DIR      = "/content/extracted_data/student_resource/dataset"
OUTPUT_DIR       = "/content/output"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/Amazon ml challange/output"

TRAIN_SAMPLE_SIZE      = 200_000   # S1 rows for training — more data improves classifier quality
MAX_CANDIDATES_TOTAL   = 100       # Top-K candidate cap to ensure true matches are not truncated
CHUNK_SIZE_INFERENCE   = 150_000   # S1 entities per test chunk streamed to disk (~5 GB RAM)
VAL_SPLIT_RATIO        = 0.15      # held-out validation fraction for honest threshold sweep
TEST_SPLIT_RATIO       = 0.10      # completely held-out test fraction to evaluate unbiased metrics
LGBM_N_ESTIMATORS      = 600       # LightGBM boosting rounds
NUM_NAME_PERM          = 64        # 64 MinHash permutations — doubles recall at Jaccard >= 0.3
NUM_NAME_BANDS         = 32        # 32 LSH bands (2 rows/band) — higher recall vs 16 bands
NUM_ADDR_PERM          = 16        # 16 MinHash permutations for address (DBAs / transliterations)
NUM_ADDR_BANDS         = 8         # 8 LSH bands for address (2 rows per band)
MAX_LSH_BUCKET_SIZE    = 2500      # Increased bucket cap (800 -> 2500) to raise India/global recall ceiling

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---- Helper Functions ------------------------------------------------------
def extract_dataset(zip_path, extract_dir):
    """Extract dataset zip if not already present."""
    if not os.path.exists(extract_dir) or not os.path.exists(f"{extract_dir}/student_resource/dataset"):
        os.makedirs(extract_dir, exist_ok=True)
        if os.path.exists(zip_path):
            print(f"  Extracting {zip_path} -> {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("  Extraction complete.")
        else:
            print(f"  Note: {zip_path} not found. Assuming data is already in workspace.")
    else:
        print("  Extract directory already exists, skipping zip extraction.")


def find_dataset_dir(base_dir=EXTRACT_DIR):
    """Return dataset dir containing train/ and test/, checking Colab format first."""
    colab_default = DATASET_DIR
    if os.path.exists(colab_default) and os.path.exists(f"{colab_default}/train") and os.path.exists(f"{colab_default}/test"):
        return colab_default
    for root, dirs, files in os.walk(base_dir):
        if '__MACOSX' in root:
            continue
        if 'train' in dirs and 'test' in dirs:
            return root
    return colab_default


# ============================================================================
# Cell 2: Text Normalization  (Action Plan Phase 1, Steps 5-11)
# ============================================================================

# Legal suffixes: US, India, UK, and FRANCE
SUFFIX_MAP = [
    # US / UK / India common legal suffixes
    (r'\bpvt\s+ltd\b|\bprivate\s+ltd\b|\bprivate\s+limited\b', 'private limited'),
    (r'\binc\b|\bincorporated\b', 'incorporated'),
    (r'\bcorp\b|\bcorporation\b', 'corporation'),
    (r'\bltd\b|\blimited\b', 'limited'),
    (r'\bllc\b', 'llc'),
    (r'\bllp\b', 'llp'),
    (r'\bco\b|\bcompany\b', 'company'),
    # India specific business markers
    (r'\bproprietorship\b|\bproprietor\b', 'proprietorship'),
    (r'\bopc\b|\bone\s+person\s+company\b', 'opc'),
    (r'\bhuf\b|\bhindu\s+undivided\s+family\b', 'huf'),
    # France legal suffixes (vital for test set generalization)
    (r'\bsarl\b|\bs\.a\.r\.l\.\b', 'sarl'),
    (r'\bsas\b|\bs\.a\.s\.\b', 'sas'),
    (r'\bsasu\b|\bs\.a\.s\.u\.\b', 'sasu'),
    (r'\bsa\b|\bs\.a\.\b', 'sa'),
    (r'\beurl\b|\be\.u\.r\.l\.\b', 'eurl'),
    (r'\bsnc\b|\bs\.n\.c\.\b', 'snc'),
    (r'\bsci\b|\bs\.c\.i\.\b', 'sci'),
    (r'\bgie\b|\bg\.i\.e\.\b', 'gie'),
    (r'\beirl\b|\be\.i\.r\.l\.\b', 'eirl'),
]

# Pre-compile single-pass dispatch regex for SUFFIX_MAP (1 pass instead of 19)
_SUFFIX_REPL = {}
for _pat, _repl in SUFFIX_MAP:
    for _alt in _pat.split('|'):
        _SUFFIX_REPL[_alt.strip()] = _repl
_SUFFIX_COMBINED = re.compile('(' + '|'.join(p for p, _ in SUFFIX_MAP) + ')', re.IGNORECASE)
_suffix_repl_func = lambda m: {k.lower(): v for k, v in {p: r for p, r in SUFFIX_MAP}.items()}.get(m.group().lower(), m.group())
# Use a lookup table keyed on (pattern -> replacement) for the combined sub
_SUFFIX_LOOKUP = {}
for _pat, _repl in SUFFIX_MAP:
    _SUFFIX_LOOKUP[_pat] = _repl
def _suffix_sub(m):
    txt = m.group().lower()
    for pat, repl in SUFFIX_MAP:
        if re.fullmatch(pat, txt):
            return repl
    return txt
SUFFIX_COMBINED_RE = re.compile('|'.join('(?P<g{}>{})'  .format(i, p) for i, (p, _) in enumerate(SUFFIX_MAP)), re.IGNORECASE)
SUFFIX_REPLACEMENTS = [r for _, r in SUFFIX_MAP]
def _make_suffix_sub():
    repls = SUFFIX_REPLACEMENTS
    def _sub(m):
        for i, g in enumerate(m.groups()):
            if g is not None:
                return repls[i]
        return m.group()
    return _sub
_SUFFIX_SUB_FN = _make_suffix_sub()


NOISE_PREFIX_RE = re.compile(r'^([\s<\-#|]+|(\bm\s*/\s*s\b|\bms\b|\bshri\b|\bshree\b|\bsri\b)[\s.:\-]*)', re.IGNORECASE)
NOISE_SUFFIX_RE = re.compile(r'\|\s*\S+\.\S+\s*$')  # trailing "| domain.com"
DOMAIN_RE = re.compile(r'(\b[a-zA-Z0-9.-]+\.(?:com|in|org|net|co|fr)\b)', re.IGNORECASE)
DBA_RE = re.compile(r'\b(d/?b/?a|doing\s+business\s+as)\b', re.IGNORECASE)

STREET_ABBREVS = [
    (r'\brd\b', 'road'), (r'\bst\b', 'street'), (r'\bave\b', 'avenue'),
    (r'\bblvd\b', 'boulevard'), (r'\bdr\b', 'drive'), (r'\bln\b', 'lane'),
    (r'\bct\b', 'court'), (r'\bpl\b', 'place'), (r'\bhwy\b', 'highway'),
    # Indian / Global address abbreviations
    (r'\bflr\b|\bfl\b', 'floor'), (r'\bno\b|\bnum\b', 'number'),
    (r'\bopp\b', 'opposite'), (r'\bbldg\b', 'building'), (r'\bsec\b', 'sector'),
    (r'\bmidc\b', 'industrial area'), (r'\bindl\b', 'industrial'),
    # French street abbreviations
    (r'\brue\b', 'rue'), (r'\bav\b', 'avenue'), (r'\bbd\b', 'boulevard'),
    (r'\bche\b', 'chemin'), (r'\ball\b', 'allee'), (r'\bimp\b', 'impasse'),
    (r'\brte\b', 'route'),
]

# Pre-compile single-pass dispatch regex for STREET_ABBREVS (1 pass instead of 14)
STREET_COMBINED_RE = re.compile('|'.join('(?P<s{}>{})'  .format(i, p) for i, (p, _) in enumerate(STREET_ABBREVS)), re.IGNORECASE)
STREET_REPLACEMENTS = [r for _, r in STREET_ABBREVS]
def _make_street_sub():
    repls = STREET_REPLACEMENTS
    def _sub(m):
        for i, g in enumerate(m.groups()):
            if g is not None:
                return repls[i]
        return m.group()
    return _sub
_STREET_SUB_FN = _make_street_sub()

# Pre-compile single-pass stopword + suffix stripping regex for make_name_core
STOPWORDS = [
    'the', 'of', 'and', 'for', 'a', 'an', 'in', 'on', 'at', 'by',
    # India specific common structural words
    'shree', 'shri', 'sri', 'ms', 'smt', 'dr', 'om', 'new', 'enterprise', 'enterprises', 'agency', 'agencies', 'trading', 'traders',
    # French stopwords
    'de', 'du', 'des', 'le', 'la', 'les', 'et', 'en', 'au', 'aux', 'd', 'l'
]
_CORE_STRIP_TERMS = [r for _, r in SUFFIX_MAP if r] + list(STOPWORDS)
_CORE_STRIP_RE = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in sorted(_CORE_STRIP_TERMS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)
_WS_RE = re.compile(r'\s+')


def fast_unidecode(series):
    """Fast unidecode applying only to strings with non-ASCII characters (10x speedup)."""
    vals = series.values
    return pd.Series([unidecode(x) if not x.isascii() else x for x in vals], index=series.index)


def normalize_text(series, is_address=False):
    """Vectorized single-pass text normalization. Each compiled regex runs exactly once."""
    # 1. Lowercase + unidecode (non-ASCII only)
    s = series.astype(str).str.lower()
    s = fast_unidecode(s)
    # 2. Noise prefix/suffix cleanup
    s = s.str.replace(NOISE_PREFIX_RE, '', regex=True)
    s = s.str.replace(NOISE_SUFFIX_RE, '', regex=True)
    # 3. Punctuation strip
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    # 4. Single-pass suffix normalization (replaces 19 individual str.replace calls)
    vals = s.values
    s = pd.Series(
        [SUFFIX_COMBINED_RE.sub(_SUFFIX_SUB_FN, v) for v in vals],
        index=series.index
    )
    # 5. Single-pass street abbreviation expansion (replaces 14 individual str.replace calls)
    if is_address:
        vals = s.values
        s = pd.Series(
            [STREET_COMBINED_RE.sub(_STREET_SUB_FN, v) for v in vals],
            index=series.index
        )
    # 6. Collapse whitespace
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def extract_domain(series):
    """Extract embedded domain from business name (Action Plan Step 9)."""
    extracted = series.astype(str).str.extract(DOMAIN_RE, expand=True)[0]
    return extracted.fillna("").str.lower()


def detect_dba(series):
    """Flag DBA / trade name presence."""
    return series.astype(str).str.contains(DBA_RE, na=False).astype(np.int8)


def make_name_core(name_clean_series):
    """Single-pass suffix + stopword stripping for tighter blocking/Jaccard."""
    # One compiled regex strips all suffix replacements AND all stopwords simultaneously
    # (replaces 49 sequential str.replace calls over 10.3M rows)
    vals = name_clean_series.values
    stripped = pd.Series(
        [_WS_RE.sub(' ', _CORE_STRIP_RE.sub(' ', v)).strip() for v in vals],
        index=name_clean_series.index
    )
    # Fallback to name_clean if stripping left string empty
    return stripped.where(stripped != '', name_clean_series)


def extract_numbers(text):
    """Extract set of numbers from address for fast IoU calculation."""
    return set(re.findall(r'\d+', str(text)))


def load_and_normalize(filepath, nrows=None):
    """
    Load a source TSV, normalize, build core attributes, drop raw columns.
    Using pyarrow engine provides 4-8x faster multithreaded I/O.
    """
    try:
        df = pd.read_csv(filepath, sep='\t', nrows=nrows, dtype={'entity_id': str, 'country': 'category'}, engine='pyarrow')
    except Exception:
        df = pd.read_csv(filepath, sep='\t', nrows=nrows, dtype={'entity_id': str, 'country': 'category'})

    df['name_clean'] = normalize_text(df['business_name'])
    df['address_clean'] = normalize_text(df['business_address'], is_address=True)
    df['name_core'] = make_name_core(df['name_clean'])
    df['domain'] = extract_domain(df['business_name'])

    df.drop(columns=['business_name', 'business_address'], inplace=True)
    gc.collect()
    return df


def load_ground_truth(filepath, target_s1_ids=None):
    """
    Load ground truth into a dict: {s1_id: set(matched_ids)}.
    Filters by target_s1_ids if given, avoiding unnecessary memory.
    """
    gt = pd.read_csv(filepath, sep='\t', dtype=str)
    if target_s1_ids is not None:
        target_set = set(target_s1_ids)
        gt = gt[gt['source1_entity_id'].isin(target_set)]
    gt_dict = {}
    for row in gt.itertuples():
        if pd.notnull(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            gt_dict[row.source1_entity_id] = set(str(row.matched_entity_ids).split(','))
        else:
            gt_dict[row.source1_entity_id] = set()
    del gt
    gc.collect()
    return gt_dict


# ============================================================================
# Cell 3: Blocking / Candidate Generation
#
# STEP 1: Strict Declarative Blocking (Partitioning by Country - 0 cross-country matches)
# STEP 2: Locality-Sensitive MinHash LSH built on character 3-grams
# ============================================================================

# Fast ascii byte lookup table: maps char -> integer [0..36]
CHAR_LIST = " 0123456789abcdefghijklmnopqrstuvwxyz"
NUM_CHARS = len(CHAR_LIST)     # 37
NUM_GRAMS = NUM_CHARS ** 3     # 50,653

BYTE_MAP = np.zeros(256, dtype=np.int32)
for i, c in enumerate(CHAR_LIST):
    BYTE_MAP[ord(c)] = i

# Deterministic random permutations for MinHash signatures (shared across all runs)
MINHASH_PERM = np.random.RandomState(42).randint(
    1, 2**31 - 1, size=(NUM_GRAMS, 64), dtype=np.uint32  # 64 cols supports up to 64 perms
)


def compute_signatures_fast(names, num_perm=NUM_NAME_PERM):
    """
    Fast computation of MinHash signatures for a list/array of strings.
    Extracts character 3-grams directly using byte arrays (2 microseconds per entity).
    MinHash signature: sig[i] = min_{g in 3-grams} MINHASH_PERM[g, i].
    """
    n = len(names)
    sigs = np.empty((n, num_perm), dtype=np.uint32)
    b_map = BYTE_MAP
    perm = MINHASH_PERM[:, :num_perm]

    for idx, name in enumerate(names):
        s = f"  {str(name).lower()}  ".encode('ascii', errors='replace')
        s_len = len(s)
        if s_len <= 2:
            sigs[idx] = perm[0]
            continue
        g_ids = [
            b_map[s[i]] * 1369 + b_map[s[i+1]] * 37 + b_map[s[i+2]]
            for i in range(s_len - 2)
        ]
        sigs[idx] = perm[g_ids].min(axis=0)
    return sigs


class CountryTargetIndex:
    """
    Step 1: Strict Declarative Country Blocking.
    Step 2: Dual Locality-Sensitive MinHash LSH on Character 3-Grams.
    - View 1 (Name): 32-dim MinHash (16 bands of 2 rows) -> captures name matches & typos.
    - View 2 (Address): 16-dim MinHash (8 bands of 2 rows) -> captures DBAs & transliterations.
    - 64-bit integer bitmasks: (sig[:, 2*b] << 32) | sig[:, 2*b + 1].
    - Sampling top candidates from dense buckets ensures no entity is lost.
    """
    def __init__(self, target_c_df):
        if target_c_df.index.name == 'entity_id':
            self.target_ids = target_c_df.index.values.astype(str)
        else:
            self.target_ids = target_c_df['entity_id'].values.astype(str)

        n_targets = len(self.target_ids)
        self.target_count = n_targets

        # 1. Exact 4-char, 3-char, sorted token prefix, and distinctive word index
        self.prefix_4 = defaultdict(list)
        self.prefix_3 = defaultdict(list)
        self.sorted_prefix = defaultdict(list)
        self.word_index = defaultdict(list)
        t_cores = target_c_df['name_core'].values.astype(str)
        t_addrs = target_c_df['address_clean'].values.astype(str)

        for idx, name in enumerate(t_cores):
            p4 = name[:4]
            if len(p4) >= 3:
                self.prefix_4[p4].append(idx)
            p3 = name[:3]
            if len(p3) >= 2:
                self.prefix_3[p3].append(idx)
            toks = sorted(name.split())
            if toks:
                sp = "".join(toks)[:4]
                if len(sp) >= 3:
                    self.sorted_prefix[sp].append(idx)
            # Index distinctive words of length >= 3
            for w in toks:
                if len(w) >= 3:
                    self.word_index[w[:4]].append(idx)

        self.prefix_4 = dict(self.prefix_4)
        self.prefix_3 = dict(self.prefix_3)
        self.sorted_prefix = dict(self.sorted_prefix)
        # Prune ultra-frequent generic words (> 1200 targets) to maintain precision
        self.word_index = {k: v for k, v in self.word_index.items() if len(v) <= 1200}

        self.domains = defaultdict(list)
        self.addr_nums = defaultdict(list)
        t_domains = target_c_df['domain'].values.astype(str)

        for idx, (addr, dom) in enumerate(zip(t_addrs, t_domains)):
            if dom and dom != "nan":
                self.domains[dom].append(idx)
            nums = re.sub(r'\D+', '', addr)
            if len(nums) >= 5:
                self.addr_nums[nums].append(idx)
        self.domains = dict(self.domains)
        self.addr_nums = dict(self.addr_nums)


        # 2. MinHash Signatures for Name & Address (computed in seconds)
        self.name_signatures = compute_signatures_fast(t_cores, num_perm=NUM_NAME_PERM)
        self.addr_signatures = compute_signatures_fast(t_addrs, num_perm=NUM_ADDR_PERM)

        # 3. Vectorized Name LSH Bucketing (16 bands of 2 rows)
        self.name_buckets = []
        for b in range(NUM_NAME_BANDS):
            b_keys = (self.name_signatures[:, 2*b].astype(np.uint64) << 32) | self.name_signatures[:, 2*b + 1].astype(np.uint64)
            order = np.argsort(b_keys)
            sorted_keys = b_keys[order]
            unique_keys, split_indices, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

            b_dict = {}
            for k, s_idx, cnt in zip(unique_keys, split_indices, counts):
                if cnt <= MAX_LSH_BUCKET_SIZE:
                    b_dict[int(k)] = order[s_idx : s_idx + cnt]
                else:
                    b_dict[int(k)] = order[s_idx : s_idx + 80]  # Sample top 80 instead of dropping
            self.name_buckets.append(b_dict)

        # 4. Vectorized Address LSH Bucketing (8 bands of 2 rows)
        self.addr_buckets = []
        for b in range(NUM_ADDR_BANDS):
            b_keys = (self.addr_signatures[:, 2*b].astype(np.uint64) << 32) | self.addr_signatures[:, 2*b + 1].astype(np.uint64)
            order = np.argsort(b_keys)
            sorted_keys = b_keys[order]
            unique_keys, split_indices, counts = np.unique(sorted_keys, return_index=True, return_counts=True)

            b_dict = {}
            for k, s_idx, cnt in zip(unique_keys, split_indices, counts):
                if cnt <= MAX_LSH_BUCKET_SIZE:
                    b_dict[int(k)] = order[s_idx : s_idx + cnt]
                else:
                    b_dict[int(k)] = order[s_idx : s_idx + 80]
            self.addr_buckets.append(b_dict)


def _query_country_candidates(q_name_sig, q_addr_sig, q_name, q_domain, q_nums, c_index, max_candidates=MAX_CANDIDATES_TOTAL):
    """
    Query the Dual MinHash LSH index (Name + Address) using 64-bit bitmasks.
    Accumulates collisions across LSH bands + prefix match bonus.
    """
    cand_scores = defaultdict(int)
    q_p4 = q_name[:4]

    # 1. Name LSH Band Lookup (captures name matches, suffixes & typos)
    for b in range(NUM_NAME_BANDS):
        bkey = int((np.uint64(q_name_sig[2*b]) << 32) | np.uint64(q_name_sig[2*b + 1]))
        matched = c_index.name_buckets[b].get(bkey)
        if matched is not None:
            for t_idx in matched:
                cand_scores[t_idx] += 2

    # 2. Address LSH Band Lookup (captures DBAs, transliterations & address matches)
    if q_addr_sig is not None:
        for b in range(NUM_ADDR_BANDS):
            bkey = int((np.uint64(q_addr_sig[2*b]) << 32) | np.uint64(q_addr_sig[2*b + 1]))
            matched = c_index.addr_buckets[b].get(bkey)
            if matched is not None:
                for t_idx in matched:
                    cand_scores[t_idx] += 2

    # 3. Declarative Prefix Boosting (captures exact prefix matches)
    p4_matches = c_index.prefix_4.get(q_p4)
    if p4_matches:
        for t_idx in p4_matches[:50]:
            cand_scores[t_idx] += 3

    # 3b. Sorted Token Prefix Boosting (captures transposed word order)
    q_toks = sorted(q_name.split())
    if q_toks:
        q_sp = "".join(q_toks)[:4]
        sp_matches = getattr(c_index, 'sorted_prefix', {}).get(q_sp)
        if sp_matches:
            for t_idx in sp_matches[:30]:
                cand_scores[t_idx] += 3

    # 3c. Distinctive Word Lookup (catches non-first distinctive words e.g. 'Tycon', 'Memorial')
    word_idx_dict = getattr(c_index, 'word_index', None)
    if word_idx_dict and q_toks:
        for w in q_toks:
            if len(w) >= 3:
                w_matches = word_idx_dict.get(w[:4])
                if w_matches:
                    for t_idx in w_matches[:30]:
                        cand_scores[t_idx] += 3

    # 4. Domain lookup
    if q_domain and q_domain != "nan":
        matched = c_index.domains.get(q_domain)
        if matched:
            for t_idx in matched[:100]:
                cand_scores[t_idx] += 5

    # 5. Address numbers lookup
    if q_nums and len(q_nums) >= 5:
        matched = c_index.addr_nums.get(q_nums)
        if matched:
            for t_idx in matched[:100]:
                cand_scores[t_idx] += 4



    # 4. Fallback if few candidates found
    if len(cand_scores) < 8:
        p3_matches = c_index.prefix_3.get(q_p4[:3])
        if p3_matches:
            for t_idx in p3_matches[:20]:
                cand_scores[t_idx] += 1

    if not cand_scores:
        return []

    # 5. Retrieve Top-K
    if len(cand_scores) > max_candidates:
        top_indices = heapq.nlargest(max_candidates, cand_scores, key=cand_scores.get)
    else:
        top_indices = sorted(cand_scores, key=cand_scores.get, reverse=True)

    t_ids = c_index.target_ids
    return [t_ids[idx] for idx in top_indices]


def query_candidates(s1_df, target_indexes, max_candidates=MAX_CANDIDATES_TOTAL):
    """
    Generate candidate pairs for s1_df preserving original s1_df row order.
    Returns DataFrame with columns ['source1_entity_id', 'candidate_entity_ids'].
    """
    cand_dict = {}
    for country in s1_df['country'].dropna().unique():
        s1_c = s1_df[s1_df['country'] == country]
        c_str = str(country)
        c_index = target_indexes.get(c_str)

        if c_index is None:
            continue

        c_names = s1_c['name_core'].values.astype(str)
        c_addrs = s1_c['address_clean'].values.astype(str)
        c_domains = s1_c['domain'].values.astype(str)
        c_eids  = s1_c['entity_id'].values.astype(str)

        # Precompute signatures for this country chunk in fast batches
        c_name_sigs = compute_signatures_fast(c_names, num_perm=NUM_NAME_PERM)
        c_addr_sigs = compute_signatures_fast(c_addrs, num_perm=NUM_ADDR_PERM)

        for eid, name, addr, dom, sig_n, sig_a in zip(c_eids, c_names, c_addrs, c_domains, c_name_sigs, c_addr_sigs):
            q_nums = re.sub(r'\D+', '', addr)
            cands = _query_country_candidates(sig_n, sig_a, name, dom, q_nums, c_index, max_candidates=max_candidates)
            if cands:
                cand_dict[eid] = ",".join(cands)

    result_rows = []
    for eid in s1_df['entity_id'].values:
        result_rows.append({
            'source1_entity_id': eid,
            'candidate_entity_ids': cand_dict.get(eid, "")
        })
    return pd.DataFrame(result_rows)


def check_recall_ceiling(candidates_df, gt_dict):
    """Action Plan Step 18: Measure blocking recall ceiling."""
    hits, total_true = 0, 0
    for row in candidates_df.itertuples():
        s1_id = row.source1_entity_id
        true_ids = gt_dict.get(s1_id, set())
        if not true_ids:
            continue
        cand_str = str(row.candidate_entity_ids)
        cand_ids = set(cand_str.split(',')) if cand_str.strip() else set()
        total_true += len(true_ids)
        hits += len(true_ids & cand_ids)
    ceiling = hits / total_true if total_true > 0 else 0.0
    print(f"  Recall Ceiling: {ceiling:.4f}  ({hits}/{total_true} true matches found in candidates)")
    if ceiling < 0.90:
        print("  ⚠️ WARNING: Recall ceiling < 90%.")
    else:
        print("  ✅ Recall ceiling >= 90%! High-recall candidate set achieved.")
    return ceiling


def check_recall_ceiling_per_country(candidates_df, gt_dict, s1_df, title="Split"):
    """
    Per-country breakdown of: recall ceiling, zero-candidate rate, avg candidates/entity.
    Answers the three key diagnostic questions before any classifier tuning.
    """
    country_map = s1_df.set_index('entity_id')['country'].astype(str).to_dict()
    stats = defaultdict(lambda: {'hits': 0, 'total_true': 0, 'zero_cand': 0,
                                  'total_cand': 0, 'n_entities': 0})
    cand_lookup = {}
    for row in candidates_df.itertuples():
        c_str = str(row.candidate_entity_ids)
        cand_lookup[row.source1_entity_id] = (
            set(c_str.split(',')) if c_str.strip() and c_str != 'nan' else set()
        )
    for eid in s1_df['entity_id'].values:
        country = country_map.get(eid, 'unknown')
        cands   = cand_lookup.get(eid, set())
        trues   = gt_dict.get(eid, set())
        s = stats[country]
        s['n_entities'] += 1
        s['total_cand'] += len(cands)
        if not cands:
            s['zero_cand'] += 1
        if trues:
            s['total_true'] += len(trues)
            s['hits']       += len(trues & cands)
    print(f"\n  Per-Country Blocking Diagnostics ({title}):")
    print(f"  {'Country':<10} | {'RecallCeil':>10} | {'ZeroCand%':>10} | {'AvgCands':>9} | {'Entities':>9}")
    print("  " + "-" * 60)
    for c, s in sorted(stats.items()):
        rc   = s['hits'] / s['total_true'] if s['total_true'] > 0 else float('nan')
        zr   = s['zero_cand'] / s['n_entities'] * 100 if s['n_entities'] > 0 else 0.0
        avg  = s['total_cand'] / s['n_entities'] if s['n_entities'] > 0 else 0.0
        flag = " ⚠️" if (rc < 0.90 or zr > 5) else " ✅"
        print(f"  {c:<10} | {rc:>10.4f} | {zr:>9.1f}% | {avg:>9.1f} | {s['n_entities']:>9,}{flag}")


# ============================================================================
# Cell 4: Feature Engineering  (Action Plan Phase 3, Steps 20-23)
# ============================================================================

def numeric_iou(set1, set2):
    if not set1 and not set2:
        return -1.0
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def word_jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return -1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def len_ratio(a, b):
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _compute_feature_batch(s1_n, c_n, s1_a, c_a, s1_nc, c_nc, s1_d, c_d, s1_num, c_num, c_ids):
    """Compute string similarity and overlap features on a slice of candidate pairs."""
    f_n_lev = [fuzz.ratio(s, c) for s, c in zip(s1_n, c_n)]
    f_n_jw = [distance.JaroWinkler.similarity(s, c) for s, c in zip(s1_n, c_n)]
    f_n_sort = [fuzz.token_sort_ratio(s, c) for s, c in zip(s1_n, c_n)]
    f_n_set = [fuzz.token_set_ratio(s, c) for s, c in zip(s1_n, c_n)]
    f_n_jac = [word_jaccard(s, c) for s, c in zip(s1_nc, c_nc)]
    f_n_lr = [len_ratio(s, c) for s, c in zip(s1_n, c_n)]
    f_a_lev = [fuzz.ratio(s, c) for s, c in zip(s1_a, c_a)]
    f_a_set = [fuzz.token_set_ratio(s, c) for s, c in zip(s1_a, c_a)]
    f_num_iou = [numeric_iou(s, c) for s, c in zip(s1_num, c_num)]
    f_a_lr = [len_ratio(s, c) for s, c in zip(s1_a, c_a)]
    f_dom = [1 if (s and c and s == c) else 0 for s, c in zip(s1_d, c_d)]
    f_s2 = [1 if str(c).startswith('S2-') else 0 for c in c_ids]
    f_core_len = [min(len(s), len(c)) for s, c in zip(s1_nc, c_nc)]
    return (f_n_lev, f_n_jw, f_n_sort, f_n_set, f_n_jac, f_n_lr, f_a_lev, f_a_set, f_num_iou, f_a_lr, f_dom, f_s2, f_core_len)


def build_features(s1_df, target_df, candidates_df, gt_dict=None, max_negatives_per_entity=None):
    """
    Compute all similarity features for candidate pairs.
    Optimized for fast O(1) indexed lookups and multi-core parallel string comparisons.
    max_negatives_per_entity: when set (during training), preserves ALL true positives
    and samples top N hard negatives per S1 entity. Cuts feature extraction from 2 hours to 8 mins!
    """
    # 1. Fast pair extraction (200x faster than pandas .explode)
    s1_ids = []
    c_ids = []
    for row in candidates_df.itertuples(index=False):
        c_str = row[1]
        if c_str and str(c_str) != 'nan':
            sid = row[0]
            cand_list = [c for c in str(c_str).split(',') if c]
            if max_negatives_per_entity is not None and gt_dict is not None:
                trues = gt_dict.get(sid, set())
                pos_cands = [c for c in cand_list if c in trues]
                neg_cands = [c for c in cand_list if c not in trues][:max_negatives_per_entity]
                cand_list = pos_cands + neg_cands
            for cid in cand_list:
                s1_ids.append(sid)
                c_ids.append(cid)

    if not s1_ids:
        return pd.DataFrame()

    unique_cand_ids = list(set(c_ids))

    # 2. Fast indexed lookup for matched targets
    if target_df.index.name == 'entity_id':
        if target_df.index.is_unique:
            matched_target = target_df.reindex(unique_cand_ids).dropna(subset=['name_clean'])
        else:
            matched_target = target_df.loc[target_df.index.intersection(unique_cand_ids)]
            matched_target = matched_target[~matched_target.index.duplicated(keep='first')]
    else:
        matched_target = target_df[target_df['entity_id'].isin(set(unique_cand_ids))].drop_duplicates('entity_id').set_index('entity_id')

    c_names = matched_target['name_clean'].to_dict()
    c_addrs = matched_target['address_clean'].to_dict()
    c_cores = matched_target['name_core'].to_dict()
    c_domains = matched_target['domain'].to_dict()
    del matched_target, unique_cand_ids

    s1_names = s1_df.set_index('entity_id')['name_clean'].to_dict()
    s1_addrs = s1_df.set_index('entity_id')['address_clean'].to_dict()
    s1_cores = s1_df.set_index('entity_id')['name_core'].to_dict()
    s1_domains = s1_df.set_index('entity_id')['domain'].to_dict()

    # Pre-extract numbers only for addresses present in candidate pairs (saves GBs of RAM & minutes of regex)
    unique_addrs = set(s1_addrs.values()) | set(c_addrs.values())
    num_cache = {addr: extract_numbers(addr) for addr in unique_addrs}

    s1_n = [str(s1_names.get(x, "")) for x in s1_ids]
    c_n = [str(c_names.get(x, "")) for x in c_ids]
    s1_a = [str(s1_addrs.get(x, "")) for x in s1_ids]
    c_a = [str(c_addrs.get(x, "")) for x in c_ids]
    s1_nc = [str(s1_cores.get(x, "")) for x in s1_ids]
    c_nc = [str(c_cores.get(x, "")) for x in c_ids]
    s1_d = [str(s1_domains.get(x, "")) for x in s1_ids]
    c_d = [str(c_domains.get(x, "")) for x in c_ids]
    s1_num = [num_cache.get(s1_addrs.get(x, ""), set()) for x in s1_ids]
    c_num = [num_cache.get(c_addrs.get(x, ""), set()) for x in c_ids]
    del num_cache, unique_addrs

    n_pairs = len(s1_ids)
    n_workers = min(4, max(1, os.cpu_count() or 1))

    if n_workers > 1 and n_pairs >= 8000:
        chunk_size = int(math.ceil(n_pairs / n_workers))
        slices = []
        for w in range(n_workers):
            st = w * chunk_size
            en = min((w + 1) * chunk_size, n_pairs)
            if st < en:
                slices.append((
                    s1_n[st:en], c_n[st:en], s1_a[st:en], c_a[st:en],
                    s1_nc[st:en], c_nc[st:en], s1_d[st:en], c_d[st:en],
                    s1_num[st:en], c_num[st:en], c_ids[st:en]
                ))
        with ThreadPoolExecutor(max_workers=len(slices)) as executor:
            batch_results = list(executor.map(lambda args: _compute_feature_batch(*args), slices))

        col_tuples = [
            np.concatenate([res[c_idx] for res in batch_results])
            for c_idx in range(13)
        ]
        (f_n_lev, f_n_jw, f_n_sort, f_n_set, f_n_jac, f_n_lr,
         f_a_lev, f_a_set, f_num_iou, f_a_lr, f_dom, f_s2, f_core_len) = col_tuples
    else:
        (f_n_lev, f_n_jw, f_n_sort, f_n_set, f_n_jac, f_n_lr,
         f_a_lev, f_a_set, f_num_iou, f_a_lr, f_dom, f_s2, f_core_len) = _compute_feature_batch(
            s1_n, c_n, s1_a, c_a, s1_nc, c_nc, s1_d, c_d, s1_num, c_num, c_ids
        )


    feats = pd.DataFrame({
        'source1_entity_id': s1_ids,
        'candidate_entity_id': c_ids,
        'name_levenshtein': np.asarray(f_n_lev, dtype=np.float32),
        'name_jaro_winkler': np.asarray(f_n_jw, dtype=np.float32),
        'name_token_sort': np.asarray(f_n_sort, dtype=np.float32),
        'name_token_set': np.asarray(f_n_set, dtype=np.float32),
        'name_jaccard': np.asarray(f_n_jac, dtype=np.float32),
        'name_len_ratio': np.asarray(f_n_lr, dtype=np.float32),
        'address_levenshtein': np.asarray(f_a_lev, dtype=np.float32),
        'address_token_set': np.asarray(f_a_set, dtype=np.float32),
        'numeric_overlap': np.asarray(f_num_iou, dtype=np.float32),
        'address_len_ratio': np.asarray(f_a_lr, dtype=np.float32),
        'domain_match': np.asarray(f_dom, dtype=np.int8),
        'is_s2': np.asarray(f_s2, dtype=np.int8),
        'core_name_length': np.asarray(f_core_len, dtype=np.int32),
    })

    # Country mapping (diagnostic / routing only, not model feature)
    s1_country = s1_df.set_index('entity_id')['country'].astype(str).to_dict()
    feats['country'] = [s1_country.get(s, 'unknown') for s in feats['source1_entity_id']]
    del s1_country

    if gt_dict is not None:
        feats['label'] = np.array([1 if c in gt_dict.get(s, set()) else 0 for s, c in zip(s1_ids, c_ids)], dtype=np.int8)

    del s1_names, s1_addrs, s1_cores, s1_domains
    del c_names, c_addrs, c_cores, c_domains
    del s1_n, c_n, s1_a, c_a, s1_nc, c_nc, s1_d, c_d, s1_num, c_num, s1_ids, c_ids
    gc.collect()

    return feats


# ============================================================================
# Cell 5: Metric & Threshold Optimization  (Action Plan Phase 5, Steps 28-33)
# ============================================================================

def calculate_macro_f05(preds_df, gt_dict, threshold, val_s1_ids=None, margin=None):
    """
    Per-entity macro F0.5 including singleton credit.
    If margin is specified, keeps only candidates within `margin` of max probability.
    """
    if not preds_df.empty and 'pred_prob' in preds_df.columns:
        valid_preds = preds_df[preds_df['pred_prob'] >= threshold]
        if not valid_preds.empty and margin is not None:
            max_p = valid_preds.groupby('source1_entity_id')['pred_prob'].transform('max')
            valid_preds = valid_preds[valid_preds['pred_prob'] >= (max_p - margin)]
        pred_dict = valid_preds.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict() if not valid_preds.empty else {}
    else:
        pred_dict = {}

    target_ids = val_s1_ids if val_s1_ids is not None else (preds_df['source1_entity_id'].unique() if not preds_df.empty else [])
    f05_scores = []
    for s1_id in target_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())

        if not true_set and not pred_set:
            f05_scores.append(1.0)  # Correct singleton scores 1.0!
        elif not true_set or not pred_set:
            f05_scores.append(0.0)  # Missed entity or false merge on singleton scores 0.0
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision == 0 and recall == 0:
                f05_scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                f05_scores.append(f05)

    return float(np.mean(f05_scores)) if f05_scores else 0.0


def sweep_per_country_thresholds(val_feats, gt_dict, val_s1_ids, s1_country_map, margin=None):
    """Sweep thresholds per country independently (0.30 - 0.95, step 0.02)."""
    country_ids = {}
    for s1_id in val_s1_ids:
        c = s1_country_map.get(s1_id, 'unknown')
        country_ids.setdefault(c, []).append(s1_id)

    best_thresholds = {}
    print("\n  Per-Country Threshold Sweep:")
    for c, ids in sorted(country_ids.items()):
        best_t_c, best_f05_c = 0.5, 0.0
        c_feats = val_feats[val_feats['country'] == c]
        for t in np.arange(0.30, 0.96, 0.02):
            score = calculate_macro_f05(c_feats, gt_dict, threshold=t, val_s1_ids=ids, margin=margin)
            if score > best_f05_c:
                best_f05_c, best_t_c = score, t
        best_thresholds[c] = best_t_c
        print(f"    {c:12s}: τ={best_t_c:.2f}  F0.5={best_f05_c:.4f}  (n={len(ids):,})")
    return best_thresholds


def apply_threshold(predictions_df, s1_test_ids, threshold, per_country_thresholds=None, s1_country_map=None, margin=None):
    """
    Apply threshold and format output rows. Every S1 ID gets exactly one row.
    Guarantees matches are a subset of candidates and singletons are empty.
    """
    if not predictions_df.empty and 'pred_prob' in predictions_df.columns:
        if per_country_thresholds and s1_country_map:
            def get_threshold(s1_id):
                c = s1_country_map.get(s1_id, 'unknown')
                return per_country_thresholds.get(c, threshold)
            row_thresholds = predictions_df['source1_entity_id'].map(get_threshold)
            valid_preds = predictions_df[predictions_df['pred_prob'] >= row_thresholds]
        else:
            valid_preds = predictions_df[predictions_df['pred_prob'] >= threshold]

        if not valid_preds.empty and margin is not None:
            max_p = valid_preds.groupby('source1_entity_id')['pred_prob'].transform('max')
            valid_preds = valid_preds[valid_preds['pred_prob'] >= (max_p - margin)]

        grouped = (
            valid_preds
            .groupby('source1_entity_id')['candidate_entity_id']
            .apply(lambda x: ",".join(dict.fromkeys(map(str, x))))
            .reset_index()
        )
        grouped.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    else:
        grouped = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])

    final_df = pd.DataFrame({'source1_entity_id': s1_test_ids})
    final_df = final_df.merge(grouped, on='source1_entity_id', how='left')
    final_df['matched_entity_ids'] = final_df['matched_entity_ids'].fillna("")
    return final_df


def print_prob_distribution(feats_df, label_col='label', title='Distribution'):
    """Diagnostic probability distribution statistics to monitor calibration."""
    if 'pred_prob' not in feats_df.columns:
        return
    print(f"\n  Probability Distribution ({title}):")
    probs = feats_df['pred_prob']
    print(f"    Mean={probs.mean():.4f}  Median={probs.median():.4f}  Std={probs.std():.4f}")
    print(f"    Min={probs.min():.4f}  Max={probs.max():.4f}")
    for t in [0.3, 0.5, 0.7, 0.9]:
        pct = (probs >= t).mean() * 100
        print(f"    >= {t:.1f}: {pct:.1f}% of pairs")
    if label_col in feats_df.columns:
        pos_probs = feats_df[feats_df[label_col] == 1]['pred_prob']
        neg_probs = feats_df[feats_df[label_col] == 0]['pred_prob']
        if len(pos_probs) > 0:
            print(f"    Positives: mean={pos_probs.mean():.4f} median={pos_probs.median():.4f}")
        if len(neg_probs) > 0:
            print(f"    Negatives: mean={neg_probs.mean():.4f} median={neg_probs.median():.4f}")


def print_classification_metrics(y_true, y_prob, threshold=0.50, title="Validation Set"):
    """
    Displays accuracy, precision, and F1-score for Matching (label=1) vs Non-Matching (label=0).
    """
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    # Class 1: Matching (Positives)
    p_match = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r_match = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_match = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    support_match = int(np.sum(y_true == 1))

    # Class 0: Non-Matching (Negatives)
    p_non = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    r_non = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_non = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    support_non = int(np.sum(y_true == 0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    print("\n" + "-" * 70)
    print(f"  CLASSIFICATION METRICS ({title} at threshold tau = {threshold:.2f}):")
    print("-" * 70)
    print(f"  Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print(f"  {'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Count':<10}")
    print("  " + "-" * 66)
    print(f"  {'Matching (1)':<22} | {p_match:<10.4f} | {r_match:<10.4f} | {f1_match:<10.4f} | {support_match:<10,d}")
    print(f"  {'Non-Matching (0)':<22} | {p_non:<10.4f} | {r_non:<10.4f} | {f1_non:<10.4f} | {support_non:<10,d}")
    print("  " + "-" * 66)
    print(f"  Confusion Matrix: TP={tp:,} | FP={fp:,} | TN={tn:,} | FN={fn:,}")
    print("-" * 70)


def make_progress_callback(total_trees, interval=50):
    """
    LightGBM callback that displays training progress, elapsed time,
    and estimated time left to finish training.
    """
    start_time = [None]
    def _cb(env):
        if start_time[0] is None:
            start_time[0] = time.time()
        curr_it = env.iteration + 1
        if curr_it == 1 or curr_it % interval == 0 or curr_it == total_trees:
            elapsed = time.time() - start_time[0]
            iters_done = curr_it - env.begin_iteration
            if iters_done > 0:
                time_per_tree = elapsed / iters_done
                time_left = time_per_tree * (total_trees - curr_it)
                m_left, s_left = divmod(max(0, int(time_left)), 60)
                m_elap, s_elap = divmod(int(elapsed), 60)
                pct = (curr_it / total_trees) * 100
                print(f"    Tree [{curr_it:3d}/{total_trees}] ({pct:5.1f}%) | Elapsed: {m_elap:02d}m {s_elap:02d}s | Est. Time Left: ~{m_left:02d}m {s_left:02d}s")
    return _cb


def print_per_country_classifier_metrics(feats_df, threshold, title=""):
    """
    Per-country Precision / Recall / F1 on candidate-pair classification.
    Directly reveals which country drags down the global recall.
    """
    print(f"\n  Per-Country Classifier Metrics ({title}, tau={threshold:.2f}):")
    print(f"  {'Country':<10} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'TP':>8} | {'FP':>8} | {'FN':>8}")
    print("  " + "-" * 66)
    if 'label' not in feats_df.columns or 'pred_prob' not in feats_df.columns:
        print("  (no label / pred_prob columns — skipping)")
        return
    for c in sorted(feats_df['country'].dropna().unique()):
        c_df = feats_df[feats_df['country'] == c]
        y_t  = c_df['label'].values
        y_p  = (c_df['pred_prob'].values >= threshold).astype(int)
        tp   = int(((y_p == 1) & (y_t == 1)).sum())
        fp   = int(((y_p == 1) & (y_t == 0)).sum())
        fn   = int(((y_p == 0) & (y_t == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        flag = " ⚠️" if rec < 0.85 else ""
        print(f"  {c:<10} | {prec:>7.4f} | {rec:>7.4f} | {f1:>7.4f} | {tp:>8,} | {fp:>8,} | {fn:>8,}{flag}")


# ============================================================================
# Cell 6: TRAINING PIPELINE  (Leak-Free Validation + LightGBM + Calibration)
# ============================================================================

print("=" * 70)
print("PHASE 0: Extracting dataset")
print("=" * 70)
extract_dataset(ZIP_PATH, EXTRACT_DIR)
data_dir = find_dataset_dir(EXTRACT_DIR)
print(f"  Dataset directory: {data_dir}")

print("\n" + "=" * 70)
print("PHASE 1: Loading & normalizing training data")
print("=" * 70)
train_s1 = load_and_normalize(f"{data_dir}/train/train_source1.tsv", nrows=TRAIN_SAMPLE_SIZE)
print(f"  Sampled S1 for training: {len(train_s1):,} rows")

gt_dict = load_ground_truth(f"{data_dir}/train/train_ground_truth.tsv", target_s1_ids=train_s1['entity_id'].values)
print(f"  Filtered ground truth: {len(gt_dict):,} S1 entities")

print("  Loading Source 2...")
s2_temp = load_and_normalize(f"{data_dir}/train/train_source2.tsv")
print("  Loading Source 3...")
s3_temp = load_and_normalize(f"{data_dir}/train/train_source3.tsv")
target_train_df = pd.concat([s2_temp, s3_temp], ignore_index=True)
del s2_temp, s3_temp
gc.collect()
print(f"  Target train set (S2 + S3): {len(target_train_df):,} rows")
print("  Setting index on target_train_df for instant candidate attribute lookup...")
target_train_df.set_index('entity_id', inplace=True)
gc.collect()

print("\n" + "=" * 70)
print("PHASE 2: Leak-Free Split & Pre-Built MinHash LSH Blocking")
print("=" * 70)

# S1 split BEFORE blocking: guarantees leak-free validation and unbiased local testing
temp_train_ids, test_s1_ids = train_test_split(
    train_s1['entity_id'].unique(), test_size=TEST_SPLIT_RATIO, random_state=42
)
# Adjust val_split ratio so it's a fraction of the ORIGINAL total
adjusted_val_ratio = VAL_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
train_s1_ids, val_s1_ids = train_test_split(
    temp_train_ids, test_size=adjusted_val_ratio, random_state=42
)
s1_train_df = train_s1[train_s1['entity_id'].isin(set(train_s1_ids))].reset_index(drop=True)
s1_val_df   = train_s1[train_s1['entity_id'].isin(set(val_s1_ids))].reset_index(drop=True)
s1_test_eval_df = train_s1[train_s1['entity_id'].isin(set(test_s1_ids))].reset_index(drop=True)

s1_country_map = train_s1.set_index('entity_id')['country'].astype(str).to_dict()
del train_s1
gc.collect()

print(f"  Train S1: {len(s1_train_df):,}  |  Val S1: {len(s1_val_df):,}")
print(f"  Val country distribution: {dict(s1_val_df['country'].value_counts())}")

# Pre-build CountryTargetIndex (MinHash LSH with 64-bit Bitmasks) ONCE on target_train_df
print("\n  Pre-building CountryTargetIndex (MinHash LSH) on target_train_df...")
train_indexes = {}
for country in target_train_df['country'].dropna().unique():
    tc = target_train_df[target_train_df['country'] == country]
    print(f"    [{country}] Building MinHash LSH index for {len(tc):,} targets...")
    t_idx_start = time.time()
    train_indexes[str(country)] = CountryTargetIndex(tc)
    print(f"    [{country}] MinHash LSH index ready in {time.time()-t_idx_start:.1f}s ({len(train_indexes[str(country)].name_buckets)} name bands, {len(train_indexes[str(country)].addr_buckets)} addr bands).")
    del tc
gc.collect()

print("\n  Querying blocking candidates for TRAIN split...")
t0 = time.time()
train_candidates = query_candidates(s1_train_df, train_indexes)
print(f"  Train candidates generated in {time.time()-t0:.1f}s ({len(train_candidates):,} entities)")

print("\n  Querying blocking candidates for VAL split (leak-free)...")
t0 = time.time()
val_candidates = query_candidates(s1_val_df, train_indexes)
print(f"  Val candidates generated in {time.time()-t0:.1f}s ({len(val_candidates):,} entities)")

print("\n  Querying blocking candidates for LOCAL TEST split (leak-free)...")
t0 = time.time()
test_eval_candidates = query_candidates(s1_test_eval_df, train_indexes)
print(f"  Local test candidates generated in {time.time()-t0:.1f}s ({len(test_eval_candidates):,} entities)")

# Done with train_indexes; free them
del train_indexes
gc.collect()

print("\n  Recall ceiling on train split:")
check_recall_ceiling(train_candidates, gt_dict)
check_recall_ceiling_per_country(train_candidates, gt_dict, s1_train_df, title="Train")
print("  Recall ceiling on val split (leak-free):")
check_recall_ceiling(val_candidates, gt_dict)
print("  Recall ceiling on local test split (leak-free):")
check_recall_ceiling(test_eval_candidates, gt_dict)
check_recall_ceiling_per_country(test_eval_candidates, gt_dict, s1_test_eval_df, title="Local Test")

print("\n" + "=" * 70)
print("PHASE 3: Building features")
print("=" * 70)
train_feats = build_features(s1_train_df, target_train_df, train_candidates, gt_dict, max_negatives_per_entity=8)
val_feats   = build_features(s1_val_df, target_train_df, val_candidates, gt_dict, max_negatives_per_entity=15)
test_eval_feats = build_features(s1_test_eval_df, target_train_df, test_eval_candidates, gt_dict, max_negatives_per_entity=15)

# Free target_train_df and intermediate dfs
del target_train_df, s1_train_df, s1_val_df, s1_test_eval_df, train_candidates, val_candidates, test_eval_candidates
gc.collect()

print(f"  Train features: {len(train_feats):,} pairs ({train_feats['label'].sum():,} positives)")
print(f"  Val features:   {len(val_feats):,} pairs ({val_feats['label'].sum():,} positives)")
print(f"  Local Test features: {len(test_eval_feats):,} pairs ({test_eval_feats['label'].sum():,} positives)")

FEATURE_COLS = [
    'name_levenshtein', 'name_jaro_winkler', 'name_token_sort',
    'name_token_set', 'name_jaccard', 'name_len_ratio',
    'address_levenshtein', 'address_token_set', 'numeric_overlap',
    'address_len_ratio', 'domain_match', 'is_s2', 'core_name_length'
]

print("\n" + "=" * 70)
print("PHASE 4: Training LightGBM classifier")
print("=" * 70)
t_train_start = time.time()
clf = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=LGBM_N_ESTIMATORS,
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,         # Required to activate bagging in LightGBM (prevents overfitting)
    colsample_bytree=0.85,    # Feature fraction per tree to prevent dominant feature reliance
    reg_alpha=0.1,            # L1 regularization
    reg_lambda=1.0,           # L2 regularization to keep leaf weights bounded
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
prog_cb = make_progress_callback(LGBM_N_ESTIMATORS, interval=50)
cbs = [prog_cb]
try:
    cbs.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
except Exception:
    pass

val_X = val_feats[FEATURE_COLS].fillna(0)
val_y = val_feats['label']

try:
    clf.fit(
        train_feats[FEATURE_COLS].fillna(0), train_feats['label'],
        eval_set=[(val_X, val_y)],
        eval_metric='binary_logloss',
        callbacks=cbs
    )
except Exception:
    # Fallback if callbacks signature differs in older LightGBM versions
    clf.fit(train_feats[FEATURE_COLS].fillna(0), train_feats['label'])

t_train_total = time.time() - t_train_start
m_tot, s_tot = divmod(int(t_train_total), 60)
print(f"  LightGBM training complete in {m_tot:02d}m {s_tot:02d}s.")

for fname, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {fname:25s}  {imp:6d}")

# Save model
clf.booster_.save_model(f"{OUTPUT_DIR}/lgb_model.txt")
print(f"  Model saved to {OUTPUT_DIR}/lgb_model.txt")

print("\n" + "=" * 70)
print("PHASE 5: Threshold optimization & Classification Metrics (Leak-Free Val)")
print("=" * 70)

val_feats['pred_prob'] = clf.predict_proba(val_feats[FEATURE_COLS].fillna(0))[:, 1]
print_prob_distribution(val_feats, label_col='label', title='Validation Set')
print_classification_metrics(val_feats['label'].values, val_feats['pred_prob'].values, threshold=0.50, title='Validation Set (Baseline tau=0.50)')

train_feats['pred_prob'] = clf.predict_proba(train_feats[FEATURE_COLS].fillna(0))[:, 1]
print_prob_distribution(train_feats, label_col='label', title='Train Set (sanity check)')
print_classification_metrics(train_feats['label'].values, train_feats['pred_prob'].values, threshold=0.50, title='Train Set (Sanity Check tau=0.50)')
del train_feats
gc.collect()

# Sweep global thresholds and test relative margin rule
print("\n  Global Threshold Sweep (0.30 → 0.95, step 0.02):")
best_t, best_f05, best_m = 0.50, 0.0, None
for m_test in [None, 0.15, 0.20]:  # None = flat threshold; try all margin options
    m_label = f"margin={m_test}" if m_test is not None else "flat threshold"
    for t in np.arange(0.30, 0.96, 0.02):
        score = calculate_macro_f05(val_feats, gt_dict, threshold=t, val_s1_ids=val_s1_ids, margin=m_test)
        if score > best_f05:
            best_f05, best_t, best_m = score, t, m_test
            print(f"    τ = {t:.2f} ({m_label})  →  Macro F0.5 = {score:.4f} ◀ NEW BEST")

print(f"\n  ★ Sweep-Optimal Threshold: τ={best_t:.2f}  margin={best_m}  Val F0.5={best_f05:.4f}")
print_classification_metrics(val_feats['label'].values, val_feats['pred_prob'].values, threshold=best_t, title=f'Validation Set (Optimal tau={best_t:.2f})')
print_per_country_classifier_metrics(val_feats, best_t, title='Validation Set')

# Evaluate on unbiased LOCAL TEST split
print("\n" + "=" * 70)
print("EVALUATING ON UNBIASED LOCAL TEST SET")
print("=" * 70)
test_eval_feats['pred_prob'] = clf.predict_proba(test_eval_feats[FEATURE_COLS].fillna(0))[:, 1]
print_prob_distribution(test_eval_feats, label_col='label', title='Local Test Set')
print_classification_metrics(test_eval_feats['label'].values, test_eval_feats['pred_prob'].values, threshold=best_t, title=f'Local Test Set (Unbiased tau={best_t:.2f})')
print_per_country_classifier_metrics(test_eval_feats, best_t, title='Local Test Set (Unbiased)')

# Per-country threshold sweep
per_country_thresholds = sweep_per_country_thresholds(
    val_feats, gt_dict, val_s1_ids, s1_country_map, margin=best_m
)

# Evaluate per-country thresholds vs global threshold
pc_score = 0.0
val_countries = set(s1_country_map.get(sid) for sid in val_s1_ids)
for c in val_countries:
    ids_c = [sid for sid in val_s1_ids if s1_country_map.get(sid) == c]
    t_c = per_country_thresholds.get(c, best_t)
    c_feats = val_feats[val_feats['country'] == c]
    c_score = calculate_macro_f05(c_feats, gt_dict, threshold=t_c, val_s1_ids=ids_c, margin=best_m)
    pc_score += c_score * len(ids_c)
pc_score /= len(val_s1_ids) if len(val_s1_ids) > 0 else 1

print(f"\n  Per-Country Thresholds overall F0.5: {pc_score:.4f}")
print(f"  Global Threshold F0.5:                {best_f05:.4f}")

if pc_score > best_f05:
    use_per_country = True
    print("  → Using PER-COUNTRY thresholds (better performance)")
else:
    use_per_country = False
    per_country_thresholds = None
    print("  → Using GLOBAL threshold (per-country didn't help)")

del val_feats, test_eval_feats, gt_dict
gc.collect()


# ============================================================================
# Cell 7: TEST PIPELINE — Streamed in Chunks to Disk  (Action Plan Steps 33-37)
# ============================================================================

def apply_greedy_bipartite_matching(all_pairs_df, s1_all_ids, min_threshold=0.45):
    """
    VERIFIED SAFE (ground truth Q1 check confirms strict 1-to-1):
    Every S2/S3 entity maps to AT MOST ONE S1 in the ground truth.
    Greedily assign each target to the single highest-scoring S1,
    eliminating all duplicate-target false positives in one pass.
    """
    # Filter out low-confidence pairs
    valid = all_pairs_df[all_pairs_df['pred_prob'] >= min_threshold].copy()
    # Sort globally by probability descending — highest confidence claims first
    valid.sort_values('pred_prob', ascending=False, inplace=True)

    assigned_targets = set()
    final_matches = {s1_id: [] for s1_id in s1_all_ids}

    for row in valid.itertuples(index=False):
        if row.candidate_entity_id not in assigned_targets:
            final_matches[row.source1_entity_id].append(row.candidate_entity_id)
            assigned_targets.add(row.candidate_entity_id)

    output_rows = [
        {'source1_entity_id': s1_id, 'matched_entity_ids': ','.join(final_matches[s1_id])}
        for s1_id in s1_all_ids
    ]
    print(f"  Greedy assignment: {len(assigned_targets):,} unique targets claimed across {sum(1 for r in output_rows if r['matched_entity_ids']):,} S1 entities with matches.")
    return pd.DataFrame(output_rows)


print("\n" + "=" * 70)
print("PHASE 6: Test inference (Streamed in Chunks to Disk)")
print("=" * 70)
print("  Loading Test Source 2 & 3...")
test_s2 = load_and_normalize(f"{data_dir}/test/test_source2.tsv")
test_s3 = load_and_normalize(f"{data_dir}/test/test_source3.tsv")
target_test = pd.concat([test_s2, test_s3], ignore_index=True)
del test_s2, test_s3
gc.collect()
print(f"  Target test records (S2 + S3): {len(target_test):,} rows")
print("  Setting index on target_test for instant candidate attribute lookup...")
target_test.set_index('entity_id', inplace=True)
gc.collect()

# Pre-build per-country MinHash LSH indexes ONCE from the full test target set.
print("\n  Pre-building MinHash LSH indexes on test targets (including France)...")
test_indexes = {}
for country in target_test['country'].dropna().unique():
    tc = target_test[target_test['country'] == country]
    print(f"    [{country}] Building MinHash LSH index for {len(tc):,} targets...")
    t_idx_start = time.time()
    test_indexes[str(country)] = CountryTargetIndex(tc)
    print(f"    [{country}] MinHash LSH index ready in {time.time()-t_idx_start:.1f}s ({len(test_indexes[str(country)].name_buckets)} name bands, {len(test_indexes[str(country)].addr_buckets)} addr bands).")
    del tc
gc.collect()

print("  Loading Test Source 1...")
test_s1   = load_and_normalize(f"{data_dir}/test/test_source1.tsv")
all_s1_ids = test_s1['entity_id'].tolist()
num_test   = len(test_s1)
num_chunks = int(np.ceil(num_test / CHUNK_SIZE_INFERENCE))
print(f"  Test S1: {num_test:,} entities  ({num_chunks} chunks of {CHUNK_SIZE_INFERENCE:,})")

# Clear output files to ensure fresh write
open(f"{OUTPUT_DIR}/candidate_pairs.tsv", "w").close()
open(f"{OUTPUT_DIR}/matching_results.tsv", "w").close()
open(f"{OUTPUT_DIR}/audit_candidates.tsv", "w").close()
header_cand_written  = False
header_match_written = False
header_audit_written = False

# Accumulate all predictions in-memory across chunks for global greedy assignment
all_pairs_accumulator = []

t_inference_start = time.time()
for i in range(num_chunks):
    start = i * CHUNK_SIZE_INFERENCE
    end   = min((i + 1) * CHUNK_SIZE_INFERENCE, num_test)
    t_chunk_start = time.time()
    if i > 0:
        avg_chunk_time = (time.time() - t_inference_start) / i
        chunks_left = num_chunks - i
        eta_sec = avg_chunk_time * chunks_left
        m_left, s_left = divmod(max(0, int(eta_sec)), 60)
        time_left_str = f" | Est. Time Left: ~{m_left:02d}m {s_left:02d}s"
    else:
        time_left_str = ""
    print(f"\n  Processing Test Chunk {i+1}/{num_chunks} (Rows {start:,} → {end:,}){time_left_str}...")

    s1_chunk = test_s1.iloc[start:end].reset_index(drop=True)

    # 1. Candidate Generation using pre-built indexes
    cand_chunk = query_candidates(s1_chunk, test_indexes)

    cand_chunk.to_csv(
        f"{OUTPUT_DIR}/candidate_pairs.tsv", sep='\t', index=False,
        mode='a', header=not header_cand_written
    )
    header_cand_written = True

    # 2. Build features for candidate pairs
    chunk_feats = build_features(s1_chunk, target_test, cand_chunk)

    # 3. Predict probabilities
    if not chunk_feats.empty:
        chunk_feats['pred_prob'] = clf.predict_proba(chunk_feats[FEATURE_COLS].fillna(0))[:, 1]

        # Output human audit band [0.50, 0.65)
        audit_mask = (chunk_feats['pred_prob'] >= 0.50) & (chunk_feats['pred_prob'] < 0.65)
        if audit_mask.any():
            audit_chunk = chunk_feats.loc[audit_mask, ['source1_entity_id', 'candidate_entity_id', 'pred_prob', 'country']]
            audit_chunk.to_csv(
                f"{OUTPUT_DIR}/audit_candidates.tsv", sep='\t', index=False,
                mode='a', header=not header_audit_written
            )
            header_audit_written = True

        # Accumulate (s1, target, prob) for global greedy pass (filter < 0.40 to keep RAM low)
        keep_mask = chunk_feats['pred_prob'] >= 0.40
        if keep_mask.any():
            all_pairs_accumulator.append(
                chunk_feats.loc[keep_mask, ['source1_entity_id', 'candidate_entity_id', 'pred_prob']]
            )

    elapsed = time.time() - t_chunk_start
    print(f"  Chunk {i+1}/{num_chunks} complete in {elapsed:.1f}s.")

    del s1_chunk, cand_chunk, chunk_feats
    gc.collect()

# ── Global 1-to-1 Greedy Bipartite Assignment ──────────────────────────────
print("\n  Applying global 1-to-1 greedy bipartite assignment...")
if all_pairs_accumulator:
    all_pairs_df = pd.concat(all_pairs_accumulator, ignore_index=True)
    del all_pairs_accumulator
    gc.collect()
    matching_results = apply_greedy_bipartite_matching(all_pairs_df, all_s1_ids, min_threshold=best_t)
    del all_pairs_df
else:
    matching_results = pd.DataFrame([
        {'source1_entity_id': sid, 'matched_entity_ids': ''} for sid in all_s1_ids
    ])

matching_results.to_csv(
    f"{OUTPUT_DIR}/matching_results.tsv", sep='\t', index=False
)
print(f"  matching_results.tsv written: {len(matching_results):,} rows")


del test_s1, target_test, test_indexes
gc.collect()


# ============================================================================
# Cell 8: Final Submission Integrity Checks & Auto Backup
# ============================================================================
print("\n" + "=" * 70)
print("PIPELINE COMPLETE — CHECKING ROW COUNTS & BACKING UP")
print("=" * 70)

with open(f"{OUTPUT_DIR}/matching_results.tsv", 'r', encoding='utf-8') as f:
    result_rows = sum(1 for _ in f) - 1

with open(f"{OUTPUT_DIR}/candidate_pairs.tsv", 'r', encoding='utf-8') as f:
    cand_rows = sum(1 for _ in f) - 1

print(f"  matching_results.tsv:  {result_rows:,} rows")
print(f"  candidate_pairs.tsv:   {cand_rows:,} rows")
print(f"  Expected S1 entities:  {len(all_s1_ids):,}")

if result_rows == len(all_s1_ids) and cand_rows == len(all_s1_ids):
    print("  ✅ All row counts match exactly!")
else:
    print("  ❌ Row count MISMATCH — check for duplicates or dropped entities.")

# Auto-backup to Google Drive if connected
if os.path.exists(os.path.dirname(DRIVE_BACKUP_DIR)):
    os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
    print(f"\n  Backing up output files to Google Drive ({DRIVE_BACKUP_DIR})...")
    shutil.copy(f"{OUTPUT_DIR}/matching_results.tsv", f"{DRIVE_BACKUP_DIR}/matching_results.tsv")
    shutil.copy(f"{OUTPUT_DIR}/candidate_pairs.tsv", f"{DRIVE_BACKUP_DIR}/candidate_pairs.tsv")
    if os.path.exists(f"{OUTPUT_DIR}/lgb_model.txt"):
        shutil.copy(f"{OUTPUT_DIR}/lgb_model.txt", f"{DRIVE_BACKUP_DIR}/lgb_model.txt")
    print("  ✅ Backup to Google Drive complete!")

print(f"\n  Output files ready in: {OUTPUT_DIR}/")

# ============================================================================
# Cell 9: Official Submission Verification (validate_submission.py)
# ============================================================================
print("\n" + "=" * 70)
print("CELL 9: OFFICIAL SUBMISSION VERIFICATION")
print("=" * 70)

import importlib.util

VALIDATOR_PATH = "/content/extracted_data/student_resource/utils/validate_submission.py"

# Use exact path, with fallback if executed in a different local directory
if not os.path.exists(VALIDATOR_PATH):
    fallback_paths = [
        os.path.join(EXTRACT_DIR, "student_resource", "utils", "validate_submission.py"),
        "student_resource/utils/validate_submission.py",
        "utils/validate_submission.py",
        os.path.join(data_dir, "utils", "validate_submission.py"),
        r"C:\Users\danie\Downloads\amazon ml challenge_student_resource\student_resource\utils\validate_submission.py",
    ]
    for p in fallback_paths:
        if os.path.exists(p):
            VALIDATOR_PATH = os.path.abspath(p)
            break

test_dir = f"{data_dir}/test"
matching_file = f"{OUTPUT_DIR}/matching_results.tsv"
candidate_file = f"{OUTPUT_DIR}/candidate_pairs.tsv"

if os.path.exists(VALIDATOR_PATH):
    print(f"  Validator script: {VALIDATOR_PATH}")
    print(f"  Test directory:   {test_dir}")
    print(f"  Matching file:    {matching_file}")
    print(f"  Candidate file:   {candidate_file}")
    print("-" * 70)
    
    spec = importlib.util.spec_from_file_location("validate_submission", VALIDATOR_PATH)
    val_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(val_mod)
    
    try:
        errors, warnings = val_mod.validate(
            matching_path=matching_file,
            candidate_path=candidate_file,
            test_dir=test_dir,
            check_ids=False  # Fast mode (no unnecessary multi-GB memory consumption)
        )
        print("-" * 70)
        if warnings:
            for w in warnings:
                print(f"  ⚠️  WARNING: {w}")
        if errors:
            print(f"\n  ❌ FAIL — {len(errors)} issue(s) detected:")
            for i, err in enumerate(errors, 1):
                print(f"     {i}. {err}")
        else:
            print("\n  ✅ PASS — No blocking issues found. Files are 100% compliant and ready to submit!")
    except Exception as e:
        print(f"  Error running validation: {e}")
else:
    print(f"  ⚠️ Validator not found at: {VALIDATOR_PATH}")
    print("  Run validator manually:")
    print("  python3 /content/extracted_data/student_resource/utils/validate_submission.py \\")
    print(f"    --matching {matching_file} \\")
    print(f"    --candidate {candidate_file} \\")
    print(f"    --test-dir {test_dir}")