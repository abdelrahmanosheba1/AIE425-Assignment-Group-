# ============================
# CASE STUDY 2 (SAFE)
# Mean-Centered Cosine User-Based CF + DF + DS
# ============================

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import math, os, json, time

# ---- Setup output directory with correct relative path ----
output_dir = os.path.join("..", "..", "results(output_files,visualizations)", "part1_user_based_cf_results")
os.makedirs(output_dir, exist_ok=True)

# ---- Load dataset safely ----
data_dir = os.path.join("..", "..", "results(output_files,visualizations)")
dataset_paths = [
    os.path.join(data_dir, "ratings_expanded.csv"),
    os.path.join(data_dir, "ratings_cleaned.csv"),
    os.path.join(data_dir, "ratings.csv")
]

df = None
for path in dataset_paths:
    try:
        df = pd.read_csv(path)
        print(f"Loaded {path} | rows: {len(df)}")
        break
    except FileNotFoundError:
        continue

if df is None:
    raise FileNotFoundError(f"No ratings CSV found in {data_dir}/. Run preprocessing first.")

# ---- aggregate duplicates (user,item) by mean rating ----
df = df.groupby(["user_id","item_id"], as_index=False)["rating"].mean()

# ---- factorize ids to integer indices ----
user_codes, users = pd.factorize(df["user_id"])
item_codes, items = pd.factorize(df["item_id"])
n_users, n_items = len(users), len(items)

# ---- build sparse user-item matrix ----
R = csr_matrix((df["rating"].values, (user_codes, item_codes)), shape=(n_users, n_items))

# ---- helper maps ----
idx_to_user = {i:users[i] for i in range(n_users)}
idx_to_item = {j:items[j] for j in range(n_items)}

# ---- select target users (same as Case 1) ----
counts_per_user = np.array((R!=0).sum(axis=1)).reshape(-1)
total_ratings = R.nnz
user_pct = counts_per_user / total_ratings * 100

def pick_user_in_range(low_pct, high_pct):
    cand = np.where((user_pct >= low_pct) & (user_pct < high_pct))[0]
    if len(cand)>0:
        return cand[np.argmax(counts_per_user[cand])]
    center = (low_pct + high_pct)/2
    return int(np.argmin(np.abs(user_pct - center)))

target_user_idxs = [pick_user_in_range(0,2), pick_user_in_range(2,5), pick_user_in_range(5,10)]

# ---------- per-target processing ----------
def process_target_mean_centered(target_idx, overlap_pct=0.30, top_percent=0.20, max_candidates=200000):
    t0 = time.time()
    t_items = set(R[target_idx].nonzero()[1].tolist())
    n_t_items = len(t_items)
    req_overlap = math.ceil(overlap_pct * max(1, n_t_items))

    # ---- mean-center each user ----
    means = np.zeros(n_users)
    for u in range(n_users):
        r = R[u].data
        if len(r)>0:
            means[u] = r.mean()

    R_centered = R.copy().astype(float)
    for u in range(n_users):
        if R_centered[u].nnz>0:
            R_centered[u].data -= means[u]

    # ---- compute cosine similarity vector (one-to-many) ----
    sims = cosine_similarity(R_centered.getrow(target_idx), R_centered).flatten()
    sims[target_idx] = -1.0

    # ---- top 20% neighbors ----
    k = max(1, int(math.ceil(top_percent * n_users)))
    topk_raw = np.argpartition(-sims, k)[:k]
    topk_raw = topk_raw[np.argsort(-sims[topk_raw])]
    topk_raw = np.array([u for u in topk_raw if sims[u] > 0], dtype=int)
    k_raw = len(topk_raw)

    # ---- predict ratings for candidate items ----
    neigh_item_rows = R[topk_raw]
    neigh_items = np.unique(neigh_item_rows.nonzero()[1])
    candidate_items = np.setdiff1d(neigh_items, list(t_items), assume_unique=True)
    if len(candidate_items) > max_candidates: candidate_items = candidate_items[:max_candidates]

    preds_before = {}
    for it in candidate_items:
        neighbor_ratings = neigh_item_rows[:, it].toarray().flatten()
        mask = neighbor_ratings != 0
        if mask.sum() == 0: continue
        weights = sims[topk_raw][mask]
        ratings = neighbor_ratings[mask]
        denom = np.sum(np.abs(weights))
        if denom == 0: continue
        preds_before[int(it)] = float(np.dot(weights, ratings)/denom)

    # ---- compute DF and DS ----
    if len(t_items) == 0:
        overlaps = np.zeros(n_users, dtype=int)
    else:
        R_target_cols = R[:, list(t_items)].tocsr()
        overlaps = np.array(R_target_cols.getnnz(axis=1)).reshape(-1)
    dfactors = np.divide(overlaps, req_overlap, out=np.zeros_like(overlaps,dtype=float), where=req_overlap>0)
    dfactors = np.clip(dfactors, 0.0, 1.0)
    ds = sims * dfactors
    ds[target_idx] = -1.0

    # ---- top 20% DS neighbors ----
    topk_ds = np.argpartition(-ds, k)[:k]
    topk_ds = topk_ds[np.argsort(-ds[topk_ds])]
    topk_ds = np.array([u for u in topk_ds if ds[u]>0], dtype=int)

    # ---- predict using DS neighbors ----
    neigh_item_rows_ds = R[topk_ds] if len(topk_ds)>0 else None
    candidate_items_ds = np.unique(neigh_item_rows_ds.nonzero()[1]) if neigh_item_rows_ds is not None else np.array([],dtype=int)
    candidate_items_ds = np.setdiff1d(candidate_items_ds,list(t_items), assume_unique=True)
    if len(candidate_items_ds) > max_candidates: candidate_items_ds = candidate_items_ds[:max_candidates]

    preds_after = {}
    for it in candidate_items_ds:
        neighbor_ratings = neigh_item_rows_ds[:, it].toarray().flatten() if len(topk_ds)>0 else np.array([])
        mask = neighbor_ratings != 0
        if mask.sum() == 0: continue
        weights = ds[topk_ds][mask]
        ratings = neighbor_ratings[mask]
        denom = np.sum(np.abs(weights))
        if denom == 0: continue
        preds_after[int(it)] = float(np.dot(weights, ratings)/denom)

    # ---- summary ----
    set_before, set_after = set(topk_raw.tolist()), set(topk_ds.tolist())
    neighbor_overlap = len(set_before & set_after)
    common_items = set(preds_before.keys()) & set(preds_after.keys())
    mad = float(np.mean([abs(preds_before[it]-preds_after[it]) for it in common_items])) if len(common_items)>0 else None

    summary = {
        "target_idx": int(target_idx),
        "target_user_id": idx_to_user[target_idx],
        "num_neighbors_raw": int(k_raw),
        "num_neighbors_ds": int(len(topk_ds)),
        "neighbor_overlap_count": int(neighbor_overlap),
        "num_candidates_before": int(len(preds_before)),
        "num_candidates_after": int(len(preds_after)),
        "mad_on_common_predictions": mad,
        "time_seconds": round(time.time()-t0,2)
    }

    return summary

# ---- Run Case 2 for targets ----
case2_summaries = []
for t in target_user_idxs:
    print("\nProcessing Case 2 target:", idx_to_user[t])
    s = process_target_mean_centered(t)
    case2_summaries.append(s)
    print("Done. Summary:", s)

# ---- save summary with correct relative path ----
output_file = os.path.join(output_dir, "summary_case2_safe.json")
with open(output_file, "w") as f:
    json.dump(case2_summaries, f, indent=2)

print(f"\nCase 2 (safe) finished. Summaries saved to {output_file}")
