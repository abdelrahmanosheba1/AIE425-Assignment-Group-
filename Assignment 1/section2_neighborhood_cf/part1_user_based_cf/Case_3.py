# ============================
# CASE STUDY 3
# Pearson Correlation User-Based CF + DF + DS
# ============================
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import math, os, json, time

# ---- Setup output directory with correct relative path ----
output_dir = os.path.join("..", "..", "results(output_files,visualizations)", "part1_user_based_cf_results")
os.makedirs(output_dir, exist_ok=True)

# ---- Load dataset ----
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
    except Exception as e:
        continue

if df is None:
    raise FileNotFoundError(f"No ratings CSV found in {data_dir}/")

# ---- aggregate duplicates ----
df = df.groupby(["user_id","item_id"], as_index=False)["rating"].mean()

# ---- factorize ids ----
user_codes, users = pd.factorize(df["user_id"])
item_codes, items = pd.factorize(df["item_id"])
n_users = len(users)
n_items = len(items)
print(f"n_users={n_users}, n_items={n_items}, ratings={len(df)}")

# ---- build sparse matrix ----
R = csr_matrix((df["rating"].values, (user_codes, item_codes)), shape=(n_users, n_items))

# ---- helper maps ----
idx_to_user = {i:users[i] for i in range(n_users)}
idx_to_item = {j:items[j] for j in range(n_items)}

# ---- pick 3 target users (robust, low/medium activity) ----
counts_per_user = np.array((R!=0).sum(axis=1)).reshape(-1)
total_ratings = R.nnz
user_pct = counts_per_user / total_ratings * 100

def pick_user_in_range(low_pct, high_pct):
    cand = np.where((user_pct >= low_pct) & (user_pct < high_pct))[0]
    if len(cand) > 0:
        return cand[np.argmax(counts_per_user[cand])]
    return int(np.argmin(np.abs(user_pct - (low_pct + high_pct)/2)))

target_user_idxs = [pick_user_in_range(0, 2), pick_user_in_range(2, 5), pick_user_in_range(5, 10)]
print("Target users:", [(i, idx_to_user[i]) for i in target_user_idxs])

# ---- pick 2 lowest-average items ----
item_sums = np.array(R.sum(axis=0)).reshape(-1)
item_counts = np.array((R!=0).sum(axis=0)).reshape(-1)
item_avg = np.divide(item_sums, np.where(item_counts>0,item_counts,1))
low2 = np.argsort(item_avg)[:2].tolist()
print("Target items:", [(j, idx_to_item[j], item_avg[j]) for j in low2])

# ---- per-target PCC computation function ----
def process_target_pearson(target_idx, overlap_pct=0.30, top_percent=0.05, max_neighbors=500, max_items=5000):
    t0 = time.time()
    t_items = set(R[target_idx].nonzero()[1].tolist())
    n_t_items = len(t_items)
    req_overlap = max(1, math.ceil(overlap_pct * n_t_items))

    # target ratings
    target_ratings = R.getrow(target_idx).toarray().flatten()
    target_mask = target_ratings != 0
    
    # **OPTIMIZATION: Find candidate neighbors (users who rated ≥1 of target's items)**
    if len(t_items) > 0:
        potential_neighbors = set()
        for it in t_items:
            potential_neighbors.update(R[:, it].nonzero()[0].tolist())
        potential_neighbors.discard(target_idx)
    else:
        potential_neighbors = set()
    
    # Compute Pearson similarity (vectorized, safer version)
    sims = np.zeros(n_users, dtype=np.float32)
    sims[target_idx] = -1.0
    
    if target_mask.sum() < 2 or len(potential_neighbors) == 0:
        # Not enough ratings to compute meaningful PCC
        topk = np.array([], dtype=int)
    else:
        t_vals_centered = target_ratings[target_mask] - target_ratings[target_mask].mean()
        t_norm = np.linalg.norm(t_vals_centered)
        
        if t_norm < 1e-9:
            # Target user has no variance, skip PCC
            topk = np.array([], dtype=int)
        else:
            # **Only compute PCC for candidate neighbors (max 5000 to prevent timeout)**
            for u in list(potential_neighbors)[:min(len(potential_neighbors), 5000)]:
                u_ratings = R.getrow(u).toarray().flatten()
                common_mask = (target_mask & (u_ratings != 0))
                
                if common_mask.sum() < 2:
                    sims[u] = 0.0
                    continue
                
                u_common = u_ratings[common_mask]
                t_common = target_ratings[common_mask]
                
                u_centered = u_common - u_common.mean()
                t_centered = t_common - t_common.mean()
                
                u_norm = np.linalg.norm(u_centered)
                denom = t_norm * u_norm
                
                if denom > 1e-9:
                    sims[u] = float(np.dot(t_centered, u_centered) / denom)
                else:
                    sims[u] = 0.0
        
            # ---- top-k neighbors ----
            k = max(1, int(math.ceil(top_percent * len(potential_neighbors))))
            k = min(k, max_neighbors)
            
            valid_sims = sims[sims > 0]
            if len(valid_sims) == 0:
                topk = np.array([], dtype=int)
            else:
                k = min(k, len(valid_sims))
                topk = np.argpartition(-sims, k)[:k]
                topk = topk[np.argsort(-sims[topk])]
                topk = np.array([u for u in topk if sims[u] > 0], dtype=int)

    # ---- predict missing ratings ----
    if len(topk) == 0:
        preds_before = {}
    else:
        neigh_rows = R[topk]
        neigh_items = np.unique(neigh_rows.nonzero()[1])
        candidate_items = np.setdiff1d(neigh_items, list(t_items), assume_unique=True)[:max_items]

        preds_before = {}
        for it in candidate_items:
            neighbor_ratings = neigh_rows[:, it].toarray().flatten()
            mask = neighbor_ratings != 0
            if mask.sum() == 0:
                continue
            weights = sims[topk][mask]
            ratings = neighbor_ratings[mask]
            denom = np.sum(np.abs(weights))
            if denom > 1e-9:
                preds_before[int(it)] = float(np.dot(weights, ratings) / denom)

    # ---- DF & DS ----
    if len(t_items)==0:
        overlaps = np.zeros(n_users, dtype=int)
    else:
        overlaps = np.array(R[:, list(t_items)].getnnz(axis=1)).reshape(-1)
    dfactors = np.clip(overlaps/req_overlap,0,1)
    ds = sims * dfactors
    ds[target_idx] = -1.0

    # ---- top-k by DS ----
    k = max(1, int(math.ceil(top_percent * len(potential_neighbors))))
    k = min(k, max_neighbors)
    topk_ds = np.argpartition(-ds, k)[:k]
    topk_ds = topk_ds[np.argsort(-ds[topk_ds])]
    topk_ds = np.array([u for u in topk_ds if ds[u]>0], dtype=int)

    # ---- predict again using DS neighbors ----
    neigh_rows_ds = R[topk_ds] if len(topk_ds)>0 else None
    candidate_items_ds = np.unique(neigh_rows_ds.nonzero()[1]) if neigh_rows_ds is not None else np.array([], dtype=int)
    candidate_items_ds = np.setdiff1d(candidate_items_ds, list(t_items), assume_unique=True)[:max_items]

    preds_after = {}
    for it in candidate_items_ds:
        neighbor_ratings = neigh_rows_ds[:, it].toarray().flatten() if len(topk_ds)>0 else np.array([])
        mask = neighbor_ratings != 0
        if mask.sum() == 0: continue
        weights = ds[topk_ds][mask]
        ratings = neighbor_ratings[mask]
        denom = np.sum(np.abs(weights))
        if denom==0: continue
        preds_after[int(it)] = float(np.dot(weights, ratings)/denom)

    # ---- comparisons & analysis ----
    neighbor_overlap = len(set(topk) & set(topk_ds)) if len(topk) > 0 and len(topk_ds) > 0 else 0
    common_items_preds = set(preds_before.keys()) & set(preds_after.keys())
    mad = float(np.mean([abs(preds_before[it]-preds_after[it]) for it in common_items_preds])) if common_items_preds else None

    # users with negative PCC
    neg_examples = [(pi, idx_to_user[pi], counts_per_user[pi]) for pi in np.where(sims<0)[0]][:5]

    summary = {
        "target_idx": int(target_idx),
        "target_user_id": idx_to_user[target_idx],
        "n_target_items": n_t_items,
        "num_neighbors_pearson": len(topk),
        "num_neighbors_ds": len(topk_ds),
        "neighbor_overlap_count": neighbor_overlap,
        "num_candidates_before": len(preds_before),
        "num_candidates_after": len(preds_after),
        "mad_on_common_predictions": mad,
        "negative_pearson_examples": neg_examples,
        "time_seconds": round(time.time()-t0,2)
    }

    return summary, preds_before, preds_after


case3_summaries = []
for t in target_user_idxs:
    print("\nProcessing target:", idx_to_user[t])
    s, p_before, p_after = process_target_pearson(t)
    case3_summaries.append(s)
    print(f"Done in {s['time_seconds']} sec | neighbors={s['num_neighbors_pearson']} | neighbors_ds={s['num_neighbors_ds']} | preds_before={s['num_candidates_before']} | preds_after={s['num_candidates_after']}")

# ---- Save summary JSON with correct relative path ----
output_file = os.path.join(output_dir, "summary_case3_fast.json")
with open(output_file, "w") as f:
    json.dump(case3_summaries, f, indent=2)

print(f"\nCase 3 FAST finished. Summaries saved to {output_file}")