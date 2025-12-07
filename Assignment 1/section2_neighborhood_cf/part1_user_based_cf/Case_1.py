# ============================
# CASE STUDY 1
# ============================
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
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

# ---- per-target computation function ----
def process_target_fast(target_idx, overlap_pct=0.30, top_percent=0.05, max_neighbors=500, max_items=5000):
    t0 = time.time()
    t_items = set(R[target_idx].nonzero()[1].tolist())
    n_t_items = len(t_items)
    req_overlap = max(1, math.ceil(overlap_pct * n_t_items))

    # 1) Cosine similarity (one-to-many)
    sims = cosine_similarity(R.getrow(target_idx), R).flatten()
    sims[target_idx] = -1.0

    # 2) top-k neighbors (raw)
    k = max(1, int(math.ceil(top_percent * n_users)))
    k = min(k, max_neighbors)
    topk_raw = np.argpartition(-sims, k)[:k]
    topk_raw = topk_raw[np.argsort(-sims[topk_raw])]
    topk_raw = np.array([u for u in topk_raw if sims[u]>0], dtype=int)

    # 3) Predict ratings (raw)
    neigh_rows = R[topk_raw]
    neigh_items = np.unique(neigh_rows.nonzero()[1])
    candidate_items = np.setdiff1d(neigh_items, list(t_items), assume_unique=True)[:max_items]

    preds_before = {}
    for it in candidate_items:
        neighbor_ratings = neigh_rows[:, it].toarray().flatten() if len(topk_raw)>0 else np.array([])
        mask = neighbor_ratings != 0
        if mask.sum() == 0: continue
        weights = sims[topk_raw][mask]
        ratings = neighbor_ratings[mask]
        denom = np.sum(np.abs(weights))
        if denom==0: continue
        preds_before[int(it)] = float(np.dot(weights, ratings)/denom)

    # 4) Compute DF & DS
    if len(t_items)==0:
        overlaps = np.zeros(n_users, dtype=int)
    else:
        overlaps = np.array(R[:, list(t_items)].getnnz(axis=1)).reshape(-1)
    dfactors = np.clip(overlaps/req_overlap,0,1)
    ds = sims * dfactors
    ds[target_idx] = -1.0

    # 5) top-k by DS
    topk_ds = np.argpartition(-ds, k)[:k]
    topk_ds = topk_ds[np.argsort(-ds[topk_ds])]
    topk_ds = np.array([u for u in topk_ds if ds[u]>0], dtype=int)

    # 6) Predict using DS neighbors
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

    # 7-8) Compare neighbors & predictions
    neighbor_overlap = len(set(topk_raw)&set(topk_ds))
    common_items_preds = set(preds_before.keys()) & set(preds_after.keys())
    mad = float(np.mean([abs(preds_before[it]-preds_after[it]) for it in common_items_preds])) if common_items_preds else None

    # 9) Perfect cosine =1 examples
    perfect_examples = [ (pi, idx_to_user[pi], counts_per_user[pi])
                         for pi in np.where(np.abs(sims-1.0)<1e-9)[0]
                         if pi!=target_idx and counts_per_user[pi]!=counts_per_user[target_idx] ][:5]

    # 10) Neighbors with common items
    neighbor_common_info = [(int(v), idx_to_user[v], int(overlaps[v])) for v in topk_raw[:20]]

    # Save summaries
    summary = {
        "target_idx": int(target_idx),
        "target_user_id": idx_to_user[target_idx],
        "n_target_items": n_t_items,
        "num_neighbors_raw": len(topk_raw),
        "num_neighbors_ds": len(topk_ds),
        "neighbor_overlap_count": neighbor_overlap,
        "num_candidates_before": len(preds_before),
        "num_candidates_after": len(preds_after),
        "mad_on_common_predictions": mad,
        "perfect_similarity_examples": perfect_examples,
        "sample_neighbor_common_info": neighbor_common_info,
        "time_seconds": round(time.time()-t0,2)
    }

    return summary, preds_before, preds_after

# ---- Run for all target users ----
case1_summaries = []
for t in target_user_idxs:
    print("\nProcessing target:", idx_to_user[t])
    s, p_before, p_after = process_target_fast(t)
    case1_summaries.append(s)
    print(f"Done in {s['time_seconds']} sec | neighbors_raw={s['num_neighbors_raw']} | neighbors_ds={s['num_neighbors_ds']} | preds_before={s['num_candidates_before']} | preds_after={s['num_candidates_after']}")

# ---- Save summary JSON with correct relative path ----
output_file = os.path.join(output_dir, "summary_case1_fast.json")
with open(output_file, "w") as f:
    json.dump(case1_summaries, f, indent=2)

print(f"\nCase 1 FAST finished. Summaries saved to {output_file}")
