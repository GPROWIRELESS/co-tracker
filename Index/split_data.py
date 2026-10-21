# # %% [markdown]
# # ## Cell — Create a leakage-safe train/val/test split at *content_id* level (stratified by src_skill × task_group)
# #
# # What this does
# # - Reads your existing `content_index.csv` (with your manual `src_skill`, `Task Name`, etc.)
# # - Assigns `split` for each `content_id` (content-level, so Active+all Passive for that content stay together)
# # - Writes a new CSV (so it won't overwrite your labeled file)
# #
# # Key design choices
# # - Split unit = content_id (prevents leakage between Active and Passive or repeated Passive views)
# # - Stratify by (src_skill, task_group) so distributions are similar across splits
# # - Special handling for extremely small groups (e.g., novice Task D count=3):
# #   By default, groups with < MIN_GROUP_FOR_VALTEST go entirely to TRAIN (stable; avoids meaningless val/test)
# #
# # You must provide how to get task_group (A/B/C/D):
# # - Option 1 (recommended): you already have a column, e.g., "Task Group" with values A/B/C/D
# # - Option 2: map from Task Num to A/B/C/D via TASKNUM_TO_GROUP dict below

# # %%
# from pathlib import Path
# import pandas as pd
# import numpy as np

# # ====== CONFIG ======
# IN_CSV  = Path(__file__).resolve().parent / "content_index_hpc.csv"
# OUT_CSV = Path(__file__).resolve().parent / "content_index_with_split.csv"

# RANDOM_SEED = 20251229

# # split ratios (by content count)
# TRAIN_FRAC = 0.60
# VAL_FRAC   = 0.2
# TEST_FRAC  = 0.2
# assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

# # Minimum group size to make val/test meaningful.
# # If a (skill, task_group) group has fewer than this, we put all into train by default.
# MIN_GROUP_FOR_VALTEST = 5

# # How to derive task_group (A/B/C/D)
# TASK_GROUP_COL = "Task Name"  # e.g., "Task Group" if you already have it; otherwise keep None

# # If TASK_GROUP_COL is None, we map from "Task Num" (int) to "A/B/C/D".
# # Fill this mapping based on your definition.
# TASKNUM_TO_GROUP = {
#     # Example (YOU MUST EDIT):
#     # 1: "A", 2: "A",
#     # 3: "B", 4: "B",
#     # 5: "C", 6: "C",
#     # 7: "D", 8: "D",
# }

# # Column names in your CSV
# CONTENT_ID_COL = "content_id"
# SRC_SKILL_COL  = "src_skill"      # you already filled (novice/intermediate)
# TASKNUM_COL    = "Task Num"       # exists in your schema
# SPLIT_COL      = "split"          # will be created/overwritten in OUT_CSV only
# # =====================

# if not IN_CSV.exists():
#     raise FileNotFoundError(f"IN_CSV not found: {IN_CSV}")

# df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
# for col in [CONTENT_ID_COL, SRC_SKILL_COL, TASKNUM_COL]:
#     if col not in df.columns:
#         raise ValueError(f"Missing required column in CSV: {col}")

# # ----- build task_group -----
# if TASK_GROUP_COL is not None:
#     if TASK_GROUP_COL not in df.columns:
#         raise ValueError(f"TASK_GROUP_COL='{TASK_GROUP_COL}' not found in CSV columns.")
#     df["_task_group"] = df[TASK_GROUP_COL].astype(str).str.strip()
# else:
#     if not TASKNUM_TO_GROUP:
#         raise ValueError(
#             "TASK_GROUP_COL is None but TASKNUM_TO_GROUP is empty.\n"
#             "Either set TASK_GROUP_COL to your A/B/C/D column, or fill TASKNUM_TO_GROUP mapping."
#         )
#     # Map Task Num -> group; unknown will be 'UNK'
#     df["_task_group"] = df[TASKNUM_COL].map(TASKNUM_TO_GROUP).fillna("UNK").astype(str)

# # Normalize src_skill values
# df["_src_skill"] = df[SRC_SKILL_COL].astype(str).str.strip().str.lower()

# # Sanity check: one row per content_id expected (if not, we still split by unique content_id)
# # We'll compute split at content_id level, then merge back.
# content_meta = (
#     df[[CONTENT_ID_COL, "_src_skill", "_task_group"]]
#     .drop_duplicates(subset=[CONTENT_ID_COL])
#     .copy()
# )

# # If any content_id maps to multiple (skill, task_group) combos, that's a data consistency issue
# dup_check = (
#     df.groupby(CONTENT_ID_COL)[["_src_skill", "_task_group"]]
#     .nunique()
# )
# inconsistent = dup_check[(dup_check["_src_skill"] > 1) | (dup_check["_task_group"] > 1)]
# if len(inconsistent) > 0:
#     print("[WARN] Some content_id have inconsistent skill/task_group across rows. Using the first occurrence.")
#     print(inconsistent.head(20))

# # ----- stratified split -----
# rng = np.random.default_rng(RANDOM_SEED)

# def split_group(cids: np.ndarray, train_frac: float, val_frac: float, test_frac: float):
#     """Return dict content_id->split for one group."""
#     n = len(cids)
#     order = rng.permutation(n)
#     cids = cids[order]

#     # initial counts
#     n_train = int(round(n * train_frac))
#     n_val   = int(round(n * val_frac))
#     # ensure total matches
#     n_test  = n - n_train - n_val

#     # fix edge cases (make sure no negative)
#     if n_test < 0:
#         # reduce val first, then train
#         need = -n_test
#         take = min(need, n_val); n_val -= take; need -= take
#         take = min(need, n_train); n_train -= take; need -= take
#         n_test = 0

#     splits = {}
#     splits.update({cid: "train" for cid in cids[:n_train]})
#     splits.update({cid: "val"   for cid in cids[n_train:n_train+n_val]})
#     splits.update({cid: "test"  for cid in cids[n_train+n_val:]})
#     return splits

# content_to_split = {}

# # group by (skill, task_group)
# groups = content_meta.groupby(["_src_skill", "_task_group"])
# for (skill, tg), g in groups:
#     cids = g[CONTENT_ID_COL].to_numpy()
#     n = len(cids)

#     # Special handling for tiny groups (e.g., novice-D=3)
#     if n < MIN_GROUP_FOR_VALTEST:
#         for cid in cids:
#             content_to_split[cid] = "train"
#         continue

#     # Otherwise, do stratified random split within the group
#     sdict = split_group(cids, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
#     content_to_split.update(sdict)

# # For any content not assigned (shouldn't happen), default to train
# for cid in content_meta[CONTENT_ID_COL].to_list():
#     content_to_split.setdefault(cid, "train")

# # merge splits back
# df[SPLIT_COL] = df[CONTENT_ID_COL].map(content_to_split).fillna("train")

# # ----- report summary -----
# def show_counts(title, dfx):
#     print("\n" + title)
#     print(dfx.pivot_table(index=["_src_skill", "_task_group"], columns=SPLIT_COL, values=CONTENT_ID_COL,
#                           aggfunc="count", fill_value=0, margins=True))

# show_counts("Counts by (src_skill, task_group) and split (ROW LEVEL, usually 1 row per content)", df)

# # content-level counts (true split unit)
# tmp = df[[CONTENT_ID_COL, "_src_skill", "_task_group", SPLIT_COL]].drop_duplicates(CONTENT_ID_COL)
# show_counts("Counts by (src_skill, task_group) and split (CONTENT LEVEL)", tmp)

# # How many UNK task_group?
# unk_n = (tmp["_task_group"] == "UNK").sum()
# if unk_n > 0:
#     print(f"\n[WARN] content_id with task_group='UNK': {unk_n}. Fill TASKNUM_TO_GROUP or provide TASK_GROUP_COL.")

# # ----- write new CSV -----
# df.drop(columns=["_src_skill", "_task_group"], inplace=True, errors="ignore")
# df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
# print("\n[OK] Wrote:", OUT_CSV.resolve())

# # Optional: also write just the content-level split table (handy for debugging)
# OUT_SPLIT_ONLY = OUT_CSV.with_name("content_split_only.csv")
# tmp2 = tmp.rename(columns={"_src_skill": "src_skill_norm", "_task_group": "task_group"}) \
#          .sort_values(["src_skill_norm", "task_group", SPLIT_COL, CONTENT_ID_COL])
# tmp2.to_csv(OUT_SPLIT_ONLY, index=False, encoding="utf-8-sig")
# print("[OK] Wrote:", OUT_SPLIT_ONLY.resolve())

from pathlib import Path
import json
import pandas as pd
import numpy as np

# ====== CONFIG ======
IN_CSV  = Path(__file__).resolve().parent / "content_index_hpc.csv"
OUT_CSV = Path(__file__).resolve().parent / "content_index_with_split.csv"

RANDOM_SEED = 20251229
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20

TASK_GROUP_COL = "Task Name"   # A/B/C/D
CONTENT_ID_COL = "content_id"
SRC_SKILL_COL  = "src_skill"   # novice/intermediate (SOURCE operator skill)
PASSIVE_USED_COL = "passive_used"
PASSIVE_MANIFEST_COL = "passive_manifest"
SPLIT_COL = "split"

# passive 训练时希望覆盖的 observer skill
OBSERVER_SKILLS = ["novice", "intermediate"]
TASK_GROUPS = ["A", "B", "C", "D"]

# 小组太小时是否允许“覆盖失败”并给 warning（必须允许，否则会直接崩）
ALLOW_COVERAGE_FAILURE = True

# =====================

rng = np.random.default_rng(RANDOM_SEED)

df = pd.read_csv(IN_CSV, encoding="utf-8-sig")

# ---- normalize columns ----
df["_task_group"] = df[TASK_GROUP_COL].astype(str).str.strip()
df["_src_skill"]  = df[SRC_SKILL_COL].astype(str).str.strip().str.lower()

# content-level table (one row per content_id)
content = (
    df[[CONTENT_ID_COL, "_task_group", "_src_skill", PASSIVE_USED_COL, PASSIVE_MANIFEST_COL]]
    .drop_duplicates(subset=[CONTENT_ID_COL])
    .copy()
)

# --------------------------
# 1) Parse passive manifests to build availability by observer_skill
# --------------------------
# per content_id: passive availability for observer skills + counts
passive_has = {cid: {s: False for s in OBSERVER_SKILLS} for cid in content[CONTENT_ID_COL]}
passive_cnt = {cid: {s: 0 for s in OBSERVER_SKILLS} for cid in content[CONTENT_ID_COL]}

missing_manifest = 0
bad_manifest = 0

for _, row in content.iterrows():
    cid = row[CONTENT_ID_COL]
    used = int(row.get(PASSIVE_USED_COL, 0)) if pd.notna(row.get(PASSIVE_USED_COL, 0)) else 0
    mpath = row.get(PASSIVE_MANIFEST_COL, None)

    if used != 1:
        continue
    if not isinstance(mpath, str) or len(mpath.strip()) == 0:
        missing_manifest += 1
        continue

    mp = Path(mpath)
    if not mp.exists():
        missing_manifest += 1
        continue

    try:
        items = json.loads(mp.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            bad_manifest += 1
            continue
    except Exception:
        bad_manifest += 1
        continue

    for it in items:
        # manifest schema: observer_skill / observer_user / passive_* paths...
        obs_skill = str(it.get("observer_skill", "")).strip().lower()
        if obs_skill in OBSERVER_SKILLS:
            passive_has[cid][obs_skill] = True
            passive_cnt[cid][obs_skill] += 1

print(f"[INFO] missing_manifest={missing_manifest}, bad_manifest={bad_manifest}")

# Attach passive availability to content table
for s in OBSERVER_SKILLS:
    content[f"_passive_has_{s}"] = content[CONTENT_ID_COL].map(lambda cid: passive_has[cid][s])
    content[f"_passive_cnt_{s}"] = content[CONTENT_ID_COL].map(lambda cid: passive_cnt[cid][s])

# --------------------------
# 2) Define "buckets" that must be covered in val and test
#    Passive buckets are defined by (observer_skill, task_group)
# --------------------------
def eligible_for_passive_bucket(row, obs_skill, task_group):
    return (row["_task_group"] == task_group) and bool(row[f"_passive_has_{obs_skill}"])

def eligible_for_active_bucket(row, src_skill, task_group):
    return (row["_src_skill"] == src_skill) and (row["_task_group"] == task_group)

# Build list of required buckets (val and test each need these)
required_passive_buckets = [(obs, tg) for obs in OBSERVER_SKILLS for tg in TASK_GROUPS]
required_active_buckets  = [(sk, tg) for sk in ["novice", "intermediate"] for tg in TASK_GROUPS]

# --------------------------
# 3) Greedy pick for coverage (val then test)
#    We pick content_ids that cover the most yet-uncovered buckets, prefer those with more passive views.
# --------------------------
content_ids = content[CONTENT_ID_COL].tolist()
unassigned = set(content_ids)

split_map = {}  # cid -> split

def score_cid(cid, need_passive, need_active):
    row = content.loc[content[CONTENT_ID_COL] == cid].iloc[0]
    covers = 0
    # how many unmet passive buckets can this cid satisfy?
    for (obs, tg) in need_passive:
        if eligible_for_passive_bucket(row, obs, tg):
            covers += 2  # weight passive higher since it's the pain point
    # how many unmet active buckets can this cid satisfy?
    for (sk, tg) in need_active:
        if eligible_for_active_bucket(row, sk, tg):
            covers += 1
    # tie-breaker: prefer more passive views total
    passive_view_bonus = sum(int(row.get(f"_passive_cnt_{s}", 0)) for s in OBSERVER_SKILLS)
    return (covers, passive_view_bonus)

def greedy_cover(target_split):
    need_passive = set(required_passive_buckets)
    need_active  = set(required_active_buckets)

    chosen = []

    # helper: update needs after choosing a cid
    def update_needs(cid):
        nonlocal need_passive, need_active
        row = content.loc[content[CONTENT_ID_COL] == cid].iloc[0]
        need_passive = {b for b in need_passive if not eligible_for_passive_bucket(row, b[0], b[1])}
        need_active  = {b for b in need_active  if not eligible_for_active_bucket(row, b[0], b[1])}

    # Greedy loop
    while True:
        if len(need_passive) == 0 and len(need_active) == 0:
            break
        if len(unassigned) == 0:
            break

        # find best cid
        best = None
        best_sc = (-1, -1)

        # speed: evaluate only candidates that could help passive first
        # but for simplicity just scan unassigned (ok if content count is not massive)
        for cid in list(unassigned):
            sc = score_cid(cid, need_passive, need_active)
            if sc > best_sc:
                best_sc = sc
                best = cid

        # if best covers nothing, stop
        if best is None or best_sc[0] <= 0:
            break

        # assign
        split_map[best] = target_split
        unassigned.remove(best)
        chosen.append(best)
        update_needs(best)

    # report unmet
    unmet_passive = sorted(list(need_passive))
    unmet_active  = sorted(list(need_active))
    return chosen, unmet_passive, unmet_active

# First satisfy val coverage, then test coverage with remaining content_ids
val_chosen, val_unmet_passive, val_unmet_active = greedy_cover("val")
test_chosen, test_unmet_passive, test_unmet_active = greedy_cover("test")

def print_unmet(tag, unmet_passive, unmet_active):
    if len(unmet_passive) > 0:
        print(f"[WARN] {tag}: unmet PASSIVE buckets (observer_skill, task) = {unmet_passive[:20]} ... total={len(unmet_passive)}")
    if len(unmet_active) > 0:
        print(f"[WARN] {tag}: unmet ACTIVE buckets (src_skill, task) = {unmet_active[:20]} ... total={len(unmet_active)}")

print_unmet("VAL",  val_unmet_passive,  val_unmet_active)
print_unmet("TEST", test_unmet_passive, test_unmet_active)

if (len(val_unmet_passive) > 0 or len(test_unmet_passive) > 0) and not ALLOW_COVERAGE_FAILURE:
    raise RuntimeError("Coverage constraints cannot be satisfied. Set ALLOW_COVERAGE_FAILURE=True to proceed with warnings.")

# --------------------------
# 4) Fill remaining content_ids to hit split ratios (stratified by src_skill × task_group)
# --------------------------
all_cids = content[CONTENT_ID_COL].tolist()
n_total = len(all_cids)

n_val_target  = int(round(n_total * VAL_FRAC))
n_test_target = int(round(n_total * TEST_FRAC))
# train is the rest

def remaining_by_group():
    rem = content[content[CONTENT_ID_COL].isin(list(unassigned))].copy()
    return rem.groupby(["_src_skill", "_task_group"])

def sample_from_group(cids, k):
    cids = list(cids)
    if k <= 0 or len(cids) == 0:
        return []
    k = min(k, len(cids))
    pick = rng.choice(cids, size=k, replace=False).tolist()
    return pick

# Fill val up to target
need_val = max(0, n_val_target - sum(1 for v in split_map.values() if v == "val"))
if need_val > 0:
    groups = remaining_by_group()
    # proportional allocation
    rem_count = sum(len(g) for _, g in groups)
    for (sk, tg), g in groups:
        frac = len(g) / rem_count if rem_count > 0 else 0
        k = int(round(need_val * frac))
        picks = sample_from_group(g[CONTENT_ID_COL], k)
        for cid in picks:
            if cid in unassigned and need_val > 0:
                split_map[cid] = "val"
                unassigned.remove(cid)
                need_val -= 1

# Fill test up to target
need_test = max(0, n_test_target - sum(1 for v in split_map.values() if v == "test"))
if need_test > 0:
    groups = remaining_by_group()
    rem_count = sum(len(g) for _, g in groups)
    for (sk, tg), g in groups:
        frac = len(g) / rem_count if rem_count > 0 else 0
        k = int(round(need_test * frac))
        picks = sample_from_group(g[CONTENT_ID_COL], k)
        for cid in picks:
            if cid in unassigned and need_test > 0:
                split_map[cid] = "test"
                unassigned.remove(cid)
                need_test -= 1

# Remaining goes to train
for cid in list(unassigned):
    split_map[cid] = "train"
unassigned.clear()

# --------------------------
# 5) Merge split back to df rows and report
# --------------------------
df[SPLIT_COL] = df[CONTENT_ID_COL].map(split_map).fillna("train")

def report_passive_coverage(title, dfx_content):
    print("\n" + title)
    for obs in OBSERVER_SKILLS:
        sub = dfx_content[dfx_content[f"_passive_has_{obs}"] == True]
        if len(sub) == 0:
            print(f"  [PASSIVE {obs}] no data at all")
            continue
        tab = sub.pivot_table(index="_task_group", columns=SPLIT_COL, values=CONTENT_ID_COL,
                              aggfunc="count", fill_value=0, margins=True)
        print(f"\n  [PASSIVE observer_skill={obs}] content coverage")
        print(tab)

def report_active_coverage(title, dfx_content):
    print("\n" + title)
    tab = dfx_content.pivot_table(index=["_src_skill", "_task_group"], columns=SPLIT_COL,
                                  values=CONTENT_ID_COL, aggfunc="count", fill_value=0, margins=True)
    print(tab)

content_with_split = content.copy()
content_with_split[SPLIT_COL] = content_with_split[CONTENT_ID_COL].map(split_map)

report_active_coverage("ACTIVE coverage (content-level)", content_with_split)
report_passive_coverage("PASSIVE coverage (content-level, filtered by has_passive)", content_with_split)

# write out
df.drop(columns=["_task_group", "_src_skill"], inplace=True, errors="ignore")
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("\n[OK] Wrote:", OUT_CSV.resolve())
