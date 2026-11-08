"""Reproduce the code-mixing analysis of the paper from the released files.

Usage (from the repository root):
    python data/download_afrisenti.py      # fetches the AfriSenti Yoruba test set into data/raw/
    python src/analysis.py                 # writes results/analysis/
Inputs: data/annotations/*.csv, results/preds/*.csv, data/raw/test.tsv
"""
import os, json, glob, unicodedata, csv, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "analysis")
os.makedirs(OUT, exist_ok=True)
LABS = ["negative", "neutral", "positive"]
TEST_ID_OFFSET = 10612          # ids follow notebook 01: train (8,522) + dev (2,090) rows come first

# ---- annotations ----
layer = pd.read_csv(os.path.join(ROOT, "data", "annotations", "afrisenti_yor_test_codemixing_v1.csv")).set_index("id")
dbl = pd.read_csv(os.path.join(ROOT, "data", "annotations", "double_annotated_before_discussion.csv")).set_index("id")
# ---- test universe ----
preds = {}
models = ["lstm", "bilstm_attn", "mbert", "xlmr_base", "afriberta_large", "afroxlmr_large"]
for m in models:
    fs = sorted(glob.glob(os.path.join(ROOT, "results", "preds", f"{m}_seed*_test.csv")))
    assert len(fs) == 5, (m, len(fs))
    preds[m] = [pd.read_csv(f).set_index("id") for f in fs]
base = preds["afroxlmr_large"][0][["label"]].copy()
raw = pd.read_csv(os.path.join(ROOT, "data", "raw", "test.tsv"), sep="\t", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
raw["tweet"] = raw["tweet"].astype(str).str.replace('"', '', regex=False).str.strip()
raw["id"] = [f"yor_{TEST_ID_OFFSET + i:05d}" for i in range(len(raw))]
raw = raw.set_index("id")
T = base.join(raw[["tweet"]], how="left")
assert T.tweet.notna().all(), "test.tsv does not line up with the prediction ids"
md5 = T.loc[layer.index, "tweet"].map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
assert (md5 == layer.text_md5).all(), "tweet text hash mismatch: wrong version of test.tsv?"
T["cmi"] = layer.cmi.reindex(T.index).fillna(0.0)
T["stratum"] = np.where(T.index.isin(layer.index[layer.sampling_group == "S1_candidate"]), "S1", "S0")
N = T.stratum.value_counts().to_dict()
print("Test strata:", N)

# ---- agreement (before discussion) ----
both = pd.DataFrame({"cm_A": dbl.is_codemixed_A, "cm_B": dbl.is_codemixed_B, "dominance_A": dbl.dominance_A,
                     "dominance_B": dbl.dominance_B, "trans_A": dbl.translation_pair_A, "trans_B": dbl.translation_pair_B})
agree = {"n_shared_labelled": int(len(both)), "raw_agreement": float((both.cm_A == both.cm_B).mean()),
         "kappa": float(cohen_kappa_score(both.cm_A, both.cm_B))}
bb = both[(both.cm_A == 1) & (both.cm_B == 1)].dropna(subset=["dominance_A", "dominance_B"])
agree["dominance_n"] = int(len(bb)); agree["dominance_kappa"] = float(cohen_kappa_score(bb.dominance_A, bb.dominance_B))
rngk = np.random.default_rng(7); ks = []
a_, b_ = both.cm_A.values, both.cm_B.values
for _ in range(5000):
    ii = rngk.integers(0, len(a_), len(a_))
    if len(set(a_[ii])) > 1 or len(set(b_[ii])) > 1: ks.append(cohen_kappa_score(a_[ii], b_[ii]))
agree["kappa_ci"] = [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))]
t0 = both[(both.cm_A == 0) & (both.cm_B == 0)]
agree["trans_n"] = int(len(t0)); agree["trans_kappa"] = float(cohen_kappa_score(t0.trans_A, t0.trans_B))
agree["trans_raw"] = float((t0.trans_A == t0.trans_B).mean())
print("Agreement:", agree)

# ---- final labels (after discussion) ----
L = pd.DataFrame({"cm": layer.is_codemixed.astype(int), "dominance": layer.dominance,
                  "trans": layer.translation_pair.astype(int)})
n_adjudicated = int(layer.adjudicated.sum())
D = T.join(L, how="inner")
n_ann = D.stratum.value_counts().to_dict()
D["w"] = D.stratum.map({s: N[s] / n_ann[s] for s in n_ann})
print("Annotated by stratum:", n_ann)

# ---- features ----
def has_diacritics(s):
    s = unicodedata.normalize("NFD", str(s))
    return int(any(unicodedata.combining(ch) for ch in s))
D["n_tokens"] = D.tweet.str.split().str.len()
D["diacritics"] = D.tweet.map(has_diacritics)
D["y"] = D.label.map(LABS.index)
D["group"] = np.where(D.cm == 1, "mixed", np.where(D.trans == 1, "translation", "monolingual"))

# ---- prevalence (stratified estimator, normal CI) ----
def strat_prop(col):
    est, var = 0.0, 0.0
    Ntot = sum(N.values())
    for s, g in D.groupby("stratum"):
        p = g[col].mean(); n = len(g); W = N[s] / Ntot
        fpc = 1 - n / N[s]
        est += W * p; var += W**2 * p * (1 - p) / max(n - 1, 1) * fpc
    se = var ** 0.5
    return est, max(0, est - 1.96 * se), est + 1.96 * se
D["is_trans"] = D.trans * (1 - D.cm)
prev = {k: strat_prop(c) for k, c in [("mixed", "cm"), ("translation_pair", "is_trans")]}
def census_ci(col):
    s1 = D[D.stratum == "S1"]; s0 = D[D.stratum == "S0"]
    k1 = s1[col].sum() * N["S1"] / len(s1)          # S1 is (near) complete
    k0, n0 = s0[col].sum(), len(s0); z = 1.96
    ph = k0 / n0; den = 1 + z*z/n0; c = (ph + z*z/(2*n0)) / den; h = z*((ph*(1-ph)/n0 + z*z/(4*n0*n0))**0.5) / den
    lo0, hi0 = max(0, c - h), c + h
    Nt = sum(N.values())
    return dict(est=(k1 + ph*N["S0"]) / Nt, lo=(k1 + lo0*N["S0"]) / Nt, hi=(k1 + hi0*N["S0"]) / Nt,
                count_est=float(k1 + ph*N["S0"]), count_hi=float(k1 + hi0*N["S0"]), s0_hi_rate=float(hi0))
prev_census = {"mixed": census_ci("cm"), "translation_pair": census_ci("is_trans")}
by_stratum = D.groupby("stratum").agg(n=("cm", "size"), mixed=("cm", "sum"), trans=("is_trans", "sum")).to_dict("index")
dom = D[D.cm == 1].dominance.fillna("missing").value_counts().to_dict()
est_mixed_count = float((D.w * D.cm).sum())

# ---- heuristic detector vs human (weighted) ----
def prf(pred):
    tp = (D.w * pred * D.cm).sum(); fp = (D.w * pred * (1 - D.cm)).sum(); fn = (D.w * (1 - pred) * D.cm).sum()
    p = tp / (tp + fp); r = tp / (tp + fn); return dict(precision=p, recall=r, f1=2 * p * r / (p + r))
heur = {"CMI>0": prf((D.cmi.fillna(0) > 0).astype(int))}
for th in [10, 20, 30]:
    heur[f"CMI>{th}"] = prf((D.cmi.fillna(0) > th).astype(int))

# ---- weighted F1 helpers ----
def wf1(y, p, w):
    f, tot = 0.0, w.sum()
    for c in range(3):
        tp = w[(y == c) & (p == c)].sum(); fp = w[(y != c) & (p == c)].sum(); fn = w[(y == c) & (p != c)].sum()
        sup = w[y == c].sum()
        if sup == 0: continue
        f1 = 0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
        f += f1 * sup / tot
    return f
ids = D.index
y = D.y.values; w = D.w.values; mix = D.cm.values == 1
P = {m: np.stack([d.loc[ids, "pred"].map(LABS.index).values for d in preds[m]]) for m in models}
V = {m: np.array([np.bincount(P[m][:, i], minlength=3).argmax() for i in range(len(ids))]) for m in models}

rows = []
for m in models:
    seeds_mix = [wf1(y[mix], P[m][k][mix], w[mix]) for k in range(5)]
    seeds_non = [wf1(y[~mix], P[m][k][~mix], w[~mix]) for k in range(5)]
    gaps = [a - b for a, b in zip(seeds_mix, seeds_non)]
    rows.append(dict(model=m, mixed_mean=np.mean(seeds_mix) * 100, mixed_sd=np.std(seeds_mix, ddof=1) * 100,
                     non_mean=np.mean(seeds_non) * 100, non_sd=np.std(seeds_non, ddof=1) * 100,
                     gap_mean=np.mean(gaps) * 100, gap_sd=np.std(gaps, ddof=1) * 100))
R = pd.DataFrame(rows).set_index("model")

# stratified bootstrap of the gap on majority vote
rng = np.random.default_rng(2026)
B = 10000
s1 = np.where(D.stratum.values == "S1")[0]; s0 = np.where(D.stratum.values == "S0")[0]
gap_ci = {m: [] for m in models}
for b in range(B):
    idx = np.concatenate([rng.choice(s1, len(s1)), rng.choice(s0, len(s0))])
    yy, ww, mm = y[idx], w[idx], mix[idx]
    for m in models:
        pp = V[m][idx]
        gap_ci[m].append(wf1(yy[mm], pp[mm], ww[mm]) - wf1(yy[~mm], pp[~mm], ww[~mm]))
for m in models:
    g = np.array(gap_ci[m]) * 100
    R.loc[m, "vote_gap"] = (wf1(y[mix], V[m][mix], w[mix]) - wf1(y[~mix], V[m][~mix], w[~mix])) * 100
    R.loc[m, "gap_lo"], R.loc[m, "gap_hi"] = np.percentile(g, [2.5, 97.5])

# sensitivity: every unannotated S0 tweet treated as non-mixed (audit found none mixed); unweighted, full test set
allids = T.index
mix_full = np.isin(allids, D.index[D.cm == 1])
yF = T.label.map(LABS.index).values; wF = np.ones(len(allids))
sens = {}
for m in models:
    PF = np.stack([d.loc[allids, "pred"].map(LABS.index).values for d in preds[m]])
    gm = [wf1(yF[mix_full], PF[k][mix_full], wF[mix_full]) * 100 for k in range(5)]
    gn = [wf1(yF[~mix_full], PF[k][~mix_full], wF[~mix_full]) * 100 for k in range(5)]
    VF = np.array([np.bincount(PF[:, i], minlength=3).argmax() for i in range(len(allids))])
    bs = []
    rng2 = np.random.default_rng(11); im, inn = np.where(mix_full)[0], np.where(~mix_full)[0]
    for _ in range(2000):
        a_ = rng2.choice(im, len(im)); b_ = rng2.choice(inn, len(inn))
        bs.append((wf1(yF[a_], VF[a_], wF[a_]) - wf1(yF[b_], VF[b_], wF[b_])) * 100)
    sens[m] = dict(mixed=np.mean(gm), non=np.mean(gn), gap=np.mean(gm) - np.mean(gn), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), n_mixed=int(mix_full.sum()))

# by dominance, best two models, vote
dom_f1 = {}
for m in ["afroxlmr_large", "afriberta_large", "xlmr_base", "lstm"]:
    dom_f1[m] = {}
    for dv in ["YO", "BAL", "EN"]:
        sel = (D.cm.values == 1) & (D.dominance.values == dv)
        dom_f1[m][dv] = wf1(y[sel], V[m][sel], w[sel]) * 100
dom_n = {dv: int(((D.cm == 1) & (D.dominance == dv)).sum()) for dv in ["YO", "BAL", "EN"]}

# three-way groups (mixed / translation / monolingual), vote, weighted
grp = {}
for m in models:
    grp[m] = {gname: wf1(y[D.group.values == gname], V[m][D.group.values == gname], w[D.group.values == gname]) * 100
              for gname in ["mixed", "translation", "monolingual"]}
gsize = D.groupby("group").agg(n=("w", "size"), weighted=("w", "sum")).to_dict("index")

# label distribution by group (weighted)
labdist = (D.groupby(["group", "label"]).w.sum() / D.groupby("group").w.sum()).unstack().round(3).to_dict("index")

# ---- logistic regression: correct ~ mixed + label + log length + diacritics ----
logit = {}
X = pd.DataFrame({"mixed": D.cm.values, "neutral": (D.label == "neutral").astype(int).values,
                  "positive": (D.label == "positive").astype(int).values,
                  "log_tokens": np.log(D.n_tokens.values), "diacritics": D.diacritics.values})
X = sm.add_constant(X)
for m in models:
    correct = (V[m] == y).astype(int)
    fit = sm.GLM(correct, X, family=sm.families.Binomial(), var_weights=w / w.mean()).fit(cov_type="HC1")
    b, se = fit.params["mixed"], fit.bse["mixed"]
    logit[m] = dict(or_mixed=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)), p=float(fit.pvalues["mixed"]),
                    or_diacritics=float(np.exp(fit.params["diacritics"])), p_diacritics=float(fit.pvalues["diacritics"]))

# ---- errors of the best model on mixed tweets (ids only; categories coded by the authors) ----
best = "afroxlmr_large"
sel = (D.cm.values == 1) & (V[best] != y)
err = pd.DataFrame({"gold": D.label.values[sel], "pred": [LABS[i] for i in V[best][sel]],
                    "dominance": D.dominance.values[sel]}, index=D.index[sel])
cat_path = os.path.join(ROOT, "data", "annotations", "error_categories.csv")
if os.path.exists(cat_path):
    err = err.join(pd.read_csv(cat_path).set_index("id")[["category"]])
err.index.name = "id"; err.to_csv(os.path.join(OUT, "best_model_errors_on_mixed.csv"))
res = dict(error_categories=(err.category.value_counts().to_dict() if "category" in err else None), sensitivity=sens, dominance_f1=dom_f1, dominance_n=dom_n, seed_gaps=R[["gap_mean","gap_sd"]].to_dict("index"), prev_census=prev_census, n_adjudicated=n_adjudicated, agreement=agree, strata_N=N, annotated=n_ann, by_stratum=by_stratum, prevalence=prev,
           est_mixed_count=est_mixed_count, dominance=dom, heuristic=heur, groups=gsize, label_dist=labdist,
           three_way=grp, logit=logit, n_errors_best_mixed=int(((D.cm == 1) & (V[best] != y)).sum()))
R.round(2).to_csv(os.path.join(OUT, "mixed_vs_nonmixed_design_based.csv"))
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(R.round(2)); print(json.dumps(res, indent=1, default=lambda v: round(float(v), 4)))
