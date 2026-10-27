"""Run the full experiment: dataset -> labels -> models -> evaluation -> figures.

Usage:  python -m mltac.run_experiments
All numbers in the paper come from this script's output (results/results.json).
"""
import json
import os
import random
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import FEATURE_NAMES, extract_features, generate_program
from .tac import (FULL_PIPELINE, SEQUENCES, apply_sequence, compile_effort,
                  cost, equivalent)

SEED = 42
N_PROGRAMS = 6000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
SEQ_NAMES = list(SEQUENCES)


# ------------------------------------------------------------- dataset
def label_program(p):
    """Return (best_sequence, cost_per_sequence). Ties -> fewest passes."""
    costs = {}
    for s in SEQ_NAMES:
        q = apply_sequence(p, s)
        costs[s] = cost(q)
    best = min(SEQ_NAMES, key=lambda s: (costs[s], compile_effort(s)))
    return best, costs


def build_dataset(n, seed, n_lo=8, n_hi=40, verify=True):
    rng = random.Random(seed)
    seen, progs = set(), []
    while len(progs) < n:
        p = generate_program(rng, n_lo, n_hi)
        k = p.key()
        if k in seen or len(p.instrs) == 0:
            continue                      # de-duplicate (leakage control)
        seen.add(k)
        progs.append(p)
    X, y, C, C0 = [], [], [], []
    bad = 0
    for i, p in enumerate(progs):
        best, costs = label_program(p)
        if verify and i % 5 == 0 and not equivalent(p, apply_sequence(p, best), seed=i):
            bad += 1
        X.append(extract_features(p)); y.append(best)
        C.append([costs[s] for s in SEQ_NAMES]); C0.append(cost(p))
    return progs, np.array(X), np.array(y), np.array(C), np.array(C0), bad


# ------------------------------------------------- heuristic baseline
def heuristic_predict(X):
    """Hand-written rule-based selector (what a compiler writer might code)."""
    ix = {n: i for i, n in enumerate(FEATURE_NAMES)}
    table = {(0, 0, 0): "S1_dce", (1, 0, 0): "S2_cf", (0, 1, 0): "S3_alg",
             (0, 0, 1): "S4_cse", (1, 1, 0): "S5_cf_alg", (1, 0, 1): "S6_cf_cse",
             (0, 1, 1): "S7_alg_cse", (1, 1, 1): "S8_full"}
    out = []
    for r in X:
        cf = int(r[ix["n_const_const"]] > 0)
        alg = int(r[ix["n_identity"]] + r[ix["n_pow2_mul"]] + r[ix["n_pow2_div"]] > 0)
        cs = int(r[ix["n_dup_expr"]] > 0)
        out.append(table[(cf, alg, cs)])
    return np.array(out)


def make_models():
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
    }


def optimal_rate(pred, C):
    """Fraction of programs where the chosen sequence achieves the minimum cost."""
    idx = np.array([SEQ_NAMES.index(s) for s in pred])
    return float((C[np.arange(len(C)), idx] == C.min(1)).mean())


def policy_stats(pred, C, C0, y_true=None):
    """Cost reduction (%) and mean compile effort for a sequence-selection policy."""
    idx = np.array([SEQ_NAMES.index(s) for s in pred])
    chosen = C[np.arange(len(C)), idx]
    red = 100 * (1 - chosen.sum() / C0.sum())
    effort = float(np.mean([compile_effort(s) for s in pred]))
    return red, effort


def main():
    t0 = time.time()
    res = {"config": {"seed": SEED, "n_programs": N_PROGRAMS, "sequences": SEQUENCES}}
    print("Building dataset ...")
    progs, X, y, C, C0, bad = build_dataset(N_PROGRAMS, SEED)
    res["dataset"] = {"n": len(y), "label_counts": dict(Counter(y)),
                      "equivalence_failures_checked": int(bad),
                      "mean_instr": float(X[:, 0].mean())}
    print("labels:", dict(Counter(y)), "| equivalence failures:", bad)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    assert not ({progs[i].key() for i in tr} & {progs[i].key() for i in te}), "LEAKAGE"
    res["leakage_check"] = "no identical program appears in both train and test"
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    Ctr, Cte, C0te = C[tr], C[te], C0[te]

    # reference policies on the test set
    refs = {
        "No optimization": np.array(["S0_none"] * len(te)),
        "Fixed full pipeline": np.array([FULL_PIPELINE] * len(te)),
        "Heuristic rules": heuristic_predict(Xte),
        "Oracle": yte,
    }
    table = {}
    for name, pred in refs.items():
        red, eff = policy_stats(pred, Cte, C0te)
        table[name] = {"acc": float(accuracy_score(yte, pred)),
                       "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                       "macro_precision": float(precision_score(yte, pred, average="macro", zero_division=0)),
                       "macro_recall": float(recall_score(yte, pred, average="macro", zero_division=0)),
                       "optimal_rate": optimal_rate(pred, Cte),
                       "cost_reduction_pct": red, "mean_passes": eff}

    # ML models: 5-fold CV on train + hold-out test
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models, fitted = make_models(), {}
    for name, m in models.items():
        cv = []
        for a, b in skf.split(Xtr, ytr):
            mm = make_models()[name]
            mm.fit(Xtr[a], ytr[a])
            cv.append(accuracy_score(ytr[b], mm.predict(Xtr[b])))
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        red, eff = policy_stats(pred, Cte, C0te)
        table[name] = {"cv_acc_mean": float(np.mean(cv)), "cv_acc_std": float(np.std(cv)),
                       "acc": float(accuracy_score(yte, pred)),
                       "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                       "macro_precision": float(precision_score(yte, pred, average="macro", zero_division=0)),
                       "macro_recall": float(recall_score(yte, pred, average="macro", zero_division=0)),
                       "optimal_rate": optimal_rate(pred, Cte),
                       "cost_reduction_pct": red, "mean_passes": eff}
        fitted[name] = m
        print(f"{name:22s} acc={table[name]['acc']:.3f} f1={table[name]['macro_f1']:.3f} "
              f"opt_rate={table[name]['optimal_rate']:.3f} cost_red={red:.2f}% passes={eff:.2f}")
    res["test_results"] = table

    # best ML model
    ml_names = [n for n in models if n != "Majority baseline"]
    best_name = max(ml_names, key=lambda n: table[n]["macro_f1"])
    best = fitted[best_name]
    res["best_model"] = best_name
    pred = best.predict(Xte)

    # --- extension: confidence-threshold fallback to the full pipeline
    proba = best.predict_proba(Xte)
    conf = proba.max(1)
    thr_rows = []
    for t in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p2 = np.where(conf >= t, best.classes_[proba.argmax(1)], FULL_PIPELINE)
        red, eff = policy_stats(p2, Cte, C0te)
        thr_rows.append({"threshold": t, "optimal_rate": optimal_rate(p2, Cte), "cost_reduction_pct": red, "mean_passes": eff,
                         "fallback_rate": float((conf < t).mean())})
    res["threshold_sweep"] = thr_rows

    # --- robustness: train on short programs, test on much longer ones (OOD)
    _, Xo, yo, Co, C0o, _ = build_dataset(1200, SEED + 7, n_lo=45, n_hi=90, verify=False)
    pred_o = best.predict(Xo)
    red_o, eff_o = policy_stats(pred_o, Co, C0o)
    red_full, eff_full = policy_stats(np.array([FULL_PIPELINE] * len(yo)), Co, C0o)
    res["robustness_long_programs"] = {
        "acc": float(accuracy_score(yo, pred_o)),
        "macro_f1": float(f1_score(yo, pred_o, average="macro", zero_division=0)),
        "optimal_rate": optimal_rate(pred_o, Co),
        "cost_reduction_pct": red_o, "mean_passes": eff_o,
        "fixed_pipeline_cost_reduction_pct": red_full, "fixed_pipeline_passes": eff_full}

    # --- feature importance (RF) for interpretation
    rf = fitted["Random Forest"]
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])
    res["rf_feature_importance"] = [(n, float(v)) for n, v in imp]

    # ---------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    # 1. label distribution
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    cnt = Counter(y)
    ax.bar(range(len(SEQ_NAMES)), [cnt.get(s, 0) for s in SEQ_NAMES], color="#4C72B0")
    ax.set_xticks(range(len(SEQ_NAMES)))
    ax.set_xticklabels([s.split("_")[0] for s in SEQ_NAMES], rotation=45)
    ax.set_ylabel("Programs"); ax.set_xlabel("Best pass sequence")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_label_dist.png"); plt.close(fig)
    # 2. confusion matrix
    labs = [s for s in SEQ_NAMES if s in set(yte) | set(pred)]
    cm = confusion_matrix(yte, pred, labels=labs)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
    ax.set_xticklabels([l.split("_")[0] for l in labs], rotation=45)
    ax.set_yticklabels([l.split("_")[0] for l in labs])
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_confusion.png"); plt.close(fig)
    # 3. feature importance
    top = imp[:10][::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ax.barh([t[0] for t in top], [t[1] for t in top], color="#55A868")
    ax.set_xlabel("Random Forest importance")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_importance.png"); plt.close(fig)
    # 4. policy comparison
    names = ["No optimization", "Fixed full pipeline", "Heuristic rules", best_name, "Oracle"]
    fig, ax1 = plt.subplots(figsize=(3.6, 2.6))
    ax1.bar(range(5), [table[n]["cost_reduction_pct"] for n in names], color="#C44E52")
    ax1.set_ylabel("Cost reduction (%)")
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(["None", "Fixed", "Rules", "ML", "Oracle"])
    ax2 = ax1.twinx()
    ax2.plot(range(5), [table[n]["mean_passes"] for n in names], "ko-")
    ax2.set_ylabel("Mean passes run")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_policies.png"); plt.close(fig)
    # 5. threshold sweep
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot([r["threshold"] for r in thr_rows], [r["cost_reduction_pct"] for r in thr_rows], "o-")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Cost reduction (%)")
    ax2 = ax.twinx()
    ax2.plot([r["threshold"] for r in thr_rows], [r["mean_passes"] for r in thr_rows], "s--", color="gray")
    ax2.set_ylabel("Mean passes")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_threshold.png"); plt.close(fig)

    res["runtime_seconds"] = time.time() - t0
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nBest model:", best_name)
    print("Robustness (long programs):", json.dumps(res["robustness_long_programs"], indent=1))
    print("Saved to", OUT, f"({res['runtime_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
