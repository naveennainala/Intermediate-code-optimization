"""Demo: source -> TAC -> ML picks pass sequence -> optimized TAC (verified equivalent).

Usage:  python -m mltac.demo            (uses a built-in example)
        python -m mltac.demo my_prog.txt   (file with simple assignments)
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import compile_source, extract_features
from .run_experiments import build_dataset
from .tac import FULL_PIPELINE, apply_sequence, cost, equivalent, SEQUENCES

EXAMPLE = """x = a*8 + b*1
y = a*8 + 0
z = 3*4 + x
w = y + z
v = w * 2
out = v - 0
"""

def main():
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else EXAMPLE
    print("Training selector on 2000 synthetic programs ...")
    _, X, y, *_ = build_dataset(2000, 123, verify=False)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)

    p = compile_source(src)
    print("\n--- Source ---\n" + src.strip())
    print("\n--- Three-address code (cost = %d) ---\n%s" % (cost(p), p.text()))
    feats = np.array([extract_features(p)])
    proba = model.predict_proba(feats)[0]
    seq, conf = model.classes_[proba.argmax()], proba.max()
    if conf < 0.5:                                  # confidence-threshold extension
        print(f"\nLow confidence ({conf:.2f}) -> falling back to full pipeline")
        seq = FULL_PIPELINE
    q = apply_sequence(p, str(seq))
    print(f"\nML-predicted sequence: {seq}  passes={SEQUENCES[str(seq)]}  (confidence {conf:.2f})")
    print("\n--- Optimized TAC (cost = %d) ---\n%s" % (cost(q), q.text()))
    print("\nSemantically equivalent:", equivalent(p, q))

if __name__ == "__main__":
    main()
