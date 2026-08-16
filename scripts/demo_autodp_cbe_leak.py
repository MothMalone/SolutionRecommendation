#!/usr/bin/env python3
"""Worked example: the test-label leak in AutoDP's CBE (CatBoost) encoder.

Run with the AutoDP environment, NOT the main one:
    .venv-autodp/bin/python scripts/demo_autodp_cbe_leak.py

The site, verbatim from autodatapre==0.1.12, Search_Space/encoding.py:

    def transform(self):
        X = pd.concat([self.dataset['train'], self.dataset['test']], axis=0)   # :81
        ...
        elif (self.strategy == "CBE"):
            target = pd.concat([self.dataset['target'],
                                self.dataset['target_test']], axis=0)          # :91
            dn = self.CatBoost_encoding(X, target)                             # :92
        normd['train'] = dn.head(trainlen)                                     # :93
        normd['test']  = dn.tail(totallen - trainlen)                          # :94

    def CatBoost_encoding(self, d_train, d_target):
        obtained = enc.fit_transform(X, target)                                # :74

CatBoostEncoder is a *supervised* target encoder. Fitting it on the concatenated frame
with the concatenated target means test labels are an input to the encoding.

Part A shows the mechanism exactly (every number is hand-checkable).
Part B measures how much it is worth on an adversarial synthetic case, over 20 seeds.
"""
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.width", 200)

# CatBoostEncoder's ordered target statistic, for reference:
#     enc(row i) = (sum of y over EARLIER rows of the same category + prior * a)
#                  / (count of those earlier rows + a),        with a = 1
#     prior      = mean of the whole target vector passed to fit
# Two separate channels carry test labels into the encoding:
#   (1) prior  -- contaminated for EVERY row, train rows included
#   (2) sum/count -- a test row's label is consumed by later rows of its category


def _encode_both(df, n_train, col="city", ycol="y"):
    tr, te = df.iloc[:n_train], df.iloc[n_train:]
    leaky = ce.CatBoostEncoder().fit_transform(
        pd.concat([tr[[col]], te[[col]]]), pd.concat([tr[ycol], te[ycol]]))
    enc = ce.CatBoostEncoder().fit(tr[[col]], tr[ycol])
    clean = np.concatenate([enc.transform(tr[[col]])[col].values,
                            enc.transform(te[[col]])[col].values])
    out = df.copy()
    out["AutoDP"] = leaky[col].values
    out["leakfree"] = clean
    out["split"] = ["train"] * n_train + ["TEST"] * (len(df) - n_train)
    return out


print("=" * 78)
print("PART A -- the mechanism, on 10 hand-checkable rows")
print("=" * 78)
CITY = ["hanoi", "hue", "hanoi", "hue", "hanoi", "hue", "hanoi", "hanoi", "hue", "hue"]

# A1: train says hanoi->1, hue->0. In TEST the pattern reverses.
#     Train mean and concatenated mean are both 0.5, so the prior channel is silent
#     and ONLY the ordered-statistic channel is visible.
print("\nA1  train: hanoi=1, hue=0   |   TEST: pattern reverses (concat mean == train mean)")
print(_encode_both(pd.DataFrame({"city": CITY, "y": [1, 0, 1, 0, 1, 0, 0, 0, 1, 1]}), 6)
      .to_string(index=False))
print("""    leak-free holds hanoi=0.8750 / hue=0.1250 on every test row: the encoder was
    frozen after train, so it still insists "hanoi means 1". It is wrong on these
    rows, and being wrong is correct -- nothing legitimate could know the reversal.

    AutoDP's SECOND test hanoi drops 0.8750 -> 0.7000, because the encoder has by
    then consumed the first test row's label (y=0). Same for hue: 0.1250 -> 0.3000.
    Check it: (3 + 0.5)/(3 + 1) = 0.8750, then (3 + 0.5)/(4 + 1) = 0.7000.""")

# A2: same features, test labels all 1, so the concatenated mean (0.8) differs from
#     the train mean (0.5) and the prior channel becomes visible.
print("\nA2  same features, TEST all y=1  (concat mean 0.8 != train mean 0.5)")
print(_encode_both(pd.DataFrame({"city": CITY, "y": [1, 0, 1, 0, 1, 0, 1, 1, 1, 1]}), 6)
      .to_string(index=False))
print("""    Now compare the TRAIN rows against A1: 0.500 -> 0.700, 0.750 -> 0.850,
    0.833 -> 0.900. The features were identical and no train label changed. The only
    thing that moved is the prior, which is the mean of the concatenated target.

    So the leak is NOT conditional on a category recurring within the test split. It
    reaches every row through the prior, including the first test row of a category,
    including categories absent from train, and including the training rows.""")

print()
print("=" * 78)
print("PART B -- what it is worth, on a feature with NO real signal, 20 seeds")
print("=" * 78)


def _rf(Xtr, ytr, Xte, yte):
    m = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    return (m.predict(Xte) == yte).mean()


rows = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    n, n_tr = 400, 320
    d = pd.DataFrame({"cat": rng.integers(0, n // 2, n).astype(str),
                      "y": rng.integers(0, 2, n)})           # target is a coin flip
    tr, te = d.iloc[:n_tr], d.iloc[n_tr:]
    Xa = ce.CatBoostEncoder().fit_transform(
        pd.concat([tr[["cat"]], te[["cat"]]]), pd.concat([tr["y"], te["y"]]))
    leak = _rf(Xa.head(n_tr), tr["y"], Xa.tail(n - n_tr), te["y"])
    e = ce.CatBoostEncoder().fit(tr[["cat"]], tr["y"])
    clean = _rf(e.transform(tr[["cat"]]), tr["y"], e.transform(te[["cat"]]), te["y"])
    rows.append((clean, leak))

r = np.array(rows)
d_ = r[:, 1] - r[:, 0]
print(f"  chance     : 0.500   (the feature is random ids, independent of the target)")
print(f"  leak-free  : {r[:, 0].mean():.3f} +/- {r[:, 0].std():.3f}")
print(f"  AutoDP CBE : {r[:, 1].mean():.3f} +/- {r[:, 1].std():.3f}")
print(f"  paired diff: {d_.mean():+.3f} +/- {d_.std():.3f}"
      f"   (AutoDP higher on {(d_ > 0).sum()}/20 seeds)")
print("""
  The honest reading: NO measurable accuracy advantage here. The paired difference is
  indistinguishable from zero, and single-seed runs of this demo swing from -0.02 to
  +0.15 -- quoting any one of them would be a measurement error.

  That is not a retraction of Part A. Part A is exact arithmetic: test labels
  demonstrably enter the encoded values. What Part B shows is that a leaked feature is
  not automatically an exploitable one -- the model still has to learn a mapping from
  the TRAIN encodings that happens to pay off on the differently-constructed test
  encodings, and on high-cardinality noise it does not.

  So the correct claim is about protocol validity, not about an inflated score:
  numbers produced this way are not measurements of generalisation, whichever
  direction they happen to move. Do not claim AutoDP's results are inflated BY the
  leak without measuring it on the actual datasets.

  Scope on the actual results: of the 8 datasets where AutoDP selected any
  preprocessing, CBE -- the only label-leaking encoder -- was chosen on exactly one
  (40663). FE/BE/OE were chosen on six more; those fit on concatenated features but
  never touch the target, so they are transductive, not label leakage.
""")
