#Train classifier to distinguish between EB and NS beta parameters
#Loic Daumail 01/21/2026
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import os.path as op
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import LeaveOneOut
# from sklearn.preprocessing import StandardScaler


#Load beta coeffs
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits','ridgecv_loro_predicted_maps', 'combined', 'betas_contrast-motionXstationary_combined_tracts.csv'))

arr = (
    df
    .pivot_table(
        index="Subject",
        columns=["Hemisphere", "Tract"],
        values="MeanBeta"
    )
    .sort_index(axis=1)
    .to_numpy()
)

n_subj = df["Subject"].nunique()
n_hemi = df["Hemisphere"].nunique()
n_tract = df["Tract"].nunique()

arr = arr.reshape(n_subj, n_hemi, n_tract)

# trained_coefs shape:
# (n_runs, n_tracts, n_subjects, n_hemi)

# Average across runs
betas_avg = np.nanmean(trained_coefs, axis=0)  
# shape: (n_tracts, n_subjects, n_hemi)

# Reorder into subject × features
X = np.concatenate([
    betas_avg[:, :, 0].T,   # Left hemi
    betas_avg[:, :, 1].T    # Right hemi
], axis=1)                  # (n_subj, 2*n_tracts)

# Labels
y = np.array([1 if "EB" in p else 0 for p in participants])





loo = LeaveOneOut()

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", C=1.0, probability=True))
])

# clf = Pipeline([
#     ("scaler", StandardScaler()),
#     ("logreg", LogisticRegression(
#         penalty="l2",
#         solver="liblinear",
#         C=1.0
#     ))
# ])

y_true, y_pred, y_prob = [], [], []

for train_idx, test_idx in loo.split(X):
    clf.fit(X[train_idx], y[train_idx])

    y_true.append(y[test_idx][0])
    y_pred.append(clf.predict(X[test_idx])[0])
    y_prob.append(clf.predict_proba(X[test_idx])[0, 1])

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

print(f"LOSO Accuracy: {acc:.3f}")
print(f"LOSO ROC AUC:  {auc:.3f}")

#Inspect tract performance

weights = clf.named_steps["svm"].coef_[0]

weights_L = weights[:n_tracts]
weights_R = weights[n_tracts:]

# Plot or tabulate


#Statistical validation: Permutation test

from sklearn.model_selection import permutation_test_score

score, perm_scores, pvalue = permutation_test_score(
    clf, X, y,
    cv=loo,
    scoring="accuracy",
    n_permutations=5000
)

print(f"Permutation p-value: {pvalue:.4f}")


#Neural network option:
# from sklearn.neural_network import MLPClassifier

# mlp = Pipeline([
#     ("scaler", StandardScaler()),
#     ("mlp", MLPClassifier(
#         hidden_layer_sizes=(8,),
#         activation="relu",
#         alpha=1e-2,
#         max_iter=5000,
#         random_state=0
#     ))
# ])
