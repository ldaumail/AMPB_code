#Train classifier to distinguish between EB and NS beta parameters
#Loic Daumail 01/21/2026
import numpy as np
import pandas as pd
import os.path as op
import os

# from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import permutation_test_score


#Load beta coeffs
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
#df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits','participants_ridgecv', 'combined', 'participant_betas_contrast-motionXstationary_combined_tracts.csv'))
df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits', 'participants_linearcv', 'combined','participant_betas_contrast-motionXstationary_combined_tracts.csv'))

participants = df["Participant"].unique()
hemis = ["L", "R"]
tracts = df["Tract"].unique()

n_subj = len(participants)
n_tract = len(tracts)

# Labels
y = np.array([1 if "EB" in p else 0 for p in participants])

def get_feature_matrix(df, hemi,selected_tracts):
    """
    Returns X: (n_subjects, n_tracts)
    """
    X = (
        df[(df["Hemisphere"] == hemi) & (df["Tract"].isin(selected_tracts))]
        .pivot_table(index="Participant", columns="Tract", values="Beta")
        .loc[participants, selected_tracts]
        .to_numpy()
        )

    return X


# def make_classifier(C=1):
#     return Pipeline([
#         ("scaler", StandardScaler()),
#         ("svm", SVC(
#             kernel="linear",
#             C=C,
#             probability=True,
#             class_weight="balanced"
#         ))
#     ])
from sklearn.svm import LinearSVC

def make_classifier(C=1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=1e-3, C=C))
    ])


def loso_evaluate(X, y, C):
    loo = LeaveOneOut()
    clf = make_classifier(C)

    y_true = []
    y_pred = []
    # y_prob = []
    scores = []

    for train_idx, test_idx in loo.split(X):
        clf.fit(X[train_idx], y[train_idx])

        y_true.append(y[test_idx][0])
        y_pred.append(clf.predict(X[test_idx])[0])
        # y_prob.append(clf.predict_proba(X[test_idx])[0, 1])
        # scores.append(clf.decision_function(X[test_idx]))
        scores.append(clf.decision_function(X[test_idx])[0])


        # scores = clf.decision_function(X[test_idx])
        # auc = roc_auc_score(y_true, scores) 

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # y_prob = np.array(y_prob)
    scores = np.array(scores)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "auc_flipped": roc_auc_score(y_true, -scores),
        "clf": clf
    }
    return results

results = {}

#train model per hemisphere
# df_no_mtfef = df[df["Tract"] != "MTxFEF"].copy()
selected_tracts = tracts[:3]  # or explicit list
for hemi in hemis:
    print(f"\n=== Hemisphere {hemi} ===")

    X = get_feature_matrix(df, hemi, selected_tracts)
    Cs = np.logspace(-2.3, 1, 10)
    for c, C in enumerate(Cs):
        res = loso_evaluate(X, y, C)
        
        print(f"Regularization parameter: {C:.3f}")
        print(f"Accuracy:           {res['accuracy']:.3f}")
        print(f"Balanced accuracy:  {res['balanced_accuracy']:.3f}")
        print(f"ROC AUC:            {res['roc_auc']:.3f}")
        print(f"ROC AUC Flipped:    {res['auc_flipped']:.3f}")

    results[hemi] = res

for hemi in hemis:
    weights = results[hemi]["clf"].named_steps["svm"].coef_[0]

    weight_df = pd.DataFrame({
        "Tract": tracts,
        "Weight": weights
    }).sort_values("Weight", ascending=False)

    print(f"\nTop tracts ({hemi}):")
    print(weight_df.head(5))


#Permutation testing
for hemi in hemis:
    print(f"\nPermutation test ({hemi})")

    X = get_feature_matrix(df, hemi)
    clf = make_classifier()

    score, perm_scores, pvalue = permutation_test_score(
        clf, X, y,
        cv=LeaveOneOut(),
        scoring="balanced_accuracy",
        n_permutations=5000,
        n_jobs=-1
    )

    print(f"Balanced accuracy: {score:.3f}")
    print(f"Permutation p-value: {pvalue:.4f}")

#Left hemisphere:
# Balanced accuracy: 0.533
# Permutation p-value: 0.5051

#Right hemisphere:
# Balanced accuracy: 0.200
# Permutation p-value: 0.9648


#==================== Visualize ===================#

## Add decision boundary plane

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def fit_final_model(X, y, C):
    clf = make_classifier(C)
    clf.fit(X, y)
    return clf

def get_decision_plane_raw(clf):
    scaler = clf.named_steps["scaler"]
    svm = clf.named_steps["svm"]

    w_scaled = svm.coef_[0]
    b_scaled = svm.intercept_[0]

    # Undo scaling
    w = w_scaled / scaler.scale_
    b = b_scaled - np.sum(w_scaled * scaler.mean_ / scaler.scale_)

    return w, b

def plot_3d_with_plane(
    df, participants, y, hemis, tracts, C, elev=10, azim=20
):
    assert len(tracts) == 3, "Decision plane only works for 3 features"

    fig = plt.figure(figsize=(14, 6))

    for i, hemi in enumerate(hemis):

        # ------------------------
        # Feature matrix
        # ------------------------

        X = get_feature_matrix(df, hemi, selected_tracts)
        C = C #0.5#10**(-2.3)
        clf = fit_final_model(X, y,C)
        w, b = get_decision_plane_raw(clf)

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        # ---- Scatter points ----
        ax.scatter(
            X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
            c="blue", marker="s", s=60, label="EB"
        )
        ax.scatter(
            X[y == 0, 0], X[y == 0, 1], X[y == 0, 2],
            c="red", marker="o", s=60, label="NS"
        )

        # ---- Decision plane ----
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
            np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
        )

        zz = (-w[0] * xx - w[1] * yy - b) / w[2]

        ax.plot_surface(
            xx, yy, zz,
            alpha=0.3,
            color="gray",
            edgecolor="none"
        )

        # ---- Labels & formatting ----
        ax.set_xlabel(tracts[0])
        ax.set_ylabel(tracts[1])
        ax.set_zlabel(tracts[2])
        ax.set_title(f"{hemi} Hemisphere")
        # ax.set_zlim(-4, 3)
        ax.view_init(elev=elev, azim=azim)

        # Clean panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        if i == 0:
            ax.legend()

    plt.suptitle("SVM decision planes per hemisphere", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(
        op.join(
            saveDir,
            "beta_weights_3d_linearreg_participants_combined_tracts_plane.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

selected_tracts = tracts[:3]  # example — must be exactly 3

plot_3d_with_plane(
    df=df,
    participants=participants,
    y=y,
    hemis=["L", "R"],
    tracts=selected_tracts,
    C = 0.027,
    elev=15,
    azim=-45
)

## No plane , just the data


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def compute_global_limits(df, participants, selected_tracts):
    mins, maxs = [], []

    for hemi in ["L", "R"]:
        # X = (
        #     df[(df["Hemisphere"] == hemi) & (df["Tract"].isin(selected_tracts))]
        #     .pivot_table(index="Participant", columns="Tract", values="MeanBeta")
        #     .loc[participants, selected_tracts]
        #     .to_numpy()
        # )
        X = get_feature_matrix(df, hemi, selected_tracts)
        mins.append(X.min(axis=0))
        maxs.append(X.max(axis=0))

    mins = np.min(np.vstack(mins), axis=0)
    maxs = np.max(np.vstack(maxs), axis=0)

    pad = 0.1 * (maxs - mins)
    return mins - pad, maxs + pad

def plot_3d_both_hemispheres(df, participants, y, selected_tracts):
    fig = plt.figure(figsize=(14, 6))
    mins, maxs = compute_global_limits(df, participants, selected_tracts)
    for i, hemi in enumerate(["L", "R"]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        X = get_feature_matrix(df, hemi, selected_tracts)

        # EB
        ax.scatter(
            X[y == 1, 0],
            X[y == 1, 1],
            X[y == 1, 2],
            c="blue",
            marker="s",
            s=60,
            edgecolor="k",
            label="EB",
            # depthshade=False
        )

        # NS
        ax.scatter(
            X[y == 0, 0],
            X[y == 0, 1],
            X[y == 0, 2],
            c="red",
            marker="o",
            s=60,
            edgecolor="k",
            label="NS",
            # depthshade=False
        )
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor("lightgray")
        ax.yaxis.pane.set_edgecolor("lightgray")
        ax.zaxis.pane.set_edgecolor("lightgray")
        
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        ax.view_init(elev=15, azim=-45)
        ax.set_xlabel(selected_tracts[0])
        ax.set_ylabel(selected_tracts[1])
        ax.set_zlabel(selected_tracts[2])

        ax.set_title(f"{hemi} Hemisphere", fontsize=14)

        # Same legend only once
        if i == 0:
            ax.legend()

    plt.tight_layout()
    # Saving
    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(op.join(saveDir, "beta_weights_3d_linearreg_combined_tracts.png"), dpi=300, bbox_inches='tight')

    plt.show()


selected_tracts = tracts[:3]  # or explicit list
# Labels
y = np.array([1 if "EB" in p else 0 for p in participants])
plot_3d_both_hemispheres(df, participants, y, selected_tracts)

# ====================== Use brf Kernel ====================#
import numpy as np
import pandas as pd
import os.path as op
import os

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import permutation_test_score


#Load beta coeffs
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits','ridgecv_loro_predicted_maps', 'combined', 'betas_contrast-motionXstationary_combined_tracts.csv'))


participants = df["Participant"].unique()
hemis = ["L", "R"]
tracts = df["Tract"].unique()

n_subj = len(participants)
n_tract = len(tracts)

# Labels
y = np.array([1 if "EB" in p else 0 for p in participants])

def get_feature_matrix(df, hemi, selected_tracts):
    """
    Returns X: (n_subjects, n_tracts)
    """
    # X = (
    #     df[df["Hemisphere"] == hemi]
    #     .pivot_table(index="Participant", columns="Tract", values="MeanBeta")
    #     .loc[participants]  # ensure same order
    #     .to_numpy()
    # )
    X = (
    df[(df["Hemisphere"] == hemi) & (df["Tract"].isin(selected_tracts))]
    .pivot_table(index="Participant", columns="Tract", values="MeanBeta")
    .loc[participants, selected_tracts]
    .to_numpy()
    )
    return X

def make_classifier(C=1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma='scale',
            probability=True,
            class_weight="balanced"
        ))
    ])


def loso_evaluate(X, y, C):
    loo = LeaveOneOut()
    clf = make_classifier(C)

    y_true = []
    y_pred = []
    y_prob = []
    scores = []

    for train_idx, test_idx in loo.split(X):
        clf.fit(X[train_idx], y[train_idx])

        y_true.append(y[test_idx][0])
        y_pred.append(clf.predict(X[test_idx])[0])
        y_prob.append(clf.predict_proba(X[test_idx])[0, 1])
        scores.append(clf.decision_function(X[test_idx]))


        # scores = clf.decision_function(X[test_idx])
        # auc = roc_auc_score(y_true, scores) 

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    scores = np.array(scores)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "auc_flipped": roc_auc_score(y_true, -scores),
        "clf": clf
    }
    return results

results = {}

#train model per hemisphere
# df_no_mtfef = df[df["Tract"] != "MTxFEF"].copy()
selected_tracts = tracts[:3]  # or explicit list
for hemi in hemis:
    print(f"\n=== Hemisphere {hemi} ===")

    X = get_feature_matrix(df, hemi, selected_tracts)
    Cs = np.logspace(-2.3, 1, 10)
    for c, C in enumerate(Cs):
        res = loso_evaluate(X, y, C)
        
        print(f"Regularization parameter: {C:.3f}")
        print(f"Accuracy:           {res['accuracy']:.3f}")
        print(f"Balanced accuracy:  {res['balanced_accuracy']:.3f}")
        print(f"ROC AUC:            {res['roc_auc']:.3f}")
        print(f"ROC AUC Flipped:    {res['auc_flipped']:.3f}")

    results[hemi] = res


#==========================================
#Plot with decision boundary for RBF kernel
#==========================================
from sklearn.svm import SVC
import numpy as np

def compute_decision_grid(X, clf, margin=0.5, n_pts=50):
    """
    X: (n_subj, 3), raw feature space
    clf: trained RBF SVM
    margin: padding beyond data range
    n_pts: grid resolution

    Returns:
        grid_x, grid_y, grid_z: mesh arrays of shape (n_pts, n_pts, n_pts)
        grid_vals: decision_function values on same grid
    """
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    spans = maxs - mins

    # Expand grid a bit
    mins -= margin * spans
    maxs += margin * spans

    xs = np.linspace(mins[0], maxs[0], n_pts)
    ys = np.linspace(mins[1], maxs[1], n_pts)
    zs = np.linspace(mins[2], maxs[2], n_pts)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')

    pts = np.vstack([
        grid_x.ravel(),
        grid_y.ravel(),
        grid_z.ravel()
    ]).T

    # decision_function for each grid point
    vals = clf.decision_function(pts).reshape((n_pts, n_pts, n_pts))

    return grid_x, grid_y, grid_z, vals

from skimage import measure

def extract_isosurface(grid_vals, level=0.0):
    """
    Extract vertices and faces for f(x) = level surface
    using marching cubes.
    """
    print(
    f"decision range = "
    f"[{grid_vals.min():.3f}, {grid_vals.max():.3f}]"
    )
    verts, faces, normals, _ = measure.marching_cubes(
        grid_vals, level=level, spacing=(1, 1, 1)
    )
    return verts, faces

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d_svm_surface(df, participants, y, hemis, tracts, C=1, elev=20, azim=30):
    assert len(tracts) == 3, "Need exactly 3 features for 3D"
    fig = plt.figure(figsize=(14, 6))
    for i, hemi in enumerate(hemis):
        X = get_feature_matrix(df, hemi, selected_tracts)
        # Train RBF SVM
        clf = make_classifier(C=C)
        clf.fit(X, y)

        # Compute decision grid
        # grid_x, grid_y, grid_z, grid_vals = compute_decision_grid(X, clf, n_pts=50)
        grid_x, grid_y, grid_z, grid_vals = compute_decision_grid( X, clf, margin=2.0, n_pts=60)


        # Extract surface
        verts, faces = extract_isosurface(grid_vals, level = np.percentile(grid_vals, 50))

        # Convert vertices back to real coordinates
        # (marching cubes used a unit grid; map back to spans)
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        spans = maxs - mins
        verts_recon = verts.copy()
        verts_recon[:, 0] = mins[0] + verts[:, 0] * (spans[0] / (grid_x.shape[0] - 1))
        verts_recon[:, 1] = mins[1] + verts[:, 1] * (spans[1] / (grid_y.shape[1] - 1))
        verts_recon[:, 2] = mins[2] + verts[:, 2] * (spans[2] / (grid_z.shape[2] - 1))

        # Plot
        
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        # Points
        ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c='blue', marker='s', label='EB')
        ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c='red', marker='o', label='NS')

        # Surface mesh
        mesh = Poly3DCollection(verts_recon[faces], alpha=0.3)
        mesh.set_facecolor('green')
        ax.add_collection3d(mesh)

        ax.set_xlabel(tracts[0]); ax.set_ylabel(tracts[1]); ax.set_zlabel(tracts[2])
        ax.set_title(f"{hemi} Hemisphere RBF Decision Surface")

        ax.view_init(elev=elev, azim=azim)
        plt.suptitle("SVM decision planes per hemisphere", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        saveDir = op.join(bids_path, "analysis", "plots")
        os.makedirs(saveDir, exist_ok=True)
        plt.savefig(op.join(
            saveDir,
            "beta_weights_3d_ridgereg_loro_combined_tracts_rbf_boundary.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

selected_tracts = tracts[:3]  # exactly 3 features

plot_3d_svm_surface(
    df=df,
    participants=participants,
    y=y,
    hemis=hemis,
    tracts=selected_tracts,
    C=0.005,
    elev=25,
    azim=45)


#==== Logistic regression
import numpy as np
import pandas as pd
import os.path as op
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LogisticRegression

#Load beta coeffs
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits','ridgecv_loro_predicted_maps', 'combined', 'betas_contrast-motionXstationary_combined_tracts.csv'))


participants = df["Participant"].unique()
hemis = ["L", "R"]
tracts = df["Tract"].unique()

n_subj = len(participants)
n_tract = len(tracts)

# Labels
y = np.array([1 if "EB" in p else 0 for p in participants])

def get_feature_matrix(df, hemi, selected_tracts):
    """
    Returns X: (n_subjects, n_tracts)
    """
    # X = (
    #     df[df["Hemisphere"] == hemi]
    #     .pivot_table(index="Participant", columns="Tract", values="MeanBeta")
    #     .loc[participants]  # ensure same order
    #     .to_numpy()
    # )
    X = (
    df[(df["Hemisphere"] == hemi) & (df["Tract"].isin(selected_tracts))]
    .pivot_table(index="Participant", columns="Tract", values="MeanBeta")
    .loc[participants, selected_tracts]
    .to_numpy()
    )
    return X

def make_classifier(C=1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000
        ))
    ])

def loso_evaluate(X, y, C):
    loo = LeaveOneOut()
    clf = make_classifier(C)

    y_true, y_pred, y_prob, scores = [], [], [], []

    for train_idx, test_idx in loo.split(X):
        clf.fit(X[train_idx], y[train_idx])

        y_true.append(y[test_idx][0])
        y_pred.append(clf.predict(X[test_idx])[0])
        y_prob.append(clf.predict_proba(X[test_idx])[0, 1])
        scores.append(clf.decision_function(X[test_idx])[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = np.array(scores)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "auc_flipped": roc_auc_score(y_true, -scores),
        "clf": clf
    }

#test model performance
selected_tracts = tracts[:3]  # or explicit list
results = {}
for hemi in hemis:
    print(f"\n=== Hemisphere {hemi} ===")

    X = get_feature_matrix(df, hemi, selected_tracts)
    Cs = np.logspace(-2.3, 1, 10)
    for c, C in enumerate(Cs):
        res = loso_evaluate(X, y, C)
        
        print(f"Regularization parameter: {C:.3f}")
        print(f"Accuracy:           {res['accuracy']:.3f}")
        print(f"Balanced accuracy:  {res['balanced_accuracy']:.3f}")
        print(f"ROC AUC:            {res['roc_auc']:.3f}")
        print(f"ROC AUC Flipped:    {res['auc_flipped']:.3f}")

    results[hemi] = res

## Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def compute_decision_grid(X, clf, margin=0.5, n_pts=50):
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    spans = maxs - mins

    mins -= margin * spans
    maxs += margin * spans

    xs = np.linspace(mins[0], maxs[0], n_pts)
    ys = np.linspace(mins[1], maxs[1], n_pts)
    zs = np.linspace(mins[2], maxs[2], n_pts)

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")

    pts = np.c_[grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
    vals = clf.decision_function(pts).reshape(grid_x.shape)

    return grid_x, grid_y, grid_z, vals

from skimage import measure

def extract_isosurface(grid_vals, level=0.0):
    print(
        f"decision range = "
        f"[{grid_vals.min():.3f}, {grid_vals.max():.3f}]"
    )

    if not (grid_vals.min() <= level <= grid_vals.max()):
        raise ValueError("Decision surface (0) not in grid range")

    verts, faces, normals, _ = measure.marching_cubes(
        grid_vals, level=level
    )
    return verts, faces

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d_logistic_surface(
    df, participants, y, hemis, tracts, C=1, elev=20, azim=30
):
    assert len(tracts) == 3
    fig = plt.figure(figsize=(14, 6))

    for i, hemi in enumerate(hemis):
        X = get_feature_matrix(df, hemi, selected_tracts)

        clf = make_classifier(C=C)
        clf.fit(X, y)

        grid_x, grid_y, grid_z, grid_vals = compute_decision_grid(
            X, clf, margin=1.5, n_pts=60
        )

        verts, faces = extract_isosurface(grid_vals, level=0.0)

        # Map vertices back to data space
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        spans = maxs - mins

        verts_real = verts.copy()
        verts_real[:, 0] = mins[0] + verts[:, 0] * spans[0] / (grid_x.shape[0] - 1)
        verts_real[:, 1] = mins[1] + verts[:, 1] * spans[1] / (grid_y.shape[1] - 1)
        verts_real[:, 2] = mins[2] + verts[:, 2] * spans[2] / (grid_z.shape[2] - 1)

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        ax.scatter(X[y==1,0], X[y==1,1], X[y==1,2], c="blue", label="EB")
        ax.scatter(X[y==0,0], X[y==0,1], X[y==0,2], c="red", label="NS")

        mesh = Poly3DCollection(verts_real[faces], alpha=0.3)
        mesh.set_facecolor("green")
        ax.add_collection3d(mesh)

        ax.set_title(f"{hemi} Hemisphere – Logistic Boundary")
        ax.set_xlabel(tracts[0])
        ax.set_ylabel(tracts[1])
        ax.set_zlabel(tracts[2])
        ax.view_init(elev=elev, azim=azim)

        if i == 0:
            ax.legend()

    plt.tight_layout()
    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(op.join(
        saveDir,
        "beta_weights_3d_ridgereg_loro_combined_tracts_logistic_boundary.png"
    ),
    dpi=300,
    bbox_inches="tight"
    )
    plt.show()

plot_3d_logistic_surface(
    df=df,
    participants=participants,
    y=y,
    hemis=hemis,
    tracts=tracts[:3],
    C=0.005,
    elev=25,
    azim=45
)


#=========================================================
#---------------------------------------------------------
## Other SVM cross validation implementation with inner cv
#---------------------------------------------------------
#=========================================================

import numpy as np
import pandas as pd
import os.path as op
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import permutation_test_score

# ----------------------------
# CLASSIFIER PIPELINE
# ----------------------------

def make_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=1e-3, max_iter=10000))
    ])


# ----------------------------
# NESTED LOOCV
# ----------------------------
def nested_loso(X, y):

    outer_cv = LeaveOneOut()
    inner_cv = LeaveOneOut()

    param_grid = {
        "svm__C": np.logspace(-2.3, 1, 10)
    }

    # Store fold-wise outputs
    y_true_all = []
    y_pred_all = []
    decision_scores_all = []
    best_Cs = []

    for train_idx, test_idx in outer_cv.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(
            make_pipeline(),
            param_grid,
            cv=inner_cv,
            scoring="balanced_accuracy",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_C = grid.best_params_["svm__C"]
        best_Cs.append(best_C)

        # Predictions
        y_pred = best_model.predict(X_test)
        decision_score = best_model.decision_function(X_test)

        # Store
        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])
        decision_scores_all.append(decision_score[0])

    # Convert to arrays
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    decision_scores_all = np.array(decision_scores_all)

    # Compute global metrics
    balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)

    roc_auc = roc_auc_score(y_true_all, decision_scores_all)
    roc_auc_flipped = roc_auc_score(y_true_all, -decision_scores_all)

    return {
        "balanced_accuracy": balanced_acc,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "roc_auc_flipped": roc_auc_flipped,
        "best_C_per_fold": best_Cs,
        "mean_best_C": np.mean(best_Cs),
        "sd_best_C": np.std(best_Cs),
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all
    }




#train model per hemisphere
# df_no_mtfef = df[df["Tract"] != "MTxFEF"].copy()
selected_tracts = tracts[:3]
results = {}

for hemi in hemis:
    print(f"\n=== Hemisphere {hemi} ===")

    X = get_feature_matrix(df, hemi, selected_tracts)

    res = nested_loso(X, y)

    print(f"Balanced accuracy:  {res['balanced_accuracy']:.3f}")
    print(f"Accuracy:           {res['accuracy']:.3f}")
    print(f"ROC AUC:            {res['roc_auc']:.3f}")
    print(f"ROC AUC Flipped:    {res['roc_auc_flipped']:.3f}")
    print(f"Mean best C:        {res['mean_best_C']:.4f}")
    print(f"SD best C:        {res['sd_best_C']:.4f}")

    results[hemi] = res

#=================================
# Visualize
#=================================

## Add decision boundary plane

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def make_classifier(C=1):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=1e-3, C=C))
    ])
def fit_final_model(X, y, C):
    clf = make_classifier(C)
    clf.fit(X, y)
    return clf

def get_decision_plane_raw(clf):
    scaler = clf.named_steps["scaler"]
    svm = clf.named_steps["svm"]

    w_scaled = svm.coef_[0]
    b_scaled = svm.intercept_[0]

    # Undo scaling
    w = w_scaled / scaler.scale_
    b = b_scaled - np.sum(w_scaled * scaler.mean_ / scaler.scale_)

    return w, b

def plot_3d_with_plane(
    df, participants, y, hemis, tracts, C, elev=10, azim=20
):
    assert len(tracts) == 3, "Decision plane only works for 3 features"

    # fig = plt.figure(figsize=(14, 6))
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)

    for i, hemi in enumerate(hemis):

        # ------------------------
        # Feature matrix
        # ------------------------

        X = get_feature_matrix(df, hemi, selected_tracts)
        C_val = C[i]
        clf = fit_final_model(X, y, C_val)
        w, b = get_decision_plane_raw(clf)

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        # ---- Scatter points ----
        ax.scatter(
            X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
            c="blue", marker="s", s=60, label="EB"
        )
        ax.scatter(
            X[y == 0, 0], X[y == 0, 1], X[y == 0, 2],
            c="red", marker="o", s=60, label="NS"
        )

        # ---- Decision plane ----
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
            np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
        )

        zz = (-w[0] * xx - w[1] * yy - b) / w[2]

        ax.plot_surface(
            xx, yy, zz,
            alpha=0.3,
            color="gray",
            edgecolor="none"
        )

        # ---- Labels & formatting ----
        ax.set_xlabel(tracts[0])
        ax.set_ylabel(tracts[1])
        ax.set_zlabel(tracts[2])
        ax.set_title(f"{hemi} Hemisphere")
        # ax.set_zlim(-4, 3)
        ax.view_init(elev=elev, azim=azim)

        # Clean panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        if i == 0:
            ax.legend()

    plt.suptitle("SVM decision planes per hemisphere", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.subplots_adjust(wspace=0.3, top=0.88)

    saveDir = op.join(bids_path, "analysis", "plots")
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(
        op.join(
            saveDir,
            "beta_weights_3d_linearreg_participants_combined_tracts_innercv.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

selected_tracts = tracts[:3]  # example — must be exactly 3

plot_3d_with_plane(
    df=df,
    participants=participants,
    y=y,
    hemis=["L", "R"],
    tracts=selected_tracts,
    C = np.array([0.018, 1.714]),
    elev=15,
    azim=-45
)

#-------------------------------------------
# Evaluate performance with average C values
#-------------------------------------------


def loso_evaluate(X, y, C):
    loo = LeaveOneOut()
    clf = make_classifier(C)

    y_true = []
    y_pred = []
    scores = []

    for train_idx, test_idx in loo.split(X):
        clf.fit(X[train_idx], y[train_idx])

        y_true.append(y[test_idx][0])
        y_pred.append(clf.predict(X[test_idx])[0])
        scores.append(clf.decision_function(X[test_idx])[0])


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = np.array(scores)

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "auc_flipped": roc_auc_score(y_true, -scores),
        "clf": clf
    }
    return results

results = {}

#train model per hemisphere
# df_no_mtfef = df[df["Tract"] != "MTxFEF"].copy()
selected_tracts = tracts[:3]  # or explicit list
Cs = np.array([0.018, 1.714])
for i, hemi in enumerate(hemis):
    print(f"\n=== Hemisphere {hemi} ===")

    X = get_feature_matrix(df, hemi, selected_tracts)
    
    res = loso_evaluate(X, y, Cs[i])
    
    print(f"Regularization parameter: {Cs[i]:.3f}")
    print(f"Accuracy:           {res['accuracy']:.3f}")
    print(f"Balanced accuracy:  {res['balanced_accuracy']:.3f}")
    print(f"ROC AUC:            {res['roc_auc']:.3f}")
    print(f"ROC AUC Flipped:    {res['auc_flipped']:.3f}")

    results[hemi] = res
