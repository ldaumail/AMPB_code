#Train classifier to distinguish between EB and NS beta parameters
#Loic Daumail 10/15/2026



#============================================================================
#Claude's improvement:
#============================================================================
import os
import os.path as op
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


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

def make_pipeline(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=C, penalty="l2", loss="squared_hinge",
                          dual=False, tol=1e-3, max_iter=10000)),
    ])


# ------------------------------------------------------------
# 1. One repetition of nested CV (stratified 5-fold outer,
#    repeated stratified 5-fold inner, AUC for C selection)
# ------------------------------------------------------------
#stratified guarantees that we have both classes of groups EB and NS in the held out fold. (This wasn't the case in previous version)
C_GRID = np.logspace(-2.3, 1, 10)
def nested_cv_once(X, y, seed, n_outer=5, n_inner=5, n_inner_repeats=5,
                   n_jobs=1):
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = RepeatedStratifiedKFold(n_splits=n_inner,
                                    n_repeats=n_inner_repeats,
                                    random_state=seed)

    scores = np.zeros(len(y))      # pooled out-of-fold decision values
    preds = np.zeros(len(y), dtype=int)
    best_Cs = []

    for train_idx, test_idx in outer.split(X, y):
        gs = GridSearchCV(
            make_pipeline(),
            {"svm__C": C_GRID},
            cv=inner,
            scoring="roc_auc",     # continuous, works with k-fold
            n_jobs=n_jobs,
        )
        gs.fit(X[train_idx], y[train_idx])
        best_Cs.append(gs.best_params_["svm__C"])
        scores[test_idx] = gs.best_estimator_.decision_function(X[test_idx])
        preds[test_idx] = gs.best_estimator_.predict(X[test_idx])

    return {
        "auc": roc_auc_score(y, scores),
        "bacc": balanced_accuracy_score(y, preds),
        "best_Cs": best_Cs,
    }


# ------------------------------------------------------------
# 2. Repeat the outer split many times (different shuffles)
# ------------------------------------------------------------
# This section checks how well the model generalizes across subjects with cross validations

def nested_cv_repeated(X, y, n_repeats=20, base_seed=0, n_jobs=1, **kw):
    runs = [nested_cv_once(X, y, seed=base_seed + r, n_jobs=n_jobs, **kw)
            for r in range(n_repeats)]
    all_Cs = np.concatenate([r["best_Cs"] for r in runs])
    return {
        "auc_mean": np.mean([r["auc"] for r in runs]),
        "auc_sd": np.std([r["auc"] for r in runs]),
        "bacc_mean": np.mean([r["bacc"] for r in runs]),
        "bacc_sd": np.std([r["bacc"] for r in runs]),
        "best_Cs": all_Cs,
        "median_C": np.median(all_Cs),
        "geomean_C": np.exp(np.mean(np.log(all_Cs))),
    }


# ------------------------------------------------------------
# 3. Permutation test: shuffle y, rerun the WHOLE nested procedure
# ------------------------------------------------------------
def _one_perm(X, y, seed, n_repeats, **kw):
    rng = np.random.default_rng(10_000 + seed)
    y_perm = rng.permutation(y)    # preserves class balance
    res = nested_cv_repeated(X, y_perm, n_repeats=n_repeats,
                             base_seed=seed, n_jobs=1, **kw)
    return res["auc_mean"], res["bacc_mean"]


def permutation_test(X, y, n_perm=500, n_repeats=2, n_jobs=-1, **kw):
    # Observed statistic computed with the same procedure as the null
    obs = nested_cv_repeated(X, y, n_repeats=n_repeats, base_seed=0,
                             n_jobs=1, **kw)
    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(_one_perm)(X, y, s, n_repeats, **kw) for s in range(n_perm)
    ))
    null_auc, null_bacc = null[:, 0], null[:, 1]

    p_auc = (1 + np.sum(null_auc >= obs["auc_mean"])) / (n_perm + 1)
    p_bacc = (1 + np.sum(null_bacc >= obs["bacc_mean"])) / (n_perm + 1)
    return {"obs_auc": obs["auc_mean"], "obs_bacc": obs["bacc_mean"],
            "p_auc": p_auc, "p_bacc": p_bacc,
            "null_auc": null_auc, "null_bacc": null_bacc}


# ------------------------------------------------------------
# 4. Final C for a deployable model: tune on ALL data once and
#    inspect the full score-vs-C curve
# ------------------------------------------------------------
#This determines the best C using all the subjects data.
def final_C_curve(X, y, n_splits=5, n_repeats=50, seed=0):
    gs = GridSearchCV(
        make_pipeline(), {"svm__C": C_GRID},
        cv=RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed),
        scoring="roc_auc", n_jobs=-1,
    )
    gs.fit(X, y)
    return (gs.best_params_["svm__C"],
            gs.cv_results_["mean_test_score"],
            gs.cv_results_["std_test_score"])


# ------------------------------------------------------------
# Run per hemisphere
# ------------------------------------------------------------
bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
#df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits','participants_ridgecv', 'combined', 'participant_betas_contrast-motionXstationary_combined_tracts.csv'))
df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits', 'pyAFQ3beta_wb_participants_linearreg_ptlocal', 'beta_contrast-motionXstationary_combined_1frac.csv'))

participants = df["Participant"].unique()
hemis = ["L", "R"]
#tracts = ["PTR", "SuperiorLongitudinal", "VerticalOccipital"] #df["Tract"].unique()
tracts = ["PTR", "SuperiorLongitudinal", "InferiorFrontooccipital"]
n_subj = len(participants)
n_tract = len(tracts)

# Labels
y = np.array([1 if "EB" in p else 0 for p in participants])

tract_matches = tracts.copy() #["PTR", "SuperiorLongitudinal","InferiorFrontooccipital", "VerticalOccipital"]
df_use = df[~((df["Group"] == "NS") & (df["Localizer"] == "mtlocal")) & df["Tract"].isin(tract_matches)].copy()
selected_tracts = tracts.copy() #[[0, 2]]

for hemi in hemis:
    print(f"\n=== Hemisphere {hemi} ===")
    X = get_feature_matrix(df_use, hemi, selected_tracts)

    res = nested_cv_repeated(X, y, n_repeats=20)
    print(f"Nested CV AUC:  {res['auc_mean']:.3f} ± {res['auc_sd']:.3f}")
    print(f"Nested CV bAcc: {res['bacc_mean']:.3f} ± {res['bacc_sd']:.3f}")
    print(f"Chosen C: median={res['median_C']:.3f}, "
          f"geomean={res['geomean_C']:.3f}")

    perm = permutation_test(X, y, n_perm=500, n_repeats=2)
    print(f"Permutation p (AUC):  {perm['p_auc']:.3f}")
    print(f"Permutation p (bAcc): {perm['p_bacc']:.3f}")

    best_C, mean_curve, sd_curve = final_C_curve(X, y)
    print(f"Final-model C: {best_C:.3f}")
    print("AUC by C:", np.round(mean_curve, 3))

#--------------------------------------------------------------------------------
# Visualize 3 D Results
#-------------------------------------------------------------------------------
## Add decision boundary plane
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"

def fit_final_model(X, y, C):
    clf = make_pipeline(C)
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
def plot_3d_with_plane(df, y, hemis, tracts, Cs, elev, azim,
                       save_path, savefilename, pad=0.05):
    assert len(tracts) == 3, "Decision plane only works for 3 features"
    fig = plt.figure(figsize=(14, 6))

    for i, hemi in enumerate(hemis):
        X = get_feature_matrix(df, hemi, tracts)          # uses the argument
        clf = fit_final_model(X, y, Cs[i])
        w, b = get_decision_plane_raw(clf)

        # Cross-validated predictions, to flag points the model gets wrong
        cv_pred = cross_val_predict(
            make_pipeline(Cs[i]), X, y,
            cv=StratifiedKFold(5, shuffle=True, random_state=0)
        )
        wrong = cv_pred != y

        ax = fig.add_subplot(1, 2, i + 1, projection="3d")

        for lab, color, marker, name in [(1, "blue", "s", "EB"),
                                         (0, "red", "o", "NS")]:
            m = y == lab
            ax.scatter(X[m & ~wrong, 0], X[m & ~wrong, 1], X[m & ~wrong, 2],
                       c=color, marker=marker, s=60, label=name)
            ax.scatter(X[m & wrong, 0], X[m & wrong, 1], X[m & wrong, 2],
                       c=color, marker=marker, s=60,
                       edgecolors="k", linewidths=2,
                       label=f"{name} (CV error)")

        # Plane clipped to the (padded) data range
        lo, hi = X.min(0), X.max(0)
        span = hi - lo
        lo, hi = lo - pad * span, hi + pad * span

        if abs(w[2]) > 1e-8:
            xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 25),
                                 np.linspace(lo[1], hi[1], 25))
            zz = (-w[0] * xx - w[1] * yy - b) / w[2]
            zz = np.where((zz >= lo[2]) & (zz <= hi[2]), zz, np.nan)
            ax.plot_surface(xx, yy, zz, alpha=0.3, color="gray",
                            edgecolor="none")

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])

        ax.view_init(elev=elev[i], azim=azim[i])
        ax.set_xlabel(tracts[0], labelpad=10, fontsize=18, fontweight="bold")
        ax.set_ylabel(tracts[1], labelpad=10, fontsize=18, fontweight="bold")
        ax.set_zlabel(tracts[2], labelpad=10, fontsize=18, fontweight="bold")
        ax.set_title(f"{hemi} Hemisphere (C = {Cs[i]:.3g})")

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.set_major_locator(mticker.MaxNLocator(4))
            axis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))
        ax.tick_params(labelsize=14)

        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("SVM decision planes per hemisphere (fit on all subjects)",
                 fontsize=16)
    plt.subplots_adjust(wspace=0.1, right=0.9)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for fmt in ("png", "svg"):
            plt.savefig(op.join(save_path, f"{savefilename}.{fmt}"), dpi=600, bbox_inches="tight")

    plt.show()

C_left = 4.299
C_right = 10
plot_3d_with_plane(
    df=df_use, y=y, hemis=["L", "R"], tracts=tracts, #[tracts[i] for i in (0, 1, 3)],
    Cs=[C_left, C_right],                 # from a stated selection rule
    elev=[5, 0], azim=[-90, -90],
    save_path=op.join(bids_path, "analysis", "plots"),
    savefilename =  "pyAFQ3beta_wb_beta_weights_3d_linear_svm_EBptlocal_NSptlocal_PTR-SLF-IFOF"
)   


#==============================================
### Visualize boundary in only 2 dimensions (3rd dimension fixed)
#==============================================

import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"

def plot_2d_slice_of_3d_svm(df, y, hemis, tracts, Cs, dims=(0, 1),
                            z_value="mean", show_z_spread=True,
                            save_path=None, savefilename=None,  pad=0.05):
    """
    Fit the SVM on ALL features in `tracts` (e.g. 3), then plot only the two
    features in `dims`, with the decision boundary of the 3D model sliced at
    a fixed value of the remaining feature(s).

    z_value: "mean", "median", or a number (in raw units), or a list with one
             value per hemisphere.
    show_z_spread: also draw the boundary at z = mean ± 1 SD (dashed), to show
                   how much the 3D boundary moves along the hidden dimension.
    """
    n_feat = len(tracts)
    assert n_feat >= 3 and len(dims) == 2, "Fit on >=3 features, plot 2 of them"
    hidden = [k for k in range(n_feat) if k not in dims]

    fig, axes = plt.subplots(1, len(hemis), figsize=(14, 6),
                             constrained_layout=True)

    for i, hemi in enumerate(hemis):
        ax = axes[i]

        # ---- 3D model (same features, same C as your 3D analysis) ----
        X = get_feature_matrix(df, hemi, tracts)
        clf = fit_final_model(X, y, Cs[i])
        w, b = get_decision_plane_raw(clf)

        d0, d1 = dims
        if abs(w[d1]) < 1e-8:
            print(f"{hemi}: boundary is nearly vertical in this view "
                  f"(w[{d1}] ~ 0); line not drawn.")

        # ---- Value(s) at which hidden feature(s) are held ----
        if z_value == "mean":
            z0 = X[:, hidden].mean(axis=0)
        elif z_value == "median":
            z0 = np.median(X[:, hidden], axis=0)
        else:
            zv = z_value[i] if np.ndim(z_value) > 0 else z_value
            z0 = np.full(len(hidden), zv, dtype=float)
        z_sd = X[:, hidden].std(axis=0)

        def boundary(x_vals, z):
            offset = b + np.dot(w[hidden], z)
            return (-w[d0] * x_vals - offset) / w[d1]

        # ---- Points: the two plotted dimensions ----
        ax.scatter(X[y == 1, d0], X[y == 1, d1],
                   c="#9ecae1", marker="s", s=60, label="EB")
        ax.scatter(X[y == 0, d0], X[y == 0, d1],
                   c="#c45100", marker="o", s=60, label="NS")

        # ---- Sliced decision boundary ----
        lo, hi = X[:, [d0, d1]].min(0), X[:, [d0, d1]].max(0)
        span = hi - lo
        lo, hi = lo - pad * span, hi + pad * span
        x_vals = np.linspace(lo[0], hi[0], 200)

        if abs(w[d1]) >= 1e-8:
            ax.plot(x_vals, boundary(x_vals, z0), color="black", linewidth=2,
                    label=f"Boundary ({tracts[hidden[0]]} = mean)"
                    if z_value == "mean" else "Boundary (slice)")
            if show_z_spread:
                for sgn in (-1, 1):
                    ax.plot(x_vals, boundary(x_vals, z0 + sgn * z_sd),
                            color="black", linewidth=1, linestyle="--",
                            alpha=0.6,
                            label="± 1 SD of hidden feature" if sgn == -1 else None)

        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])   # keeps steep lines from stretching axes

        sns.despine(ax=ax, top=True, right=True)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)

        ax.set_xlabel(tracts[d0], fontsize=18, fontweight="bold")
        ax.set_ylabel(tracts[d1], fontsize=18, fontweight="bold")
        ax.set_title(f"{hemi} Hemisphere (C = {Cs[i]:.3g})")
        ax.legend()
        plt.setp(ax.get_xticklabels(), fontsize=14, fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontsize=14, fontweight="bold")

    plt.suptitle(f"3D linear SVM, 2D slice ({tracts[dims[0]]} vs {tracts[dims[1]]})",
                 fontsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for fmt in ("png", "svg"):
            plt.savefig(op.join(save_path, f"{savefilename}.{fmt}"), dpi=600, bbox_inches="tight")
    plt.show()


plot_2d_slice_of_3d_svm(
    df=df_use,
    y=y,
    hemis=["L", "R"],
    tracts=tracts[:3],               # the SAME 3 tracts used in the 3D analysis
    Cs=[C_left, C_right],               # the same Cs as the 3D model
    dims=(0,2),                     # which two features to display
    z_value="median",
    save_path=op.join(bids_path, "analysis", "plots"),
    savefilename =  "pyAFQ3beta_wb_beta_weights_3d_linear_svm_2d_slice_EBptlocal_NSptlocal_ptr-ifof"
)


#==============================================
### Visualize boundary line with only 2 tracts (rerunning 2D SVC)
#==============================================

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"

def fit_final_model(X, y, C):
    clf = make_pipeline(C)
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
    # Undo scaling
    w = w_scaled / scaler.scale_
    b = b_scaled - np.sum(w_scaled * scaler.mean_ / scaler.scale_)

    return w, b

def plot_2d_with_plane(
    df, participants, y, hemis, tracts, C, save_path, savefilename
):
    assert len(tracts) == 2, "Decision boundary requires exactly 2 features"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for i, hemi in enumerate(hemis):

        # ------------------------
        # Feature matrix
        # ------------------------
        X = get_feature_matrix(df, hemi, tracts)
        C_val = C[i]
        clf = fit_final_model(X, y, C_val)
        w, b = get_decision_plane_raw(clf)

        ax = axes[i]

        # ---- Scatter points ----
        ax.scatter(
            X[y == 1, 0], X[y == 1, 1],
            c="#9ecae1", marker="s", s=60, label="EB"
        )
        ax.scatter(
            X[y == 0, 0], X[y == 0, 1],
            c="#c45100", marker="o", s=60, label="NS"
        )

        # ---- Decision boundary ----
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        y_vals = (-w[0] * x_vals - b) / w[1]

        ax.plot(
            x_vals,
            y_vals,
            color="black",
            linewidth=2,
            label="Decision boundary"
        )
        #1 remove box
        sns.despine(ax=ax, top=True, right=True)

        # 2. Make the remaining axes (Left and Bottom) thick
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['bottom'].set_linewidth(2.5)

        # ---- Optional: margin lines ----
        # margin = 1 / np.linalg.norm(w)
        # y_margin_up = (-w[0] * x_vals - b + 1) / w[1]
        # y_margin_down = (-w[0] * x_vals - b - 1) / w[1]

        # ax.plot(x_vals, y_margin_up, "k--", linewidth=1)
        # ax.plot(x_vals, y_margin_down, "k--", linewidth=1)

        # ---- Formatting ----
        ax.set_xlabel(tracts[0],fontsize=18,fontweight='bold')
        ax.set_ylabel(tracts[1],fontsize=18,fontweight='bold')
        ax.set_title(f"{hemi} Hemisphere (C = {C_val})")
        ax.legend()
        plt.setp(ax.get_yticklabels(),fontsize=18,fontweight='bold')

    plt.suptitle("Linear SVM decision boundaries", fontsize=16)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for fmt in ("png", "svg"):
            plt.savefig(op.join(save_path, f"{savefilename}.{fmt}"), dpi=600, bbox_inches="tight")
    plt.show()

C_left = 4.299
C_right = 4.299
plot_2d_with_plane(
    df=df_use,
    participants=participants,
    y=y,
    hemis=["L", "R"],
    tracts=selected_tracts,
    C = np.array([C_left, C_right]),
    save_path = op.join(bids_path, "analysis", "plots"),
    savefilename = "pyAFQ3beta_wb_beta_weights_2d_linear_svm_PTR-IFOF"
)


# #=========================================================
# #---------------------------------------------------------
# ## NEW WAY: SVM cross validation implementation with nested inner cv
# #---------------------------------------------------------
# #=========================================================

# import numpy as np
# import pandas as pd
# import os.path as op
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import LeaveOneOut, GridSearchCV
# from sklearn.metrics import (
#     balanced_accuracy_score,
#     accuracy_score,
#     roc_auc_score
# )
# from sklearn.model_selection import permutation_test_score
# #Load beta coeffs
# bids_path = op.join('/Users', 'ldaumail3', 'Documents', 'research', 'ampb_mt_tractometry_analysis', 'ampb')
# #df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits','participants_ridgecv', 'combined', 'participant_betas_contrast-motionXstationary_combined_tracts.csv'))
# df = pd.read_csv(op.join(bids_path, 'analysis','diff2func_model_fits', 'pyAFQ3beta_targ_participants_linearreg_ptlocal', 'beta_contrast-motionXstationary_combined_1frac.csv'))

# participants = df["Participant"].unique()
# hemis = ["L", "R"]
# tracts = df["Tract"].unique()

# n_subj = len(participants)
# n_tract = len(tracts)

# # Labels
# y = np.array([1 if "EB" in p else 0 for p in participants])

# def get_feature_matrix(df, hemi,selected_tracts):
#     """
#     Returns X: (n_subjects, n_tracts)
#     """
#     X = (
#         df[(df["Hemisphere"] == hemi) & (df["Tract"].isin(selected_tracts))]
#         .pivot_table(index="Participant", columns="Tract", values="Beta")
#         .loc[participants, selected_tracts]
#         .to_numpy()
#         )

#     return X
# # ----------------------------
# # CLASSIFIER PIPELINE
# # ----------------------------

# def make_pipeline(C=1):
#     return Pipeline([
#         ("scaler", StandardScaler()),
#         ("svm", LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=1e-3, max_iter=10000, C = C))
#     ])


# # ----------------------------
# # NESTED LOOCV
# # ----------------------------
# def nested_loso(X, y):

#     outer_cv = LeaveOneOut()
#     inner_cv = LeaveOneOut()

#     param_grid = {
#         "svm__C": np.logspace(-2.3, 1, 10)
#     }

#     # Store fold-wise outputs
#     y_true_all = []
#     y_pred_all = []
#     decision_scores_all = []
#     best_Cs = []

#     for train_idx, test_idx in outer_cv.split(X):

#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]

#         grid = GridSearchCV(
#             make_pipeline(),
#             param_grid,
#             cv=inner_cv,
#             scoring="balanced_accuracy",
#             n_jobs=-1
#         )

#         grid.fit(X_train, y_train)

#         best_model = grid.best_estimator_
#         best_C = grid.best_params_["svm__C"]
#         best_Cs.append(best_C)

#         # Predictions
#         y_pred = best_model.predict(X_test)
#         decision_score = best_model.decision_function(X_test)

#         # Store
#         y_true_all.append(y_test[0])
#         y_pred_all.append(y_pred[0])
#         decision_scores_all.append(decision_score[0])

#     # Convert to arrays
#     y_true_all = np.array(y_true_all)
#     y_pred_all = np.array(y_pred_all)
#     decision_scores_all = np.array(decision_scores_all)

#     # Compute global metrics
#     balanced_acc = balanced_accuracy_score(y_true_all, y_pred_all)
#     acc = accuracy_score(y_true_all, y_pred_all)

#     roc_auc = roc_auc_score(y_true_all, decision_scores_all)
#     roc_auc_flipped = roc_auc_score(y_true_all, -decision_scores_all)

#     return {
#         "balanced_accuracy": balanced_acc,
#         "accuracy": acc,
#         "roc_auc": roc_auc,
#         "roc_auc_flipped": roc_auc_flipped,
#         "best_C_per_fold": best_Cs,
#         "mean_best_C": np.exp(np.mean(np.log(best_Cs))), #geometric mean is better than arithmetic one in log space
#         "sd_best_C": np.std(best_Cs),
#         "y_true_all": y_true_all,
#         "y_pred_all": y_pred_all
#     }




# #1. train model per hemisphere: find the best Cs from each crossvalidation iteration
# #do this if you want to remove a tract:
# # df_no_mtptsts1 = df[df["Tract"] != "MTxPTxSTS1"].copy()

# df_no_NS_ptlocal = df[
#     ~((df["Group"] == "NS") & (df["Localizer"] == "ptlocal"))
# ].copy() #    & (df["Tract"] != "MTxPTxSTS1")
# selected_tracts = tracts[[0,2]]
# # selected_tracts = tracts[:3]
# results = {}

# for hemi in hemis:
#     print(f"\n=== Hemisphere {hemi} ===")

#     X = get_feature_matrix(df_no_NS_ptlocal, hemi, selected_tracts)

#     res = nested_loso(X, y)

#     print(f"Balanced accuracy:  {res['balanced_accuracy']:.3f}")
#     print(f"Accuracy:           {res['accuracy']:.3f}")
#     print(f"ROC AUC:            {res['roc_auc']:.3f}")
#     print(f"ROC AUC Flipped:    {res['roc_auc_flipped']:.3f}")
#     print(f"Mean best C:        {res['mean_best_C']:.4f}")
#     print(f"SD best C:        {res['sd_best_C']:.4f}")

#     results[hemi] = res


# #=================================
# # Visualize
# #=================================

# ## Add decision boundary plane

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
# import os

# # def make_pipeline(C=1):
# #     return Pipeline([
# #         ("scaler", StandardScaler()),
# #         ("svm", LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=1e-3, C=C))
# #     ])

# def fit_final_model(X, y, C):
#     clf = make_pipeline(C)
#     clf.fit(X, y)
#     return clf

# def get_decision_plane_raw(clf):
#     scaler = clf.named_steps["scaler"]
#     svm = clf.named_steps["svm"]

#     w_scaled = svm.coef_[0]
#     b_scaled = svm.intercept_[0]

#     # Undo scaling
#     w = w_scaled / scaler.scale_
#     b = b_scaled - np.sum(w_scaled * scaler.mean_ / scaler.scale_)

#     return w, b

# import matplotlib.ticker as mticker
# def plot_3d_with_plane(
#     df, participants, y, hemis, tracts, C, elev=10, azim=20
# ):
#     assert len(tracts) == 3, "Decision plane only works for 3 features"

#     fig = plt.figure(figsize=(14, 6))

#     for i, hemi in enumerate(hemis):

#         # ------------------------
#         # Feature matrix
#         # ------------------------

#         X = get_feature_matrix(df, hemi, selected_tracts)
#         C_val = C[i]
#         clf = fit_final_model(X, y, C_val)
#         w, b = get_decision_plane_raw(clf)

#         ax = fig.add_subplot(1, 2, i + 1, projection="3d")

#         # ---- Scatter points ----
#         ax.scatter(
#             X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
#             c="blue", marker="s", s=60, label="EB"
#         )
#         ax.scatter(
#             X[y == 0, 0], X[y == 0, 1], X[y == 0, 2],
#             c="red", marker="o", s=60, label="NS"
#         )

#         # ---- Decision plane ----
#         xx, yy = np.meshgrid(
#             np.linspace(X[:, 0].min(), X[:, 0].max(), 20),
#             np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
#         )

#         zz = (-w[0] * xx - w[1] * yy - b) / w[2]

#         ax.plot_surface(
#             xx, yy, zz,
#             alpha=0.3,
#             color="gray",
#             edgecolor="none"
#         )
#         if i == 0:
#             ax.view_init(elev=elev[0], azim=azim[0])
#         if i == 1:
#             ax.view_init(elev=elev[1], azim=azim[1])
#         # ---- Labels & formatting ----
#         ax.set_xlabel(tracts[0], labelpad=10,fontsize=18, fontweight='bold')
#         ax.set_ylabel(tracts[1], labelpad=10,fontsize=18, fontweight='bold')
#         ax.set_zlabel(tracts[2], labelpad=15,fontsize=18, fontweight='bold')
#         ax.set_title(f"{hemi} Hemisphere")
#         # ax.set_zlim(-4, 3)
        

#         # Clean panes
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False

#         ax.spines['left'].set_linewidth(2) #axis thickness
#         ax.spines['right'].set_linewidth(2)
#         ax.spines['bottom'].set_linewidth(2) #axis thickness
#         #plt.setp(ax.get_yticklabels(),fontsize=18,fontweight='bold')

#         ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.1f}'))
#         ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.1f}'))
#         ax.zaxis.set_major_formatter(mticker.StrMethodFormatter('{x:.1f}'))
#         # Helper to bold and hide every other label
#         def format_every_other(labels):
#             for i, label in enumerate(labels):
#                 if i % 2 == 0:  # Keep even indices
#                     label.set_visible(False)
#                 else:           # Hide odd indices
#                     label.set_fontweight('bold')
#                     label.set_fontsize(14)
                    

#         format_every_other(ax.xaxis.get_ticklabels())
#         format_every_other(ax.yaxis.get_ticklabels())
#         format_every_other(ax.zaxis.get_ticklabels())
            
        
#         ax.legend()
#         if i == 1:
#             # Sometimes the right plot needs more space on the right side
#             ax.set_zlabel(tracts[2], labelpad=0)

#     plt.suptitle("SVM decision planes per hemisphere", fontsize=16)
#     # plt.tight_layout(rect=[0, 0, 1, 0.95])
#     # plt.subplots_adjust(wspace=0.3, top=0.88)
#     plt.subplots_adjust(wspace=0.1, right=0.9)

#     saveDir = op.join(bids_path, "analysis", "plots")
#     os.makedirs(saveDir, exist_ok=True)
#     plt.savefig(
#         op.join(
#             saveDir,
#             "pyAFQ3beta_targ_beta_weights_3d_linreg_participants_combined_tracts_innercv.png"
#         ),
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.show()

# selected_tracts = tracts[:3]  # example — must be exactly 3
# df_no_PTxSTS1 = df[
#     ~((df["Group"] == "NS") & (df["Localizer"] == "ptlocal"))
# ].copy()
# plot_3d_with_plane(
#     df=df_no_PTxSTS1,
#     participants=participants,
#     y=y,
#     hemis=["L", "R"],
#     tracts=selected_tracts,
#     C = np.array([0.005011872336272724, 1.5914209198620377]),
#     elev=np.array([15,15]), #15
#     azim=np.array([-40,-40])
# )
# #np.array([0.0183, 1.7138]),
# #-------------------------------------------
# # Evaluate performance with average C values
# #-------------------------------------------


# def loso_evaluate(X, y, C):
#     loo = LeaveOneOut()
#     clf = make_pipeline(C)

#     y_true = []
#     y_pred = []
#     scores = []

#     for train_idx, test_idx in loo.split(X):
#         clf.fit(X[train_idx], y[train_idx])

#         y_true.append(y[test_idx][0])
#         y_pred.append(clf.predict(X[test_idx])[0])
#         scores.append(clf.decision_function(X[test_idx])[0])


#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     scores = np.array(scores)

#     results = {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
#         "roc_auc": roc_auc_score(y_true, scores),
#         "auc_flipped": roc_auc_score(y_true, -scores),
#         "clf": clf
#     }
#     return results

# results = {}

# #Quick diagnostics
# Cs = np.logspace(-2.3, 1, 10)
# for hemi in hemis:
#     X = get_feature_matrix(df_no_NS_ptlocal, hemi, selected_tracts)
#     accs, aucs = [], []
#     for C in Cs:
#         r = loso_evaluate(X, y, C)
#         accs.append(r["accuracy"]); aucs.append(r["roc_auc"])
#     print(hemi, Cs, np.round(accs, 2), np.round(aucs, 2))

# #train model per hemisphere
# # df_no_mtfef = df[df["Tract"] != "MTxPTxSTS1"].copy()
# df_no = df[~((df["Group"] == "NS") & (df["Localizer"] == "ptlocal"))].copy()
# selected_tracts = tracts[[0,2]]
# Cs = np.array([0.005011872336272724, 1.5914209198620377])#np.array([0.018, 1.714])
# for i, hemi in enumerate(hemis):
#     print(f"\n=== Hemisphere {hemi} ===")

#     X = get_feature_matrix(df_no, hemi, selected_tracts)
    
#     res = loso_evaluate(X, y, Cs[i])
    
#     print(f"Regularization parameter: {Cs[i]:.3f}")
#     print(f"Accuracy:           {res['accuracy']:.3f}")
#     print(f"Balanced accuracy:  {res['balanced_accuracy']:.3f}")
#     print(f"ROC AUC:            {res['roc_auc']:.3f}")
#     print(f"ROC AUC Flipped:    {res['auc_flipped']:.3f}")

#     results[hemi] = res


#-----------------------------------------------------------------------
# #Show only left hemi:


# def plot_2d_with_plane(
#     df, participants, y, hemis, tracts, C
# ):
#     assert len(tracts) == 2, "Decision boundary requires exactly 2 features"

#     fig, axes = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)

#     for i, hemi in enumerate(hemis):

#         # ------------------------
#         # Feature matrix
#         # ------------------------
#         X = get_feature_matrix(df, hemi, tracts)
#         C_val = C[i]
#         clf = fit_final_model(X, y, C_val)
#         w, b = get_decision_plane_raw(clf)

#         ax = axes

#         # ---- Scatter points ----
#         ax.scatter(
#             X[y == 1, 0], X[y == 1, 1],
#             c="blue", marker="s", s=60, label="EB"
#         )
#         ax.scatter(
#             X[y == 0, 0], X[y == 0, 1],
#             c="red", marker="o", s=60, label="NS"
#         )

#         # ---- Decision boundary ----
#         x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
#         y_vals = (-w[0] * x_vals - b) / w[1]

#         ax.plot(
#             x_vals,
#             y_vals,
#             color="black",
#             linewidth=2,
#             label="Decision boundary"
#         )
#         #1 remove box
#         sns.despine(ax=ax, top=True, right=True)

#         # 2. Make the remaining axes (Left and Bottom) thick
#         ax.spines['left'].set_linewidth(2.5)
#         ax.spines['bottom'].set_linewidth(2.5)

#         # ---- Optional: margin lines ----
#         # margin = 1 / np.linalg.norm(w)
#         # y_margin_up = (-w[0] * x_vals - b + 1) / w[1]
#         # y_margin_down = (-w[0] * x_vals - b - 1) / w[1]

#         # ax.plot(x_vals, y_margin_up, "k--", linewidth=1)
#         # ax.plot(x_vals, y_margin_down, "k--", linewidth=1)

#         # ---- Formatting ----
#         ax.set_xlabel(tracts[0],fontsize=18,fontweight='bold')
#         ax.set_ylabel(tracts[1],fontsize=18,fontweight='bold')
#         ax.set_title(f"{hemi} Hemisphere",fontsize=18,fontweight='bold')
#         ax.legend()
#         plt.setp(ax.get_yticklabels(),fontsize=18,fontweight='bold')

#     plt.suptitle("Linear SVM decision boundaries", fontsize=16)

#     saveDir = op.join(bids_path, "analysis", "plots")
#     os.makedirs(saveDir, exist_ok=True)
#     plt.savefig(
#         op.join(
#             saveDir,
#             "left_beta_weights_2d_linear_svm_innercv.png"
#         ),
#         dpi=300,
#         bbox_inches="tight"
#     )

#     plt.show()

# selected_tracts = tracts[[0,2]]  # example — must be exactly 3

# plot_2d_with_plane(
#     df=df,
#     participants=participants,
#     y=y,
#     hemis=["L"],
#     tracts=selected_tracts,
#     C = np.array([0.0089, 0.0331])
# )
