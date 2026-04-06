import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu


# =========================================================
# FIND CLUSTERS OF SIGNIFICANT TIMEPOINTS
# =========================================================
def find_clusters(stats):
    """
    stats: boolean array (True = significant timepoint)
    returns: list of arrays (clusters of indices)
    """
    sig_timepts = np.where(stats)[0]

    clusters = []

    if len(sig_timepts) == 0:
        return clusters

    # find breaks between consecutive indices
    diffs = np.diff(sig_timepts)
    jumps = np.where(diffs != 1)[0] #indices of boundaries between any two clusters

    start = 0
    for jump in jumps:
        clusters.append(sig_timepts[start:jump+1])
        start = jump + 1

    # last cluster
    clusters.append(sig_timepts[start:])

    return clusters


# =========================================================
# GENERATE NULL DISTRIBUTION (PERMUTATION TEST)
# =========================================================

def sig_cluster_dist(data, num_iterations=1000, alpha=0.05):

    hits = data["hits"]   # shape: (n1, timepoints)
    miss = data["miss"]   # shape: (n2, timepoints)

    n1, n_timepoints = hits.shape
    n2 = miss.shape[0]

    # pool data
    all_data = np.vstack([hits, miss])
    n_total = n1 + n2

    max_interval_len = np.zeros(num_iterations)

    for iteration in range(num_iterations):

        # shuffle subject indices
        perm_idx = np.random.permutation(n_total)

        # reassign groups (same sizes)
        perm_hits = all_data[perm_idx[:n1], :]
        perm_miss = all_data[perm_idx[n1:], :]

        # independent t-test at each timepoint
        tstat, pvals = ttest_ind(
            perm_miss,
            perm_hits,
            axis=0,
            equal_var=False,
            nan_policy='omit'
        )
        # stat, pvals = mannwhitneyu(perm_miss, perm_hits, axis=0, alternative='two-sided')

        sig = pvals < alpha

        clusters = find_clusters(sig)

        if len(clusters) > 0:
            max_interval_len[iteration] = max(len(c) for c in clusters)
        else:
            max_interval_len[iteration] = 0

    upper_ci = np.quantile(max_interval_len, 0.95)

    return upper_ci, max_interval_len
# =========================================================
# MAIN ANALYSIS
# =========================================================
def run_cluster_test(yvar, alpha=0.05, n_iter=1000):
    """
    yvar shape: (timepoints, condition[2], subjects)
    """

    # reshape like MATLAB
    hits = np.squeeze(yvar[0:7, :])  # (subjects, timepoints)
    miss = np.squeeze(yvar[7:, :])

    data = {
        "hits": hits,
        "miss": miss
    }

    # -----------------------------------
    # Null distribution
    # -----------------------------------
    cluster_limit, cluster_dist = sig_cluster_dist(data, n_iter, alpha)
    print("done making cluster dist")

    # -----------------------------------
    # Real t-tests
    # -----------------------------------
    pvals = np.array([
        ttest_ind(miss[:, t], hits[:, t], nan_policy='omit').pvalue
        for t in range(hits.shape[1])
    ])

    sig = pvals < alpha
    print("done running t-tests")

    # -----------------------------------
    # Find clusters
    # -----------------------------------
    clusters = find_clusters(sig)

    sig_cluster_idx = [
        i for i, c in enumerate(clusters)
        if len(c) > cluster_limit
    ]

    print("done running sig clusters")

    return {
        "clusters": clusters,
        "significant_clusters": sig_cluster_idx,
        "cluster_limit": cluster_limit,
        "pvals": pvals
    }