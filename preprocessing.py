import numpy as np
import config
import metrics
import helpers
import os


def replace_brfss_special_codes(X):
    """Normalize BRFSS special codes and convert obvious weight/height encodings.

    - 4-digit codes: 7777/9999 -> NaN, 8888 -> 0
    - 3-digit codes: 777/999 -> NaN, 888/555 -> 0
    - 2-digit codes: 77/99 -> NaN, 88 -> 0
    - Weight-like columns: (9000, 9999) encode weight in lbs offset by 9000 -> subtract and convert to kg*? (apply 2.20462 factor).
    - Height-like columns: values in [200, 711] encode ft'in -> convert to cm.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    for j in range(X.shape[1]):
        x = X[:, j]
        l = 0
        if np.sum((x > 9000) & (x < 9999)) >= 400:
            l = 1 if np.sum(x < 200) > 10 else 2
        if np.isin(x, [7777.0, 8888.0, 9999.0]).any():
            x = np.where(x == 7777.0, np.nan, x)
            x = np.where(x == 9999.0, np.nan, x)
            x = np.where(x == 8888.0, 0.0, x)
        elif np.isin(x, [777.0, 888.0, 999.0, 555.0]).any():
            x = np.where(x == 777.0, np.nan, x)
            x = np.where(x == 999.0, np.nan, x)
            x = np.where(x == 888.0, 0.0, x)
            x = np.where(x == 555.0, 0.0, x)
        elif np.isin(x, [77.0, 88.0, 99.0]).any():
            x = np.where(x == 77.0, np.nan, x)
            x = np.where(x == 99.0, np.nan, x)
            x = np.where(x == 88.0, 0.0, x)
        if l == 1:
            # Decode weights encoded as 9000 + pounds, then convert pounds -> kg
            x = np.where((x > 9000.0) & (x < 9999.0), (x - 9000.0) / 2.20462, x)
        elif l == 2:
            m = (x >= 200.0) & (x <= 711.0)
            if np.any(m):
                ft = x // 100.0
                inch = x % 100.0
                x = np.where(m, ft * 30.48 + inch * 2.54, x)
        X[:, j] = x
    return X


# ==========================================


def is_integer_array(v, tol=1e-6):
    """Return True if all non-NaN values are within a tolerance of an integer.

    Args:
        v (np.ndarray): Input vector.
        tol (float): Numerical tolerance.

    Returns:
        bool: True if all non-NaN values are near integers.
    """
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return False
    return np.all(np.abs(v - np.round(v)) < tol)


def infer_feature_types(X, max_unique_cat=None):
    """Infer coarse types: binary, nominal, ordinal, continuous.

    Heuristics: 2 unique values to binary; low-card integer-ish to ordinal;
    other low-card to nominal; remaining to continuous.

    Args:
        X (np.ndarray): Data matrix.
        max_unique_cat (int, optional): Max unique values for low-card detection.

    Returns:
        dict: {"binary", "nominal", "ordinal", "continuous"} → list of column indices.
    """
    if max_unique_cat is None:
        max_unique_cat = config.LOW_CARD_MAX_UNIQUE
    _, d = X.shape
    types = {"binary": [], "nominal": [], "ordinal": [], "continuous": []}
    for j in range(d):
        col = X[:, j]
        v = col[~np.isnan(col)]
        if len(v) == 0:
            types["nominal"].append(j)
            continue
        uniq = np.unique(v)
        nunique = len(uniq)
        if nunique == 2 and set(np.round(uniq).tolist()).issubset({0, 1}):
            types["binary"].append(j)
            continue
        if nunique <= max_unique_cat:
            if is_integer_array(v):
                umin, umax = int(np.min(uniq)), int(np.max(uniq))
                if (umax - umin) <= 6:
                    types["ordinal"].append(j)
                else:
                    types["nominal"].append(j)
            else:
                types["nominal"].append(j)
            continue
        types["continuous"].append(j)
    return types


def smart_impute(
    Xtr,
    Xte,
    skew_rule=0.5,
    allnan_fill_cont=0.0,
    allnan_fill_nom=-1.0,
    allnan_fill_bin=0.0,
):
    """Impute missing values using robust, simple rules.

    - Categorical (binary/nominal/ordinal): mode, with per-family fallbacks when fully missing.
    - Continuous: median if mean/median differ relative to std (skew_rule), else mean.
    - Unit-interval ordinal-like with few levels are treated as continuous.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        skew_rule (float): Controls when to prefer median over mean for continuous columns.
        allnan_fill_cont (float): Fallback for continuous columns with only missing values.
        allnan_fill_nom (float): Fallback for nominal or ordinal columns with only missing values.
        allnan_fill_bin (float): Fallback for binary columns with only missing values.

    Returns:
        tuple[np.ndarray, np.ndarray]: Imputed training and test matrices.
    """
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)
    n, d = Xtr.shape
    assert Xte.shape[1] == d

    types = infer_feature_types(Xtr, max_unique_cat=config.LOW_CARD_MAX_UNIQUE)
    cont_set = set(types["continuous"])

    for j in range(d):
        v = Xtr[:, j]
        w = v[~np.isnan(v)]
        if len(w) == 0:
            continue
        w_min, w_max = float(np.min(w)), float(np.max(w))
        if (w_min >= -1e-6) and (w_max <= 1.0 + 1e-6):
            nunique = np.unique(np.round(w, 6))
            if 3 <= len(nunique) <= 7:
                cont_set.add(j)

    is_cont = np.zeros(d, dtype=bool)
    if cont_set:
        is_cont[list(cont_set)] = True

    fam_bin = [j for j in types["binary"] if not is_cont[j]]
    fam_nom = [j for j in types["nominal"] if not is_cont[j]]
    fam_ord = [j for j in types["ordinal"] if not is_cont[j]]
    fam_cont = sorted(list(cont_set))

    fill_vals = np.empty(d, dtype=np.float32)

    for _, idxs, fallback in (
        ("binary", fam_bin, allnan_fill_bin),
        ("nominal", fam_nom, allnan_fill_nom),
        ("ordinal", fam_ord, allnan_fill_nom),
    ):
        for j in idxs:
            w = Xtr[:, j][~np.isnan(Xtr[:, j])]
            if len(w) == 0:
                fill_vals[j] = float(fallback)
            else:
                vals, counts = np.unique(w, return_counts=True)
                winners = np.where(counts == counts.max())[0]
                fill_vals[j] = float(vals[winners].min())

    for j in fam_cont:
        w = Xtr[:, j][~np.isnan(Xtr[:, j])]
        if len(w) == 0:
            fill_vals[j] = float(allnan_fill_cont)
        else:
            mean_val = float(np.mean(w))
            median_val = float(np.median(w))
            std_val = float(np.std(w)) + 1e-12
            prefer_median = abs(mean_val - median_val) > float(skew_rule) * std_val
            fill_vals[j] = median_val if prefer_median else mean_val

    nan_tr = np.isnan(Xtr)
    nan_te = np.isnan(Xte)
    if nan_tr.any():
        Xtr[nan_tr] = np.take(fill_vals, np.where(nan_tr)[1])
    if nan_te.any():
        Xte[nan_te] = np.take(fill_vals, np.where(nan_te)[1])

    return Xtr, Xte


# ==========================================


def one_hot_encoding_selected(
    Xtr,
    Xte,
    cat_idx,
    drop_first=True,
    total_cap=None,
):
    """One-hot encode selected categorical columns with baseline drop and capacity control.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        cat_idx (list[int]): Indices of categorical columns to encode.
        drop_first (bool): If True, drop the most frequent level as baseline.
        total_cap (int or None): Global cap on the number of added dummy columns.

    Returns:
        tuple: (Xtr_new, Xte_new, plan, keep_idx, dummy_map)
    """
    if total_cap is None:
        total_cap = config.MAX_ADDED_ONEHOT

    new_tr_cols, new_te_cols = [], []
    used_idx, plan = [], []
    dummy_map = {}
    added_total = 0

    for j in cat_idx:
        col_tr = Xtr[:, j]
        col_te = Xte[:, j]

        valid_tr = ~np.isnan(col_tr)
        if np.any(valid_tr):
            uniq, counts = np.unique(col_tr[valid_tr], return_counts=True)
            order = np.argsort(-counts)
            uniq = uniq[order]
        else:
            uniq = np.array([], dtype=col_tr.dtype)

        kept_all = uniq

        baseline = None
        kept_for_ohe = kept_all
        if drop_first and len(kept_all) > 0:
            baseline = float(kept_all[0])
            kept_for_ohe = kept_all[1:]

        values_to_encode = kept_for_ohe.tolist()
        values_to_encode.append(np.nan)

        k_add = len(values_to_encode)
        if (total_cap is not None) and (added_total + k_add > int(total_cap)):
            continue

        used_idx.append(j)
        plan.append(
            (
                j,
                {
                    "kept_values": [float(x) for x in kept_all.tolist()],
                    "baseline": (None if baseline is None else float(baseline)),
                },
            )
        )

        tr_isnan = np.isnan(col_tr)
        te_isnan = np.isnan(col_te)

        for v in values_to_encode:
            if (isinstance(v, float) and np.isnan(v)) or (v is np.nan):
                new_tr_cols.append(tr_isnan)
                new_te_cols.append(te_isnan)
            else:
                new_tr_cols.append((col_tr == v))
                new_te_cols.append((col_te == v))

        added_total += k_add
        dummy_map[j] = []

    keep_idx = [jj for jj in range(Xtr.shape[1]) if jj not in used_idx]
    Xtr_keep, Xte_keep = Xtr[:, keep_idx], Xte[:, keep_idx]

    if new_tr_cols:
        Xtr_new = np.column_stack([Xtr_keep] + new_tr_cols)
        Xte_new = np.column_stack([Xte_keep] + new_te_cols)

        base = Xtr_keep.shape[1]
        cursor = base
        k_per_feat = []
        for j, meta in plan:
            k_feat = len(meta["kept_values"])
            if drop_first and meta["baseline"] is not None and k_feat > 0:
                k_feat -= 1
            k_feat += 1
            k_per_feat.append(k_feat)

        for (j, meta), k in zip(plan, k_per_feat):
            dummy_map[j] = list(range(cursor, cursor + k))
            cursor += k
    else:
        Xtr_new, Xte_new = Xtr_keep, Xte_keep

    print(
        f"[Preprocess] one-hot:"
        f" kept_raw={len(keep_idx)}"
        f", encoded_feat={len(used_idx)}"
        f", added_cols={sum(len(v) for v in dummy_map.values())}"
        f", drop_first={drop_first}"
        f", total_cap={total_cap}"
    )

    return Xtr_new, Xte_new, plan, keep_idx, dummy_map


# ==========================================


def remove_highly_correlated_continuous(Xtr, Xte, cont_idx, y_train, threshold=0.90):
    """Prune highly correlated continuous features, favoring target-aligned and higher-variance columns.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        cont_idx (array-like): Indices of continuous features to consider.
        y_train (np.ndarray): Targets used to guide tie-breaking.
        threshold (float): Absolute correlation threshold for pruning.

    Returns:
        tuple: (Xtr_new, Xte_new, dropped_indices, kept_indices)
    """
    Xtr = np.ascontiguousarray(np.asarray(Xtr, dtype=np.float32))
    Xte = np.ascontiguousarray(np.asarray(Xte, dtype=np.float32))
    y = np.asarray(y_train, dtype=np.float32).ravel()
    cont_idx = np.asarray(cont_idx, dtype=int)
    if len(cont_idx) <= 1:
        print(
            f"[Preprocess] corr prune (continuous): nothing to do (n_cont={len(cont_idx)})."
        )
        kept = list(range(Xtr.shape[1]))
        return Xtr, Xte, [], kept
    Xc = Xtr[:, cont_idx]
    if np.isnan(Xc).any():
        raise ValueError(
            "NaNs in continuous block. Impute or standardize before pruning."
        )
    corr = np.corrcoef(Xc, rowvar=False)
    y_center = y - y.mean()
    Xc_center = Xc - Xc.mean(axis=0, keepdims=True)
    denom = np.sqrt((Xc_center**2).sum(axis=0)) * np.sqrt((y_center**2).sum())
    tgt_corr = np.zeros(Xc.shape[1], dtype=np.float32)
    nz = denom > 0
    tgt_corr[nz] = np.abs((Xc_center[:, nz].T @ y_center) / denom[nz])
    variances = Xc.var(axis=0)
    keep_local = np.ones(Xc.shape[1], dtype=bool)
    D = Xc.shape[1]
    for i in range(D):
        if not keep_local[i]:
            continue
        high = np.abs(corr[i, (i + 1) :]) >= threshold
        if not high.any():
            continue
        js = np.where(high)[0] + (i + 1)
        for j in js:
            if not keep_local[j]:
                continue
            ti, tj = tgt_corr[i], tgt_corr[j]
            if ti > tj:
                keep_local[j] = False
            elif tj > ti:
                keep_local[i] = False
            else:
                if variances[i] >= variances[j]:
                    keep_local[j] = False
                else:
                    keep_local[i] = False
    cont_keep_idx = cont_idx[keep_local]
    cont_drop_idx = cont_idx[~keep_local]
    keep_global = np.ones(Xtr.shape[1], dtype=bool)
    keep_global[cont_drop_idx] = False
    Xtr_new = Xtr[:, keep_global]
    Xte_new = Xte[:, keep_global]
    print(
        f"[Preprocess] corr prune (continuous): thr={threshold} → dropped {len(cont_drop_idx)} / kept {len(cont_keep_idx)} continuous (final D={Xtr_new.shape[1]})"
    )
    return Xtr_new, Xte_new, cont_drop_idx.tolist(), np.where(keep_global)[0].tolist()


# ==========================================


def pca_local_on_ohe(Xtr, Xte, dummy_map, cfg=None):
    """Apply PCA independently within each one-hot block to reduce dimensionality.

    PCA is fit on training data per block and applied to the matched columns in test.
    You can replace original dummy columns or append the components.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        dummy_map (dict[int, list[int]]): Mapping from source categorical feature to its one-hot column indices.
        cfg (dict or float or int or None): Configuration for variance target, fixed components, minimum block size, and replacement behavior.

    Returns:
        tuple: (Xtr_out, Xte_out, spec) with projection metadata and component indices.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)

    if cfg is None:
        cfg = config.PCA_Local
    if cfg is None or cfg is False:
        return Xtr, Xte, {"groups": {}, "total_k": 0}

    if isinstance(cfg, (float, int)):
        cfg = {"variance_ratio": float(cfg)}
    vr = (
        float(np.clip(cfg.get("variance_ratio", 0.90), 0.0, 1.0))
        if "n_components" not in cfg
        else None
    )
    k_fixed = cfg.get("n_components", None)
    min_cols = int(cfg.get("min_cols", 6))
    replace = bool(cfg.get("replace", True))

    groups = {
        int(j): list(map(int, idxs))
        for j, idxs in dummy_map.items()
        if len(idxs) >= min_cols
    }
    if not groups:
        return Xtr, Xte, {"groups": {}, "total_k": 0}

    keep_mask = np.ones(Xtr.shape[1], dtype=bool)
    proj_tr_list, proj_te_list = [], []
    spec_groups = {}
    total_k = 0

    for j, cols in groups.items():
        cols = np.array(cols, dtype=int)
        Xtr_blk = Xtr[:, cols]
        Xte_blk = Xte[:, cols]

        mean_tr = np.mean(Xtr_blk, axis=0, dtype=np.float64)
        Xtr_c = Xtr_blk - mean_tr
        Xte_c = Xte_blk - mean_tr

        _, S, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
        n_samples = Xtr_c.shape[0]
        explained_var = (S**2) / max(n_samples - 1, 1)
        explained_ratio = explained_var / np.sum(explained_var)

        k_max = Vt.shape[0]
        if k_fixed is not None:
            k = int(min(max(int(k_fixed), 1), k_max))
        else:
            cumsum = np.cumsum(explained_ratio)
            k = int(np.searchsorted(cumsum, vr, side="left") + 1)

        comps = Vt[:k, :].T
        Ztr = Xtr_c @ comps
        Zte = Xte_c @ comps

        spec_groups[j] = {
            "cols": cols.tolist(),
            "k": int(k),
            "mean": mean_tr,
            "components": comps,
            "explained_ratio": explained_ratio[:k],
        }
        total_k += k

        if replace:
            keep_mask[cols] = False
        proj_tr_list.append(Ztr)
        proj_te_list.append(Zte)

    if replace:
        Xtr_out = (
            np.column_stack([Xtr[:, keep_mask]] + proj_tr_list) if proj_tr_list else Xtr
        )
        Xte_out = (
            np.column_stack([Xte[:, keep_mask]] + proj_te_list) if proj_te_list else Xte
        )

        base = int(keep_mask.sum())
        pca_idx_cursor = base
        for j in groups.keys():
            k = int(spec_groups[j]["k"])
            spec_groups[j]["pca_component_idx"] = list(
                range(pca_idx_cursor, pca_idx_cursor + k)
            )
            pca_idx_cursor += k
    else:
        Xtr_out = np.column_stack([Xtr] + proj_tr_list) if proj_tr_list else Xtr
        Xte_out = np.column_stack([Xte] + proj_te_list) if proj_te_list else Xte

        base = Xtr.shape[1]
        pca_idx_cursor = base
        for j in groups.keys():
            k = int(spec_groups[j]["k"])
            spec_groups[j]["pca_component_idx"] = list(
                range(pca_idx_cursor, pca_idx_cursor + k)
            )
            pca_idx_cursor += k

    print(
        f"[Preprocess] PCA-Local on OHE: groups={len(groups)} total_k={total_k} replace={replace}"
    )
    return Xtr_out, Xte_out, {"groups": spec_groups, "total_k": int(total_k)}


# ==========================================


def standardize(Xtr_new, Xte_new, cont_idx=None, return_updated_idx=False):
    """Standardize using training stats; drop zero-variance columns.

    Modes:
      - config.STD_CONT == False: standardize all columns globally and drop any zero-variance columns.
      - config.STD_CONT == True & cont_idx is None: fallback to global standardization (same as above).
      - config.STD_CONT == True & cont_idx set: standardize only those columns and drop zero-variance ones within that subset.
    """

    Xtr_new = np.asarray(Xtr_new, dtype=np.float32)
    Xte_new = np.asarray(Xte_new, dtype=np.float32)
    std_cont = bool(config.STD_CONT)
    use_subset = std_cont and (cont_idx is not None) and (len(cont_idx) > 0)

    # ---------- GLOBAL STANDARDIZATION ----------
    if not use_subset:
        mean_tr = np.mean(Xtr_new, axis=0)
        std_tr = np.std(Xtr_new, axis=0)

        zero_var_mask = std_tr == 0
        std_safe = std_tr.copy()
        std_safe[zero_var_mask] = 1.0

        Xtr_s = (Xtr_new - mean_tr) / std_safe
        Xte_s = (Xte_new - mean_tr) / std_safe

        if np.any(zero_var_mask):
            keep_mask = ~zero_var_mask
            dropped = int(np.sum(zero_var_mask))
            Xtr_s = Xtr_s[:, keep_mask]
            Xte_s = Xte_s[:, keep_mask]
            print(
                f"[Standardize] mode=global | dropped {dropped} zero-variance columns | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}"
            )
        else:
            print(
                f"[Standardize] mode=global | no zero-variance | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}"
            )

        return (
            (Xtr_s, Xte_s)
            if not return_updated_idx
            else (
                Xtr_s,
                Xte_s,
                np.array(cont_idx if cont_idx is not None else [], dtype=int),
            )
        )

    # ---------- SUBSET STANDARDIZATION ----------
    cont_idx = np.asarray(cont_idx, dtype=int)
    Xtr_s = Xtr_new.copy()
    Xte_s = Xte_new.copy()
    mu = np.mean(Xtr_new[:, cont_idx], axis=0)
    sd = np.std(Xtr_new[:, cont_idx], axis=0)

    zero_var_mask_local = sd == 0
    sd_safe = sd.copy()
    sd_safe[sd_safe == 0] = 1.0

    Xtr_s[:, cont_idx] = (Xtr_new[:, cont_idx] - mu) / sd_safe
    Xte_s[:, cont_idx] = (Xte_new[:, cont_idx] - mu) / sd_safe

    if np.any(zero_var_mask_local):
        to_drop_global = cont_idx[zero_var_mask_local]
        keep_mask_global = np.ones(Xtr_s.shape[1], dtype=bool)
        keep_mask_global[to_drop_global] = False
        Xtr_s = Xtr_s[:, keep_mask_global]
        Xte_s = Xte_s[:, keep_mask_global]

        new_positions = np.nonzero(keep_mask_global)[0]
        cont_idx_kept_old = cont_idx[~zero_var_mask_local]
        pos_map = {old: new for new, old in enumerate(new_positions)}
        cont_idx_kept_new = np.array([pos_map[i] for i in cont_idx_kept_old], dtype=int)

        print(
            f"[Standardize] mode=subset | dropped {len(to_drop_global)} zero-variance continuous cols | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}"
        )
        if return_updated_idx:
            return Xtr_s, Xte_s, cont_idx_kept_new
        return Xtr_s, Xte_s

    print(
        f"[Standardize] mode=subset | n_cols={len(cont_idx)} | Xtr,Xte: {Xtr_s.shape} {Xte_s.shape}"
    )
    if return_updated_idx:
        return Xtr_s, Xte_s, cont_idx
    return Xtr_s, Xte_s


# ==========================================


def pca(Xtr, Xte, cols=None, n_components=None, variance_ratio=None, replace=True):
    """Fit PCA on selected columns (train), project both splits.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        cols (array-like or None): Indices to project; defaults to all columns.
        n_components (int or None): Fixed number of components to keep.
        variance_ratio (float or None): Minimum explained variance ratio to retain.
        replace (bool): If True, replace the original block; otherwise, append components.

    Returns:
        tuple: (Xtr_out, Xte_out, spec) with PCA metadata and resulting indices.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    if cols is None:
        cols = np.arange(Xtr.shape[1])
    else:
        cols = np.array(cols, dtype=int)
    Xtr_blk = Xtr[:, cols]
    Xte_blk = Xte[:, cols]
    mean_tr = np.mean(Xtr_blk, axis=0, dtype=np.float64)
    Xtr_c = Xtr_blk - mean_tr
    Xte_c = Xte_blk - mean_tr
    _, S, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    n_samples = Xtr_c.shape[0]
    explained_var = (S**2) / max(n_samples - 1, 1)
    explained_ratio = explained_var / np.sum(explained_var)
    k_max = Vt.shape[0]
    k = k_max
    cfg_k = config.PCA_K
    if variance_ratio is not None:
        vr = float(np.clip(variance_ratio, 0.0, 1.0))
        cumsum = np.cumsum(explained_ratio)
        k = int(np.searchsorted(cumsum, vr, side="left") + 1)
    elif n_components is not None:
        k = int(min(max(n_components, 1), k_max))
    elif cfg_k is not None:
        if isinstance(cfg_k, float) and 0 < cfg_k <= 1.0:
            cumsum = np.cumsum(explained_ratio)
            k = int(np.searchsorted(cumsum, cfg_k, side="left") + 1)
        else:
            k = int(min(max(cfg_k, 1), k_max))
    components = Vt[:k, :].T
    Xtr_proj = Xtr_c @ components
    Xte_proj = Xte_c @ components
    explained = float(np.sum(explained_ratio[:k])) * 100
    print(
        f"[Preprocess] PCA: block d={Xtr_blk.shape[1]} → k={k} comps ({explained:.2f}% variance), replace={replace}"
    )
    if replace:
        keep_idx = [j for j in range(Xtr.shape[1]) if j not in set(cols)]
        Xtr_out = np.column_stack([Xtr[:, keep_idx], Xtr_proj])
        Xte_out = np.column_stack([Xte[:, keep_idx], Xte_proj])
        base = len(keep_idx)
        pca_component_idx = list(range(base, base + Xtr_proj.shape[1]))
    else:
        Xtr_out = np.column_stack([Xtr, Xtr_proj])
        Xte_out = np.column_stack([Xte, Xte_proj])
        base = Xtr.shape[1]
        pca_component_idx = list(range(base, base + Xtr_proj.shape[1]))
    spec = {
        "cols": np.array(cols, int).tolist(),
        "mean": mean_tr,
        "components": components,
        "explained_ratio": explained_ratio[:k],
        "k": k,
        "replace": bool(replace),
        "pca_component_idx": pca_component_idx,
    }
    return Xtr_out, Xte_out, spec


# ==========================================


def filter_add_predictive_nan_indicators(
    Xtr,
    Xte,
    y_train,
    threshold=0.01,
    top_k=128,
    min_prevalence=0.005,
    max_prevalence=0.995,
):
    """Add missingness indicators selected by correlation with the target.

    Drops any source feature whose NaN rate in the test split is >= 0.30,
    then creates isnan indicators and keeps those with highest absolute
    correlation to the target (threshold + top-k). The same selection
    is applied to test.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    y = np.asarray(y_train, np.float32).ravel()

    # ---- 1) ----
    test_nan_rate = (
        np.isnan(Xte).mean(axis=0) if len(Xte) > 0 else np.array([], dtype=np.float32)
    )
    keep_cols = (
        test_nan_rate < 0.70 if len(test_nan_rate) > 0 else np.array([], dtype=bool)
    )
    if len(test_nan_rate) > 0 and not np.all(keep_cols):
        n_drop = int((~keep_cols).sum())
        n_keep = int(keep_cols.sum())
        print(
            f"[Preprocess] NaN-based feature filter (test): dropped={n_drop}, kept={n_keep}, thr=0.70"
        )
        if n_keep == 0:
            return Xtr[:, :0], Xte[:, :0]
        Xtr = Xtr[:, keep_cols]
        Xte = Xte[:, keep_cols]
    elif len(test_nan_rate) > 0:
        print(
            f"[Preprocess] NaN-based feature filter (test): dropped=0, kept={Xtr.shape[1]}, thr=0.30"
        )

    # ---- 2) ----
    Mtr = np.isnan(Xtr)
    Mte = np.isnan(Xte)
    if len(Mtr.T) == 0:
        print("[Preprocess] NaN indicators: no features after test-NaN filter")
        return Xtr, Xte
    prev = Mtr.mean(axis=0)
    keep_prev = (prev >= float(min_prevalence)) & (prev <= float(max_prevalence))
    if not np.any(keep_prev):
        print("[Preprocess] NaN indicators: none selected (prevalence filter)")
        return Xtr, Xte

    # ---- 3) ----
    yz = (y - y.mean()) / (y.std() + 1e-12)
    mu = Mtr[:, keep_prev].mean(axis=0)
    sd = Mtr[:, keep_prev].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (Mtr[:, keep_prev] - mu) / sd
    corrs = (Z.T @ yz) / (Xtr.shape[0] - 1)
    scores = np.abs(np.nan_to_num(corrs, 0.0))
    if threshold is not None:
        mask_thr = scores > float(threshold)
        if not np.any(mask_thr):
            print("[Preprocess] NaN indicators: none above threshold")
            return Xtr, Xte
        cand_local = np.where(mask_thr)[0]
    else:
        cand_local = np.arange(len(scores))
    if top_k is not None and len(cand_local) > int(top_k):
        order = np.argsort(-scores[cand_local])[: int(top_k)]
        cand_local = cand_local[order]

    # ---- 4) ----
    cand_global = np.where(keep_prev)[0][cand_local]
    if len(cand_global) == 0:
        print("[Preprocess] NaN indicators: none selected")
        return Xtr, Xte
    Xtr_aug = np.hstack([Xtr, Mtr[:, cand_global]])
    Xte_aug = np.hstack([Xte, Mte[:, cand_global]])
    print(
        f"[Preprocess] NaN indicators: add {len(cand_global)}/{Xtr.shape[1]} (thr={threshold}, top_k={top_k}, prev∈[{min_prevalence},{max_prevalence}])"
    )
    return Xtr_aug, Xte_aug


# ==========================================


def _listify_idx(idx, d):
    """Return a NumPy array of indices. If None, return all column indices.

    Args:
        idx (array-like or None): Indices or None.
        d (int): Number of columns.

    Returns:
        np.ndarray: Indices as integers.
    """
    if idx is None:
        return np.arange(d, dtype=int)
    return np.array(idx, dtype=int)


def polynomial_features(
    Xtr,
    Xte,
    y_train,
    cont_idx=None,
    add_squares_cont=True,
    add_inter_within_cont=True,
    top_k_pairs=256,
    min_abs_corr=0.0,
):
    """Add polynomial features to train and test: x^2 and x_i*x_j (continuous only).

    This simplified version keeps only squares of continuous features and
    pairwise interactions among continuous features.

    Selection uses absolute correlation with the target to optionally filter
    and rank features.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        y_train (np.ndarray): Target vector used for scoring.
        cont_idx (array-like or None): Indices considered as continuous. If None, all columns.
        add_squares_cont (bool): Add squares for continuous indices.
        add_inter_within_cont (bool): Add interactions within continuous indices.
        top_k_pairs (int): Max number of interaction terms to keep (after thresholding).
        min_abs_corr (float): Minimum absolute correlation score for added features.

    Returns:
        tuple: (Xtr_aug, Xte_aug, spec) with added features and the recipe used.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    y = np.asarray(y_train, np.float32).ravel()
    d = Xtr.shape[1]
    cont_idx = _listify_idx(cont_idx, d)

    # Build candidate squares (continuous only)
    squares = []
    if add_squares_cont and len(cont_idx):
        squares = [("square", int(i)) for i in cont_idx]

    # Build candidate pairwise interactions within continuous features only
    pairs = []
    if add_inter_within_cont and len(cont_idx) > 1:
        ci = cont_idx
        for a in range(len(ci)):
            for b in range(a + 1, len(ci)):
                pairs.append(("prod", int(ci[a]), int(ci[b])))

    # If nothing to add, return early
    if not squares and not pairs:
        return Xtr, Xte, {"squares": [], "pairs": [], "n_added": 0}

    added_blocks = []
    meta_blocks = []
    if squares:
        cols = np.array([t[1] for t in squares], dtype=int)
        Z = Xtr[:, cols] ** 2
        added_blocks.append(Z)
        meta_blocks += [("square", int(k)) for k in cols]

    def score_block(Z: np.ndarray) -> np.ndarray:
        yz = (y - y.mean()) / (y.std() + 1e-12)
        mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd[sd == 0] = 1.0
        Zs = (Z - mu) / sd
        corrs = (Zs.T @ yz) / (Xtr.shape[0] - 1)
        return np.abs(np.nan_to_num(corrs, 0.0))

    # Score pairwise interactions in chunks to bound memory
    pair_chunks = []
    chunk_meta = []
    CHUNK = max(1, 16384 // max(1, Xtr.shape[0]))
    if pairs:
        cur = 0
        while cur < len(pairs):
            end = min(len(pairs), cur + CHUNK)
            P = pairs[cur:end]
            Z = np.empty((Xtr.shape[0], len(P)), dtype=np.float32)
            for k, (_, i, j) in enumerate(P):
                Z[:, k] = Xtr[:, i] * Xtr[:, j]
            sc = score_block(Z)
            pair_chunks.append((Z, sc))
            chunk_meta.append(P)
            cur = end

    # Compute scores
    scores_sq = np.array([], dtype=np.float32)
    if squares:
        scores_sq = score_block(added_blocks[0])
    scores_pairs = np.array([], dtype=np.float32)
    if pairs:
        scores_pairs = np.concatenate([sc for (_, sc) in pair_chunks], axis=0)

    # Thresholding
    keep_sq = np.ones_like(scores_sq, dtype=bool)
    if len(scores_sq) and min_abs_corr is not None:
        keep_sq = scores_sq >= float(min_abs_corr)
    keep_pairs = np.ones_like(scores_pairs, dtype=bool)
    if len(scores_pairs) and min_abs_corr is not None:
        keep_pairs = scores_pairs >= float(min_abs_corr)

    # Ranking
    order_sq = (
        np.argsort(-(scores_sq[keep_sq])) if len(scores_sq) else np.array([], dtype=int)
    )
    order_pairs = (
        np.argsort(-(scores_pairs[keep_pairs]))
        if len(scores_pairs)
        else np.array([], dtype=int)
    )

    # Keep top-k interactions (after threshold)
    if len(scores_pairs):
        idx_pairs_local = np.where(keep_pairs)[0][order_pairs]
        if top_k_pairs is not None:
            idx_pairs_local = idx_pairs_local[: int(top_k_pairs)]
    else:
        idx_pairs_local = np.array([], dtype=int)

    # Materialize selected features
    added_list = []
    meta_added = []
    if len(scores_sq):
        idx_sq_keep = np.where(keep_sq)[0][order_sq]
        if len(idx_sq_keep):
            Z_sq = added_blocks[0][:, idx_sq_keep]
            added_list.append(Z_sq)
            meta_added += [meta_blocks[k] for k in idx_sq_keep.tolist()]

    if len(idx_pairs_local):
        Z_pairs_keep = []
        pairs_keep_meta = []
        base = 0
        for (Zchunk, _), Pmeta in zip(pair_chunks, chunk_meta):
            nloc = Zchunk.shape[1]
            loc_range = np.arange(base, base + nloc)
            m = np.intersect1d(idx_pairs_local, loc_range, assume_unique=False)
            if len(m):
                take = m - base
                Z_pairs_keep.append(Zchunk[:, take])
                pairs_keep_meta += [Pmeta[t] for t in take.tolist()]
            base += nloc
        if Z_pairs_keep:
            added_list.append(np.column_stack(Z_pairs_keep))
            meta_added += pairs_keep_meta

    if not added_list:
        return Xtr, Xte, {"squares": [], "pairs": [], "n_added": 0}

    Zall = np.column_stack(added_list)
    Xtr_aug = np.column_stack([Xtr, Zall])
    spec = {
        "squares": [t[1] for t in meta_added if t[0] == "square"],
        "pairs": [(t[1], t[2]) for t in meta_added if t[0] == "prod"],
        "n_added": int(Zall.shape[1]),
    }
    # Build test features in the same order as meta_added
    te_added = []
    for t in meta_added:
        if t[0] == "square":
            te_added.append((Xte[:, int(t[1])] ** 2).reshape(-1, 1))
        else:  # ("prod", i, j)
            te_added.append((Xte[:, int(t[1])] * Xte[:, int(t[2])]).reshape(-1, 1))
    if te_added:
        Zte = np.column_stack(te_added)
        Xte_aug = np.column_stack([Xte, Zte])
    else:
        Xte_aug = Xte

    print(
        f"[Preprocess] Polynomial features: added {spec['n_added']} (squares={len(spec['squares'])}, pairs={len(spec['pairs'])})"
    )
    return Xtr_aug, Xte_aug, spec


def encode_ordinal_as_score(Xtr, Xte, ord_idx, scale_to_unit=True):
    """Encode ordinal features as monotone scores learned on the training set.

    Levels observed in training are mapped to increasing ranks. Unseen levels in test are mapped
    to the maximum rank. Missing values are preserved. Scores can optionally be scaled to the unit interval.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        ord_idx (array-like): Indices of ordinal features.
        scale_to_unit (bool): If True, divide ranks by the maximum rank per feature.

    Returns:
        tuple: (Xtr_new, Xte_new, spec) with per-column mapping metadata.
    """
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    ordinal_maps = {}

    for j in ord_idx:
        vtr = Xtr[:, j]
        cats_sorted = np.unique(vtr[~np.isnan(vtr)])
        if len(cats_sorted) == 0:
            ordinal_maps[j] = {"levels": [], "K": 1, "scaled": scale_to_unit}
            continue

        K = max(len(cats_sorted) - 1, 1)

        ranks_tr = np.full(vtr.shape, np.nan, dtype=np.float32)
        valid_tr = ~np.isnan(vtr)
        if valid_tr.any():
            idx_tr = np.searchsorted(cats_sorted, vtr[valid_tr], side="left")
            in_bounds_tr = idx_tr < len(cats_sorted)
            match_tr = np.zeros_like(idx_tr, dtype=bool)
            if in_bounds_tr.any():
                match_tr[in_bounds_tr] = (
                    cats_sorted[idx_tr[in_bounds_tr]] == vtr[valid_tr][in_bounds_tr]
                )
            ranks_tr[valid_tr] = np.where(match_tr, idx_tr, K)

        vte = Xte[:, j]
        ranks_te = np.full(vte.shape, np.nan, dtype=np.float32)
        valid_te = ~np.isnan(vte)
        if valid_te.any():
            idx_te = np.searchsorted(cats_sorted, vte[valid_te], side="left")
            in_bounds_te = idx_te < len(cats_sorted)
            match_te = np.zeros_like(idx_te, dtype=bool)
            if in_bounds_te.any():
                match_te[in_bounds_te] = (
                    cats_sorted[idx_te[in_bounds_te]] == vte[valid_te][in_bounds_te]
                )
            ranks_te[valid_te] = np.where(match_te, idx_te, K)

        if scale_to_unit and K > 0:
            ranks_tr = ranks_tr / K
            ranks_te = ranks_te / K

        Xtr[:, j] = ranks_tr
        Xte[:, j] = ranks_te

        ordinal_maps[j] = {
            "levels": cats_sorted.tolist(),
            "K": int(K),
            "scaled": bool(scale_to_unit),
        }

    return Xtr, Xte, {"ordinal_maps": ordinal_maps}


def drop_useless_columns(Xtr, Xte, drop_n=0):
    """Drop the first N categorical columns but keep column 1 (state) in front.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        drop_n (int): Number of leading categorical columns to remove.

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated Xtr and Xte with column 1 preserved.
    """
    keep_col1 = Xtr[:, [1]]  # State can influence health
    Xtr = Xtr[:, drop_n:]
    Xte = Xte[:, drop_n:]
    Xtr = np.hstack([keep_col1, Xtr])
    Xte = np.hstack([keep_col1[: Xte.shape[0]], Xte])
    return Xtr, Xte


# ==========================================
# ==========================================


def preprocess2():
    """
    Full preprocessing pipeline for BRFSS-like data.

    Steps:
      0. Drop first N categorical columns (keep state).
      1. Replace BRFSS special codes with NaN.
      2. Optionally encode ordinal features as scores in [0,1].
      3. One-hot encode nominal features, optionally reduce with PCA.
      4. Add predictive NaN indicators.
      5. Impute missing values with smart_impute.
      6. Optionally add polynomial features for continuous vars.
      7. Standardize continuous features.
      8. Optionally prune highly correlated features.
      9. Optionally add bias column.
     10. save processed data to .npz.

    Args:
        Xtr_raw: Raw training feature matrix.
        Xte_raw: Raw test feature matrix.
        ytr_pm1: Training labels in {-1, 1} format.
        train_ids: Array of training sample identifiers.
        test_ids: Array of test sample identifiers.
        filename: Path to save the processed dataset (.npz).

    Returns:
        Tuple (Xtr, Xte, ytr_01): processed train/test features and labels converted to {0, 1}.
    """
    if config.PREPROCESSING:
        Xtr_raw, Xte_raw, ytr_pm1, train_ids, test_ids = helpers.load_csv_data(
            config.DATA_DIR
        )

        Xtr = np.asarray(Xtr_raw, np.float32)
        Xte = np.asarray(Xte_raw, np.float32)
        ytr_01 = metrics.to_01_labels(ytr_pm1)
        print(f"[Pipeline] Start | Xtr={Xtr.shape} Xte={Xte.shape} y={ytr_01.shape}")

        # --- Step 0: Drop useless columns ---
        drop_n = config.DROP_FIRST_N_CAT_COLS
        if drop_n > 0:
            Xtr, Xte = drop_useless_columns(Xtr, Xte, drop_n=drop_n)
        print(
            f"[Step0] Dropped useless cols (col 1 kept) | Xtr={Xtr.shape} Xte={Xte.shape}"
        )

        # --- Step 1: BRFSS special codes ---
        Xtr = replace_brfss_special_codes(Xtr)
        Xte = replace_brfss_special_codes(Xte)
        print(
            f"[Step1] BRFSS cleaned | NaN train={np.isnan(Xtr).sum()} NaN test={np.isnan(Xte).sum()}"
        )

        # --- Step 1.b: Ordinal encoding ---
        types = infer_feature_types(Xtr, max_unique_cat=config.LOW_CARD_MAX_UNIQUE)
        ord_idx = types["ordinal"]
        if len(ord_idx) > 0 and config.ORDINAL_ENCODE:
            Xtr, Xte, _ = encode_ordinal_as_score(
                Xtr, Xte, ord_idx=ord_idx, scale_to_unit=config.ORDINAL_SCALE_TO_UNIT
            )
            print(f"[Ordinals] encoded {len(ord_idx)} columns as monotone scores")

        # -- Step 3: One-hot encoding ---
        cont_idx_std = types["continuous"]
        cat_nom_idx = types["nominal"]

        print(
            f"[Step1.b] OHE | #nominal={len(cat_nom_idx)} drop_first={config.ONEHOT_DROP_FIRST} total_cap={config.MAX_ADDED_ONEHOT}"
        )

        Xtr, Xte, _, keep_idx, dummy_map = one_hot_encoding_selected(
            Xtr,
            Xte,
            cat_nom_idx,
            drop_first=config.ONEHOT_DROP_FIRST,
            total_cap=config.MAX_ADDED_ONEHOT,
        )
        ohe_start = len(keep_idx)
        ohe_end = Xtr.shape[1]
        ohe_cols = np.arange(ohe_start, ohe_end, dtype=int)  # for PCA later
        print(f"[Step2] OHE done | Xtr={Xtr.shape} Xte={Xte.shape}")

        # --- Step 4: PCA on OHE ---
        if config.PCA_Local is not None:
            Xtr, Xte, pca_local_spec = pca_local_on_ohe(
                Xtr, Xte, dummy_map, cfg=config.PCA_Local
            )

            # for later use
            pca_idx_list = []
            for meta in pca_local_spec.get("groups", {}).values():
                pca_idx_list.extend(meta.get("pca_component_idx", []))
            pca_spec = {
                "k": int(pca_local_spec.get("total_k", 0)),
                "pca_component_idx": pca_idx_list,
            }
            print(
                f"[Step3] PCA-Local on OHE | Xtr={Xtr.shape} Xte={Xte.shape} | total_k={pca_spec['k']}"
            )

        else:  # Global PCA but not use at the end of project
            pca_var = config.PCA_VAR
            pca_k = config.PCA_K
            if len(ohe_cols) > 0:
                Xtr, Xte, pca_spec = pca(
                    Xtr,
                    Xte,
                    cols=ohe_cols,
                    n_components=(None if isinstance(pca_k, float) else pca_k),
                    variance_ratio=(pca_var if isinstance(pca_var, float) else None),
                    replace=True,
                )
            else:
                pca_spec = {"k": 0, "pca_component_idx": []}
            print(
                f"[Step3] PCA on OHE | Xtr={Xtr.shape} Xte={Xte.shape} | k={pca_spec.get('k', 0)}"
            )

        # --- Step 5: NaN indicators ---
        n_before = Xtr.shape[1]
        Xtr, Xte = filter_add_predictive_nan_indicators(
            Xtr,
            Xte,
            ytr_01,
            threshold=config.NAN_INDICATOR_MIN_ABS_CORR,
            top_k=config.NAN_INDICATOR_TOPK,
            min_prevalence=config.NAN_INDICATOR_MIN_PREV,
            max_prevalence=config.NAN_INDICATOR_MAX_PREV,
        )
        print(
            f"[Step4] NaN indicators | +{Xtr.shape[1] - n_before} cols | Xtr={Xtr.shape} Xte={Xte.shape}"
        )

        # --- Step 5: Imputation ---
        Xtr, Xte = smart_impute(
            Xtr,
            Xte,
            skew_rule=0.5,
            allnan_fill_cont=0.0,
            allnan_fill_nom=-1.0,  # fallback values
            allnan_fill_bin=0.0,
        )

        print(f"[Step5] Impute Done")
        assert (np.isnan(Xtr).sum() == 0) and (
            np.isnan(Xte).sum() == 0
        ), "There are still NaNs in Xtr or Xte after imputation!"  # sanity check

        # --- Step 6: Polynomial feature expansion ---
        if config.POLY_ENABLE:
            cont_idx_base = cont_idx_std

            Xtr, Xte, poly_spec = polynomial_features(
                Xtr,
                Xte,
                ytr_01,
                cont_idx=cont_idx_base,
                add_squares_cont=config.POLY_ADD_SQUARES_CONT,
                add_inter_within_cont=config.POLY_ADD_INTER_CONT,
                top_k_pairs=config.POLY_TOPK_PAIRS,
                min_abs_corr=config.POLY_MIN_ABS_CORR,
            )
            print(
                f"[Step6] Polynomial features | added={poly_spec.get('n_added', 0)} | Xtr={Xtr.shape} Xte={Xte.shape}"
            )

        # -- Step 7: Standardization ---
        Xtr, Xte, cont_idx_std = standardize(
            Xtr, Xte, cont_idx=cont_idx_std, return_updated_idx=True
        )
        print(f"[Step7] Standardize | Xtr={Xtr.shape} Xte={Xte.shape}")

        # --- Step 8: Correlation-based pruning ---
        if config.PRUNE_CORR_THRESHOLD is not None and len(cont_idx_std) > 0:
            Xtr, Xte, dropped_g, kept_g = remove_highly_correlated_continuous(
                Xtr,
                Xte,
                cont_idx=cont_idx_std,
                y_train=ytr_01,
                threshold=config.PRUNE_CORR_THRESHOLD,
            )
            kept_g = np.asarray(kept_g, dtype=int)
            print(
                f"[Step8] Corr prune | thr={config.PRUNE_CORR_THRESHOLD} | dropped={len(dropped_g)}"
            )
        else:
            print("[Step8] Corr prune | skipped")

        # --- Step 9: Bias term ---
        if config.ADD_BIAS:
            Xtr = np.hstack([np.ones((Xtr.shape[0], 1), dtype=np.float32), Xtr])
            Xte = np.hstack([np.ones((Xte.shape[0], 1), dtype=np.float32), Xte])
            print(f"[Step9] Bias added | Xtr={Xtr.shape} Xte={Xte.shape}")

        # --- Step 10: Save processed dataset ---
        save(Xtr, Xte, ytr_01, train_ids, test_ids)

        print(f"[Pipeline] Done | Xtr={Xtr.shape} Xte={Xte.shape}")
    else:
        Xtr, Xte, ytr_01, train_ids, test_ids = load_preproc_data()

    return Xtr, Xte, ytr_01, train_ids, test_ids


# ==========================================
# ==========================================


def save(Xtr, Xte, ytr, train_ids, test_ids, filename=config.PREPROC_DATA_PATH):
    """
    Save arrays to a compressed .npz archive.

    Args:
        Xtr (np.ndarray): Training matrix.
        Xte (np.ndarray): Test matrix.
        ytr (np.ndarray): Training labels.
        train_ids (np.ndarray): Training identifiers.
        test_ids (np.ndarray): Test identifiers.
        filename (str): Output file path.

    Returns:
        None
    """
    np.savez_compressed(
        filename,
        X_train=Xtr,
        X_test=Xte,
        y_train=ytr,
        train_ids=train_ids,
        test_ids=test_ids,
    )
    print(f"[Step10] Saved -> {filename}")


def load_preproc_data(filename=config.PREPROC_DATA_PATH):
    """Load best hyperparameters from disk."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    npz = np.load(filename)
    Xtr = npz["X_train"]
    Xte = npz["X_test"]
    ytr_01 = npz["y_train"]
    train_ids = npz["train_ids"]
    test_ids = npz["test_ids"]
    print(f"[Loaded] Preprocessed data from -> {filename}")
    return Xtr, Xte, ytr_01, train_ids, test_ids
