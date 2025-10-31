# Cross-validation and training utilities
import numpy as np
import metrics
import implementations
import config


# stratified_kfold_indices
def create_stratified_folds(ytr_01, k=config.N_FOLDS, random_seed=config.RNG_SEED):
    """
    Create stratified k-fold cross-validation indices.

    Args:
        y_binary: Binary labels (0 or 1)
        k: Number of folds
        random_seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples, one per fold
    """
    rng = np.random.RandomState(random_seed)
    y = np.asarray(ytr_01, dtype=int)

    # Seperate indices by class
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Shuffle indices
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Splite each class into k equal parts
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    folds = []
    n_samples = len(ytr_01)

    for fold_idx in range(k):
        # Validation set:
        val_idx = np.concatenate([pos_folds[fold_idx], neg_folds[fold_idx]])
        rng.shuffle(val_idx)  # Mix positive and negative samples

        # Training set:
        is_train = np.ones(n_samples, dtype=bool)
        is_train[val_idx] = False
        train_idx = np.where(is_train)[0]

        folds.append((train_idx, val_idx))

    return folds


# best_threshold_by_f1()
def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal classification threshold that maximizes F1 score.

    Args:
        y_true: Binary ground truth labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities/scores, shape (n_samples,)

    Returns:
        tuple: (optimal_threshold, precision, recall, f1_score)
            - optimal_threshold: Best threshold for classification
            - precision: Precision at optimal threshold
            - recall: Recall at optimal threshold
            - f1_score: F1 score at optimal threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Sort scores in descending order
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    # Total number of positive samples
    n_positives = np.sum(y_true == 1)

    # Compute cumulative TP and FP counts at each threshold
    cumulative_tp = np.cumsum(y_true_sorted == 1)
    cumulative_fp = np.cumsum(y_true_sorted == 0)

    # Compute precision, recall, and F1 at each threshold
    # Add small epsilon to avoid division by zero
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + config.EPS)
    recall = cumulative_tp / (n_positives + config.EPS)
    f1_scores = 2 * precision * recall / (precision + recall + config.EPS)

    # Find index with maximum F1 score
    best_idx = np.argmax(f1_scores)

    # Use midpoint between consecutive scores as threshold
    # This ensures consistent classification for edge cases
    if best_idx < len(y_prob_sorted) - 1:
        optimal_threshold = (
            y_prob_sorted[best_idx] + y_prob_sorted[best_idx + 1]
        ) / 2.0
    else:
        # For the last element, use a value slightly below it
        optimal_threshold = y_prob_sorted[best_idx] - config.EPS

    return (
        float(optimal_threshold),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(f1_scores[best_idx]),
    )


def schedule_onecycle(lr0, t, T, pct_start=0.3, max_lr_ratio=10.0, min_lr_ratio=1e-3):
    """
    One-Cycle learning rate policy (linear variant).
    - Warmup phase: increase LR from lr0 to lr0*max_lr_ratio over pct_start*T iters.
    - Cooldown phase: decrease LR from lr0*max_lr_ratio to lr0*min_lr_ratio over the rest.
    Args:
        lr0 (float): Base learning rate.
        t (int): Current iteration index (0-based).
        T (int): Total iterations.
        pct_start (float): Fraction of iterations for warmup.
        max_lr_ratio (float): Peak LR relative to lr0.
        min_lr_ratio (float): Final LR floor relative to lr0.
    Returns:
        float: Learning rate at iteration t.
    """
    T = max(1, int(T))
    warm = max(1, int(float(pct_start) * T))
    peak = float(lr0) * float(max_lr_ratio)
    floor = float(lr0) * float(min_lr_ratio)
    if t < warm:
        # linear ramp up from lr0 to peak
        return float(lr0) + (peak - float(lr0)) * (float(t + 1) / float(warm))
    # linear decay from peak to floor
    remain = max(1, T - warm)
    alpha = min(1.0, max(0.0, float(t - warm + 1) / float(remain)))
    lr = peak - (peak - floor) * alpha
    # clamp
    return max(min(lr, peak), floor)


def schedule_nagfree(lr0, t, T, warmup_frac=0.05, min_lr_ratio=1e-2):
    """
    Warmup + inverse square-root decay ("NAG-free style").
    - Linear warmup from 0 → lr0 during the first warmup_frac * T steps.
    - Then decay as lr = lr0 / sqrt(1 + (t - warmup)), clipped at lr0 * min_lr_ratio.
    This schedule is simple and robust: warmup stabilizes early updates and
    1/sqrt(t) decay maintains useful step sizes for long horizons.
    Args:
        lr0 (float): Initial/base learning rate.
        t (int): Current iteration index (0-based).
        T (int): Total iterations.
        warmup_frac (float): Fraction of T used for linear warmup.
        min_lr_ratio (float): Floor as a fraction of lr0 to avoid vanishing steps.
    Returns:
        float: Learning rate at iteration t.
    """
    import math

    warmup = max(1, int(T * float(warmup_frac)))
    if t < warmup:
        return lr0 * (float(t + 1) / float(warmup))
    tau = t - warmup
    lr = lr0 / math.sqrt(1.0 + float(tau))
    return max(lr, lr0 * float(min_lr_ratio))


def cv_train_and_eval(
    y_tr_01,
    X_tr,
    lam,
    gam,
    max_iters,
    use_adam,
    schedule_name,
    early_stopping,
    patience,
    tol,
):  # (args):
    """
    Cross-validated training and evaluation with advanced optimization options.

    Performs stratified K-fold cross-validation for regularized logistic regression
    with support for multiple optimizers (Adam/GD), learning rate schedules
    (nagfree/onecycle), and early stopping. For each fold, trains on the
    training split and collects out-of-fold predictions. Aggregates all validation
    predictions to find the globally optimal F1-maximizing threshold, then
    evaluates each fold using this shared threshold.

    Args:
        args (tuple): Configuration tuple containing:
            - y_tr_01 (np.ndarray): Binary labels (0/1) for entire training set
            - X_tr (np.ndarray): Feature matrix for entire training set
            - lam (float): L2 regularization strength (lambda)
            - gam (float): Initial learning rate (gamma)
            - max_iters (int): Maximum training iterations per fold
            - use_adam (int/bool): If True/1, use Adam optimizer; else use GD
            - schedule_name (str): Learning rate schedule ('nagfree', 'onecycle', or None)
            - early_stopping (int/bool): If True/1, enable early stopping
            - patience (int): Number of iterations to wait for improvement before stopping
            - tol (float): Minimum improvement threshold for early stopping

    Returns:
        dict: Cross-validation results containing:
            - 'lambda' (float): Regularization parameter used
            - 'gamma' (float): Initial learning rate used
            - 'max_iters' (int): Maximum iterations used
            - 'mean_accuracy' (float): Average accuracy across all folds
            - 'std_accuracy' (float): Standard deviation of accuracy across folds
            - 'mean_precision' (float): Average precision across all folds
            - 'std_precision' (float): Standard deviation of precision across folds
            - 'mean_recall' (float): Average recall across all folds
            - 'std_recall' (float): Standard deviation of recall across folds
            - 'mean_f1' (float): Average F1 score across all folds
            - 'std_f1' (float): Standard deviation of F1 across folds
            - 'optimal_threshold' (float): Global F1-optimal classification threshold
            - 'adam' (bool): Whether Adam optimizer was used
            - 'schedule' (str): Learning rate schedule used ('nagfree', 'onecycle', or 'none')
    """

    folds = create_stratified_folds(y_tr_01)

    if schedule_name == "nagfree":
        schedule = schedule_nagfree
    elif schedule_name == "onecycle":
        schedule = schedule_onecycle
    else:
        schedule = None

    per_fold_probs, per_fold_idx = [], []

    for tr_idx, va_idx in folds:
        w0 = np.zeros(X_tr.shape[1], dtype=np.float32)

        w, _ = implementations.reg_logistic_regression(
            y_tr_01[tr_idx],
            X_tr[tr_idx],
            lam,
            w0,
            max_iters=max_iters,
            gamma=gam,
            adam=bool(use_adam),
            schedule=schedule,
            early_stopping=bool(early_stopping),
            patience=patience,
            tol=tol if tol > 0 else config.TOL_DEFAULT,
            verbose=False,
            val_data=(y_tr_01[va_idx], X_tr[va_idx]) if early_stopping else None,
        )

        probs_va = implementations.sigmoid(X_tr[va_idx].dot(w))
        per_fold_probs.append(probs_va)
        per_fold_idx.append(va_idx)

    va_idx_concat = np.concatenate(per_fold_idx)
    probs_concat = np.concatenate(per_fold_probs)
    y_val_concat = y_tr_01[va_idx_concat]
    best_thr, _, _, _ = find_optimal_threshold(y_val_concat, probs_concat)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    for probs_va, va_idx in zip(per_fold_probs, per_fold_idx):
        preds = (probs_va >= best_thr).astype(int)
        y_va = y_tr_01[va_idx]
        acc_list.append(metrics.accuracy_score(y_va, preds))
        p, r, f1 = metrics.precision_recall_f1(y_va, preds)
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)

    return {
        "lambda": float(lam),
        "gamma": float(gam),
        "max_iters": int(max_iters),
        "mean_accuracy": float(np.mean(acc_list)),
        "std_accuracy": float(np.std(acc_list)),
        "mean_precision": float(np.mean(prec_list)),
        "std_precision": float(np.std(prec_list)),
        "mean_recall": float(np.mean(rec_list)),
        "std_recall": float(np.std(rec_list)),
        "mean_f1": float(np.mean(f1_list)),
        "std_f1": float(np.std(f1_list)),
        "optimal_threshold": float(best_thr),
        "adam": bool(use_adam),
        "schedule": schedule_name if schedule_name else "none",
    }
