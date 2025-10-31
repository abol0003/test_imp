import os
import time
import numpy as np

import helpers 
# clearer than : from helpers import * As we use suffixes.
import implementations 
# avoid using Aliases: import implementations as impl
import config 
import preprocessing 
import metrics 
import tuning
import cv_utils

os.makedirs(config.PICT_DIR, exist_ok=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)
np.random.seed(config.RNG_SEED)

def train_final_model(X_tr, y_tr_01, best_lambda, best_gamma, use_adam=None, schedule_name=None):
    """Train the final logistic model on the entire training dataset.

    Depending on the configuration, the function trains either a standard logistic
    regression model or its NAG-Free variant, using the best hyperparameters found
    during tuning.

    Args:
        X_tr (np.ndarray): Training feature matrix.
        y_tr_01 (np.ndarray): Binary training labels (0/1).
        best_lambda (float): Regularization parameter.
        best_gamma (float): Learning rate.
        use_adam (bool, optional): Use Adam optimizer if True.
        schedule_name (str, optional): Learning rate schedule name.

    Returns:
        np.ndarray: Final trained model weights.
    """
    t_final = time.time()
    w0 = np.zeros(X_tr.shape[1], dtype=np.float32)

    final_use_adam = use_adam if use_adam is not None else config.USE_ADAM_DEFAULT
    final_sched_name = schedule_name if schedule_name is not None else config.SCHEDULE_DEFAULT
    if final_sched_name == "nagfree":
        schedule = cv_utils.schedule_nagfree
    elif final_sched_name == "onecycle":
        schedule = cv_utils.schedule_onecycle
    else:
        schedule = None
    print(f"[Final Training] Using schedule: {final_sched_name}, Adam: {final_use_adam} with lambda={best_lambda}, gamma={best_gamma}")
    w_final, final_loss = implementations.reg_logistic_regression(
        y_tr_01, X_tr, best_lambda, w0,
        max_iters=config.FINAL_MAX_ITERS,
        gamma=best_gamma if best_gamma is not None else 1e-2,
        adam=final_use_adam, schedule=schedule,
        early_stopping=config.EARLY_STOP_DEFAULT,
        patience=config.PATIENCE_DEFAULT, tol=config.TOL_DEFAULT,
        verbose=False,
    )
    np.save(config.SAVE_WEIGHTS, w_final)
    print(f"[Saved] Final weights → {config.SAVE_WEIGHTS}")

    print(f"[Final] loss (unpenalized) = {final_loss:.6f}")
    print(f"[Final Training] {time.time() - t_final:.1f}s")
    return w_final


def make_submission(X_te, w_final, best_thr, test_ids):
    """Generete predications and submission file."""
    yprob_te = implementations.sigmoid(X_te.dot(w_final))
    ypred_01_te = (yprob_te >= best_thr).astype(int)
    ypred_pm1_te = metrics.to_pm1_labels(ypred_01_te)
    helpers.create_csv_submission(test_ids, ypred_pm1_te, config.OUTPUT_PRED)
    print(f"[Submission] saved -> {config.OUTPUT_PRED}")


def main():
    t0 = time.time()
    Xtr, Xte, ytr_01, train_ids, test_ids = preprocessing.preprocess2()
    print(f"[Preprocessing] {time.time() - t0:.1f}s")

    #==========================================
    t = time.time()   
    #seperation of concerns : 1)tuning.py should contain the tuning logic 2) run.py simply orchestrate everything based on config.py
    best_params = tuning.tune(Xtr, ytr_01) 
    print(f"[Tuning] {time.time() - t:.1f}s")

    #==========================================
    # Extract parameters 
    best_lambda = best_params['lambda']
    best_gamma = best_params['gamma']
    best_threshold = best_params['optimal_threshold']
    best_adam= best_params['adam']
    best_schedule= best_params['schedule']
    #...
    mean_f1= best_params['mean_f1']
    print(f"Best hyperparameters found: lambda={best_lambda}, gamma={best_gamma}, threshold={best_threshold}, mean_f1={mean_f1}")
    #Final training + submission
    t = time.time()   
    if config.SUBMISSION:
        w_final = train_final_model(
            Xtr, ytr_01, best_lambda, best_gamma,
            use_adam=best_adam, schedule_name=best_schedule
        )
    else:
        w_final = np.load(config.SAVE_WEIGHTS)
    make_submission(Xte, w_final, best_threshold, test_ids)
    
    print(f"[Final Training + Submission] {time.time() - t:.1f}s")
    print(f"[TOTAL] {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
