from custom_losses.loss_functions import *
from custom_losses.loss_wrappers import loss_wrapper, cat_wrapper
import optuna
from constants import MIN_OPTIMAL_METRICS, MAX_OPTIMAL_METRICS, CUSTOM_OBJECTIVES, CUSTOM_PARAMS
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
import numpy as np
import scipy.optimize as opt
from scipy.signal import savgol_filter


def is_max_optimal(metric: str) -> bool:
    metric = metric.lower().strip()

    if metric in MAX_OPTIMAL_METRICS:
        return True
    elif metric in MIN_OPTIMAL_METRICS:
        return False
    else:
        ValueError(f'Unknown metric: {metric}')


def create_objective(objective_name, y_true, params, algorithm):

    if objective_name == 'LDAM':
        loss = LDAM_loss(y_true, params['LDAM_c'])

    elif objective_name == 'Focal':
        loss = Focal_loss(y_true, params['Focal_gamma'])

    elif objective_name == 'LA':
        loss = LA_loss(y_true, params['LA_tau'])

    # Wrappers
    if algorithm in ['XGBoost', 'LightGBM']:
        objective = loss_wrapper(loss, clip=True)

    elif algorithm == 'CatBoost':
        objective = cat_wrapper(loss)

    return objective


def unpack_params(trial: optuna.Trial, config, algorithm, y_true):
    params = {}

    for param, value in config.params.model_dump().items():
        if type(value) is list:
            distribution = trial.suggest_categorical(param, value)
        elif type(value) is tuple:
            if type(value[0]) is float:
                distribution = trial.suggest_float(param, *value)
            else:
                distribution = trial.suggest_int(param, *value)
        else:
            distribution = value

        params[param] = distribution

    # Custom Loss
    if params['objective'] in CUSTOM_OBJECTIVES:
        params.pop('scale_pos_weight')
        objective = create_objective(params['objective'], y_true, params, algorithm)

        if algorithm == 'XGBoost':
            config.obj = objective.get_gradient
            params.pop('objective')

        elif algorithm == 'CatBoost':
            params['objective'] = objective

        elif algorithm == 'LightGBM':
            params['objective'] = objective.get_gradient

    # Remove unnecessary custom params
    for param in CUSTOM_PARAMS:
        params.pop(param)

    return params, config


def get_optimal_threshold(y_train, y_train_proba):
    """
    - Find optimal threshold using CV on training set (optimizing MCC)
    - Loss function was trained using PR-AUC which is threshold-invariant, thus
    using the default 0.5 threshold likely doesn't yield the best results.
    """

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds = []

    for train_idx, val_idx in skf.split(np.zeros(len(y_train)), y_train):
        y_val = y_train[val_idx]
        y_val_proba = y_train_proba[val_idx]

        res = opt.minimize_scalar(
            lambda p: -matthews_corrcoef(y_val, (y_val_proba >= p).astype(int)),
            bounds=(0, 1),
            method='bounded'
        )
        thresholds.append(res.x)

    return np.mean(thresholds)


def get_metrics(y_test, y_test_proba, optimal_threshold):

    test_preds = (y_test_proba >= optimal_threshold).astype(int)

    metrics = {
        'roc': roc_auc_score(y_test, test_preds),
        'pr': average_precision_score(y_test, y_test_proba),
        'acc': accuracy_score(y_test, test_preds),
        'bacc': balanced_accuracy_score(y_test, test_preds),
        'pre': precision_score(y_test, test_preds),
        'rec': recall_score(y_test, test_preds),
        'f1': f1_score(y_test, test_preds),
        'mcc': matthews_corrcoef(y_test, test_preds),
        'brier': brier_score_loss(y_test, y_test_proba)
    }

    cm = confusion_matrix(y_test, test_preds)

    return metrics, cm


def decision_curve(y_true, y_prob, thresholds):
    """
    y_true: array of 0/1 outcomes
    y_prob: predicted probabilities
    thresholds: array of threshold probabilities
    """
    N = len(y_true)
    net_benefit = []

    for pt in thresholds:
        preds = (y_prob >= pt).astype(int)

        TP = np.sum((preds == 1) & (y_true == 1))
        FP = np.sum((preds == 1) & (y_true == 0))

        nb = (TP/N) - (FP/N)*(pt/(1-pt))
        net_benefit.append(nb)

    return savgol_filter(np.array(net_benefit), 51, 2)
