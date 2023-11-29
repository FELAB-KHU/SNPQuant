import pickle as pickle

import math
import os
import numpy as np
import pandas as pd
import torch


two_sided_tstat_threshold_dict = {0.1: 1.645, 0.05: 1.96, 0.01: 2.575}
one_sided_tstat_threshold_dict = {0.1: 1.28, 0.05: 1.645, 0.01: 2.33}


def get_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def df_empty(columns, dtypes, index=None):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def binary_one_hot(y, device=None):
    y = y.to("cpu")
    y_onehot = torch.FloatTensor(y.shape[0], 2)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    y_onehot = y_onehot.to(y.dtype)
    if device is not None:
        y_onehot = y_onehot.to(device)
    return y_onehot


# def cross_entropy_loss(pred_prob, true_label):
#     pred_prob = np.array(pred_prob)
#     x = np.zeros((len(pred_prob), 2))
#     x[:, 1] = pred_prob
#     x[:, 0] = 1 - x[:, 1]
#     pred_prob = x

#     true_label = np.array(true_label)
#     y = np.zeros((len(true_label), 2))
#     y[np.arange(true_label.size), true_label] = 1
#     true_label = y

#     loss = -np.sum(true_label * np.log(pred_prob)) / len(pred_prob)
#     return loss

def cross_entropy_loss(pred_prob, true_label):
    # Convert pred_prob to a 2-column format for binary classification
    pred_prob = np.array(pred_prob)
    prob_matrix = np.zeros((len(pred_prob), 2))
    prob_matrix[:, 1] = pred_prob
    prob_matrix[:, 0] = 1 - prob_matrix[:, 1]

    # Clip probabilities to avoid log(0)
    prob_matrix = np.clip(prob_matrix, 1e-7, 1 - 1e-7)

    # Convert true_label to a one-hot encoded format
    true_label = np.array(true_label)
    label_matrix = np.zeros((len(true_label), 2))
    label_matrix[np.arange(len(true_label)), true_label] = 1

    # Calculate the cross-entropy loss
    loss = -np.sum(label_matrix * np.log(prob_matrix)) / len(pred_prob)
    return loss


def rank_corr(df, col1, col2, method="spearman"):
    if method == "spearman":
        col1_series = df[col1].rank(method="average", ascending=False)
        col2_series = df[col2].rank(method="average", ascending=False)
    else:
        col1_series = df[col1]
        col2_series = df[col2]

    return col2_series.corr(col1_series, method=method)


def rank_normalization(c: pd.Series):
    rank = c.rank(ascending=True)
    normed_rank = 2.0 * (rank - rank.min()) / (rank.max() - rank.min()) - 1.0
    normed_rank.fillna(0, inplace=True)
    return normed_rank


# def calculate_test_log(pred_prob, label):
#     pred = np.where(pred_prob > 0.5, 1, 0)
#     num_samples = len(pred)
#     TP = np.nansum(pred * label, dtype=np.int64) / num_samples
#     TN = np.nansum((pred - 1) * (label - 1)) / num_samples
#     FP = np.abs(np.nansum(pred * (label - 1))) / num_samples
#     FN = np.abs(np.nansum((pred - 1) * label)) / num_samples
#     test_log = {
#         "diff": 1.0 * ((TP + FP) - (TN + FN)),
#         "loss": cross_entropy_loss(pred_prob, label),
#         "accy": 1.0 * (TP + TN),
#         "MCC": np.nan
#         if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0
#         else 1.0
#         * (TP * TN - FP * FN)
#         / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
#     }
#     return test_log

def calculate_test_log(pred_prob, label):
    pred = np.where(pred_prob > 0.5, 1, 0)
    num_samples = len(pred)
    
    TP = np.sum(np.where((pred == 1) & (label == 1), 1, 0)) / num_samples
    TN = np.sum(np.where((pred == 0) & (label == 0), 1, 0)) / num_samples
    FP = np.sum(np.where((pred == 1) & (label == 0), 1, 0)) / num_samples
    FN = np.sum(np.where((pred == 0) & (label == 1), 1, 0)) / num_samples
    
    mcc_denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = (TP * TN - FP * FN) / mcc_denom if mcc_denom != 0 else np.nan
    
    test_log = {
        "diff": (TP + FP) - (TN + FN),
        "loss": cross_entropy_loss(pred_prob, label),
        "accy": TP + TN,
        "MCC": MCC,
    }
    return test_log


def save_pkl_obj(obj, path):
    with open(path, "wb+") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_latex_w_turnover(pf_df, cut=10):
    assert len(pf_df) == cut + 2
    pf_df = pf_df.rename(columns={"ret": "Ret", "std": "Std"})
    pf_df = pf_df.round(3)
    pf_df = pf_df.set_index(
        pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L", "Turnover"])
    )
    latex = (pf_df.iloc[: cut + 1]).to_latex()

    latex_list = latex.splitlines()

    latex_list.insert(len(latex_list) - 2, "\hline")
    line = (
        "\multicolumn{4}{c}{Turnover: "
        + str(int(pf_df.loc["Turnover", "SR"] * 100))
        + "\%}"
        + "\\\\"
    )
    latex_list.insert(len(latex_list) - 2, line)
    latex_new = "\n".join(latex_list)
    return latex_new


def pvalue_surfix(pv):
    pv_surfix = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.1 else ""
    return pv_surfix


def add_star_by_pvalue(value, pvalue, decimal=2):
    return f"{value:.{decimal}f}" + pvalue_surfix(pvalue)


def star_significant_value_by_sample_num(val, sample_num, one_sided=True, decimal=2):
    tstat = val * np.sqrt(sample_num)
    res = add_stars_to_value_by_tstat(val, tstat, one_sided, decimal)
    return res


def add_stars_to_value_by_tstat(value, tstat, one_sided, decimal):
    if one_sided:
        if tstat > one_sided_tstat_threshold_dict[0.01]:
            res = f"{value:.{decimal}f}***"
        elif tstat > one_sided_tstat_threshold_dict[0.05]:
            res = f"{value:.{decimal}f}**"
        elif tstat > one_sided_tstat_threshold_dict[0.1]:
            res = f"{value:.{decimal}f}*"
        else:
            res = f"{value:.{decimal}f}"
    else:
        tstat = np.abs(tstat)
        if tstat > two_sided_tstat_threshold_dict[0.01]:  # %99.5 (0.01) 0.5919
            res = f"{value:.{decimal}f}***"
        elif tstat > two_sided_tstat_threshold_dict[0.05]:  # %97.5 (0.05) 0.44965
            res = f"{value:.{decimal}f}**"
        elif tstat > two_sided_tstat_threshold_dict[0.1]:  # %95 (0.1) 0.3774
            res = f"{value:.{decimal}f}*"
        else:
            res = f"{value:.{decimal}f}"
    return res
