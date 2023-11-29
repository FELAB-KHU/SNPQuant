import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, r2_score
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

from Data.dgp_config import CACHE_DIR
from Misc import config as cf
from Misc import utilities as ut


class SMRegression(object):
    def __init__(
        self,
        regression_type,
        logit_threshold=0.0,
        rank_norm_x=False,
        ols_rank_norm_y=True,
    ):
        assert regression_type in ["logit", "ols"]
        self.regression_type = regression_type
        self.logit_threshold = logit_threshold
        self.rank_norm_x = rank_norm_x
        self.ols_rank_norm_y = ols_rank_norm_y
        self.r2_name = "McFadden $R^2$" if regression_type == "logit" else "$R^2$"
        self.mod = None
        self.is_mean = None
        self.is_mean_by_stock: pd.Series = None

    def transform_y(self, y):
        y = y.copy()
        if self.regression_type == "logit":
            print("Logistic regression: Converting y to 0 and 1")
            return np.where(y > self.logit_threshold, 1, 0)
        if self.regression_type == "ols":
            if self.ols_rank_norm_y:
                print("Linear regression: Rank normalizing y")
                return np.array(y.groupby("Date").transform(ut.rank_normalization))
            else:
                return np.array(y)
        raise ValueError

    def transform_x(self, x):
        x = x.copy()
        if self.rank_norm_x:
            print("Rank Normalizing X")
            x = x.groupby("Date").transform(ut.rank_normalization)
        return x

    def calculate_is_mean_by_stock(
        self, X: pd.DataFrame, y: np.array, is_mean_min_sample: int
    ):
        x = X.copy()
        x.reset_index(inplace=True)
        x["is_mean"] = y
        is_mean = x[["is_mean", "StockID"]].groupby("StockID").mean()
        is_mean = is_mean[
            (
                (x[["is_mean", "StockID"]].groupby("StockID").count())["is_mean"]
                >= is_mean_min_sample
            )
        ]
        print(
            f"IS mean by stock has {is_mean['is_mean'].isna().sum()}/{len(is_mean)} nan values"
        )
        self.is_mean_by_stock = is_mean.copy()

    def _fit_logit(self, X: pd.DataFrame, y: np.array):
        X = X.copy()
        X["const"] = 1
        y = self.transform_y(y)
        self.is_mean = np.nanmean(y)
        return Logit(y, X, missing="drop").fit(method="lbfgs", maxiter=200, disp=0)

    def _fit_ols(self, X: pd.DataFrame, y):
        X = X.copy()
        X["const"] = 1
        y = self.transform_y(y)
        self.is_mean = np.nanmean(y)
        return OLS(y, X, missing="drop").fit(disp=0)

    def fit(self, X, y):
        X = self.transform_x(X)
        X["const"] = 1
        self.mod = None
        if self.regression_type == "logit":
            self.mod = self._fit_logit(X, y)
        elif self.regression_type == "ols":
            self.mod = self._fit_ols(X, y)
        else:
            raise ValueError()
        return {
            "model": self.mod,
            "params": self.mod.params,
            "tstats": self.mod.tvalues,
            "pvalues": self.mod.pvalues,
        }

    def predict(self, X):
        X = self.transform_x(X)
        X["const"] = 1
        return np.array(self.mod.predict(X))

    def _fill_by_is_mean_per_stock(self, oos_X):
        null_pred = pd.DataFrame(
            data={"pred": [np.nan] * len(oos_X)}, index=oos_X.index
        )
        null_pred.reset_index(inplace=True)
        null_pred.set_index("StockID", inplace=True)
        null_pred["pred"] = self.is_mean_by_stock["is_mean"]
        print(
            f"Null Pred (is_mean) by stock has {null_pred['pred'].isna().sum()}/{len(null_pred)} nan values"
        )
        null_pred.fillna(self.is_mean, inplace=True)
        return null_pred["pred"]

    def _mcfadden_r2(self, oos_X, oos_y, use_is_mean_per_stock):
        oos_y = self.transform_y(oos_y)
        mod_pred = self.predict(oos_X)
        mod_ll = log_loss(oos_y.flatten(), mod_pred.flatten())
        if use_is_mean_per_stock:
            null_pred = self._fill_by_is_mean_per_stock(oos_X)
            null_ll = log_loss(oos_y.flatten(), np.array(null_pred).flatten())
        else:
            null_ll = log_loss(oos_y.flatten(), [self.is_mean] * len(oos_y))
        print(f"Calculating McFadden R2: mod_ll {mod_ll} null_ll {null_ll}")
        return 1 - mod_ll / null_ll

    def _ols_r2(self, oos_X, oos_y, use_is_mean_per_stock=True):
        oos_y = self.transform_y(oos_y)
        if use_is_mean_per_stock:
            null_pred = np.array(self._fill_by_is_mean_per_stock(oos_X))
        else:
            null_pred = [oos_y.mean()] * len(oos_y)

        mod_pred = self.predict(oos_X)
        print(
            f"sklearn R2: {r2_score(y_true=oos_y.flatten(), y_pred=mod_pred.flatten())}"
        )
        oos_y, mod_pred, null_pred = (
            np.array(oos_y).flatten(),
            np.array(mod_pred).flatten(),
            np.array(null_pred).flatten(),
        )
        r2 = (
            1
            - ((oos_y - mod_pred) * (oos_y - mod_pred)).sum()
            / ((oos_y - null_pred) * (oos_y - null_pred)).sum()
        )
        print(f"Calculated R2: {r2}")
        return r2

    def oos_r2(self, oos_X, oos_y, use_is_mean_per_stock=True):
        if self.regression_type == "logit":
            return self._mcfadden_r2(oos_X, oos_y, use_is_mean_per_stock)
        if self.regression_type == "ols":
            return self._ols_r2(oos_X, oos_y)
        raise ValueError

    def is_r2(self):
        if self.regression_type == "logit":
            return self.mod.prsquared
        if self.regression_type == "ols":
            return self.mod.rsquared


def cnn_pred_on_monthly_stock_char(pw, regression_type):
    assert regression_type in ["logit", "ols"]
    regr = SMRegression(
        regression_type, logit_threshold=0.5, rank_norm_x=True, ols_rank_norm_y=True
    )

    def generate_latex_table_for_cs_regression_res(columns, res_dict):
        res_df = pd.DataFrame(
            columns=[c for c in columns if c != "const"], index=list(res_dict.keys())
        )
        for idx in res_df.index:
            res, r2 = res_dict[idx]
            for c in columns:
                res_df.loc[idx, c] = f"{res['params'][c]:.2f}"
            res_df.loc[idx, regr.r2_name] = f"{r2 * 100:.2f}"
        res_df = res_df.T
        print(res_df)
        return res_df

    res_dict = {}
    stock_char_df = load_cnn_and_monthly_stock_char("oos")
    df = stock_char_df.swaplevel(i=0, j=1).sort_index().copy()
    stock_chars = [
        "MOM",
        "STR",
        "Lag Weekly Return",
        "TREND",
        "Beta",
        "Volatility",
        "52WH",
        "Bid-Ask",
        "Dollar Volume",
        "Zero Trade",
        "Price Delay",
        "Size",
        "Illiquidity",
    ]
    for ws in [5, 20, 60]:
        print(f"length before dropping nan: {len(df)}")
        df.dropna(inplace=True)
        print(f"length after dropping nan: {len(df)}")
        y = df[f"I{ws}/R{pw}"]
        x = df[stock_chars]
        mod_res = regr.fit(X=df[stock_chars], y=y)
        res_dict[f"{ws}D{pw}P"] = (mod_res, regr.is_r2())

    res_df = generate_latex_table_for_cs_regression_res(stock_chars, res_dict)
    latex = res_df.to_latex(
        na_rep="", column_format="l" + "c" * len(res_df.columns), escape=False
    )
    print(latex)
    return res_df


def load_cnn_and_monthly_stock_char(data_type):
    assert data_type in ["is", "oos"]
    save_path = CACHE_DIR / f"cnn_and_monthly_stock_char_{data_type}.parquet"
    print(f"loading from {save_path}")
    df = pd.read_parquet(save_path)
    return df


def cnn_and_ret_and_stock_char_regression(
    pw=5,
    ws_list=[5, 20, 60],
    regression_type="logit",
):
    stock_chars = [
        "MOM",
        "STR",
        "Lag Weekly Return",
        "TREND",
        "Beta",
        "Volatility",
        "52WH",
        "Bid-Ask",
        "Dollar Volume",
        "Zero Trade",
        "Price Delay",
        "Size",
        "Illiquidity",
    ]
    assert regression_type in ["logit", "ols"]
    regr = SMRegression(
        regression_type=regression_type,
        logit_threshold=0,
        rank_norm_x=True,
        ols_rank_norm_y=True,
    )
    factor_list = [f"I{ws}/R{pw}" for ws in [5, 20, 60]]
    col_index = []
    for i, factor in enumerate(factor_list):
        col_index += [
            (f"del{i}", f"del{i}"),
            (factor, "Uni"),
            (factor, "Multi w/o CNN"),
            (factor, "Multi"),
        ]
    col_index = pd.MultiIndex.from_tuples(col_index)
    oos_r2_name = "OOS " + regr.r2_name
    res_df = pd.DataFrame(
        columns=col_index, index=["CNN"] + stock_chars + [oos_r2_name]
    )
    for i, factor in enumerate(factor_list):
        res_df[(f"del{i}", f"del{i}")] = np.nan
        res_df[(factor, "Uni")] = np.nan
        res_df[(factor, "Multi w/o CNN")] = np.nan

    is_factor_df = load_cnn_and_monthly_stock_char("is")
    oos_factor_df = load_cnn_and_monthly_stock_char("oos")
    period = 12 if pw == 20 else 4 if pw == 60 else 52
    is_y = is_factor_df[f"Future_Ret_{pw}d"] * period
    oos_y = oos_factor_df[f"Future_Ret_{pw}d"] * period
    for ws in ws_list:
        regressors_dict = {
            "Multi": [f"I{ws}/R{pw}"] + stock_chars,
            "Multi w/o CNN": stock_chars,
            "Uni": [f"I{ws}/R{pw}"],
        }
        for x_name, x_cols in regressors_dict.items():
            oos_coeff = regr.fit(X=oos_factor_df[x_cols], y=oos_y)
            for char in x_cols:
                idx_name = "CNN" if char in factor_list else char
                res_df.loc[
                    idx_name, (f"I{ws}/R{pw}", x_name)
                ] = f"{oos_coeff['params'][char]:.2f}"

            mod_res = regr.fit(X=is_factor_df[x_cols], y=is_y)
            min_is_year = 3
            is_mean_min_sample = int(min_is_year * 12)
            regr.calculate_is_mean_by_stock(
                X=is_factor_df[x_cols],
                y=regr.transform_y(is_y),
                is_mean_min_sample=is_mean_min_sample,
            )
            oos_r2 = regr.oos_r2(oos_X=oos_factor_df[x_cols], oos_y=oos_y)
            res_df.loc[
                oos_r2_name,
                (f"I{ws}/R{pw}", x_name),
            ] = f"{oos_r2 * 100:.2f}"

    latex = res_df.to_latex(
        column_format="l" + len(col_index) * "c",
        escape=False,
        multicolumn_format="c",
        multirow=True,
        na_rep="",
    )
    for i in range(len(factor_list)):
        latex = latex.replace(f"del{i}", "")
    print(latex)

    return res_df


def load_image_scaled_market_data_with_cnn_pred(ws, pw, data_type):
    assert data_type in ["is", "oos"]
    file_path = CACHE_DIR / f"image_scaled_market_data_I{ws}R{pw}_{data_type}.parquet"
    ohlc_df = pd.read_parquet(file_path)
    print(f"Load from cache {file_path}")
    return ohlc_df


def regress_ret_on_cnn_pred_and_raw_image_data(
    ws,
    regression_type,
    pw_list=[5, 20, 60],
):
    assert regression_type in ["logit", "ols"]
    regr = SMRegression(regression_type)
    raw_image_data_names = [
        f"{p} lag {i + 1}"
        for p in ["open", "high", "low", "close", "ma", "vol"]
        for i in range(ws)
        if not (p == "close" and i == ws - 1)
    ]
    regressors_dict = {
        "Multi": ["CNN"] + raw_image_data_names,
        "Multi w/o CNN": raw_image_data_names,
        "Uni": ["CNN"],
    }

    model_list = [f"I{ws}/R{pw}" for pw in np.sort(pw_list)]
    col_index = []
    for i, model in enumerate(model_list):
        col_index += [
            (f"del{i}", f"del{i}"),
            (model, "Uni"),
            (model, "Multi w/o CNN"),
            (model, "Multi"),
        ]
    oos_r2_name = "OOS " + regr.r2_name
    res_df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(col_index),
        index=["CNN"] + raw_image_data_names + [oos_r2_name],
    )

    csv_indices = ["CNN"] + raw_image_data_names
    for col in ["CNN"] + raw_image_data_names:
        csv_indices += [col + " Tstat"]
    csv_res_df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(col_index), index=csv_indices + [oos_r2_name]
    )

    for pw in pw_list:
        is_ohlc = load_image_scaled_market_data_with_cnn_pred(ws, pw, data_type="is")
        oos_ohlc = load_image_scaled_market_data_with_cnn_pred(ws, pw, data_type="oos")
        is_y, oos_y = is_ohlc["period_ret"].copy(), oos_ohlc["period_ret"].copy()

        for x_name, x_cols in regressors_dict.items():
            oos_coeff = regr.fit(X=oos_ohlc[x_cols], y=oos_y)
            for char in x_cols:
                res_df.loc[
                    char, (f"I{ws}/R{pw}", x_name)
                ] = f"{oos_coeff['params'][char]:.2f}"

                csv_res_df.loc[char, (f"I{ws}/R{pw}", x_name)] = oos_coeff["params"][
                    char
                ]
                csv_res_df.loc[char + " Tstat", (f"I{ws}/R{pw}", x_name)] = oos_coeff[
                    "tstats"
                ][char]

            mod_res = regr.fit(X=is_ohlc[x_cols], y=is_y)
            regr.calculate_is_mean_by_stock(
                X=is_ohlc[x_cols],
                y=regr.transform_y(is_y),
                is_mean_min_sample=int(2 * (240 / pw)),
            )
            min_is_year = 3
            is_mean_min_sample = int(min_is_year * (240 / pw))
            regr.calculate_is_mean_by_stock(
                X=is_ohlc[x_cols],
                y=regr.transform_y(is_y),
                is_mean_min_sample=is_mean_min_sample,
            )
            oos_r2 = regr.oos_r2(oos_X=oos_ohlc[x_cols], oos_y=oos_y)
            res_df.loc[
                oos_r2_name,
                (f"I{ws}/R{pw}", x_name),
            ] = f"{oos_r2 * 100:.2f}"
            csv_res_df.loc[oos_r2_name, (f"I{ws}/R{pw}", x_name)] = oos_r2 * 100

    csv_res_df.to_csv(
        os.path.join(
            cf.LIGHT_CSV_RES_DIR,
            f"regress_ret_on_cnn_pred_and_raw_image_data_{regression_type}_{ws}d_is_mean_per_stock_v2.csv",
        )
    )
    latex = res_df.to_latex(
        column_format="l" + len(col_index) * "c",
        escape=False,
        multicolumn_format="c",
        multirow=True,
        na_rep="",
    )
    for i in range(len(model_list)):
        latex = latex.replace(f"del{i}", "")
    print(latex)

    return res_df


def regression_on_market_data_combined(ws, regression_type, pw_list=[5, 20, 60]):
    x_cols = [
        f"{p} lag {i + 1}"
        for p in ["open", "high", "low", "close", "ma", "vol"]
        for i in range(ws)
        if not (p == "close" and i == ws - 1)
    ]
    model_list = [f"I{ws}/R{pw}" for pw in np.sort(pw_list)]
    col_index = [(model, f"({i + 1})") for i, model in enumerate(model_list)]
    ret_reg_col_index = []
    for i, model in enumerate(model_list, start=1):
        ret_reg_col_index += [
            (f"del{i}", f"del{i}"),
            (model, f"({i * 3 + 1})"),
            (model, f"({i * 3 + 2})"),
            (model, f"({i * 3 + 3})"),
        ]

    col_index += ret_reg_col_index

    regr = SMRegression(
        regression_type, logit_threshold=0.5, rank_norm_x=True, ols_rank_norm_y=False
    )
    oos_r2_name = "OOS " + regr.r2_name
    res_df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(col_index),
        index=["CNN"] + x_cols + [oos_r2_name],
    )

    for i, pw in enumerate(pw_list, start=1):
        oos_ohlc = load_image_scaled_market_data_with_cnn_pred(ws, pw, data_type="oos")
        oos_y = oos_ohlc["CNN"].copy()
        mod_res = regr.fit(X=oos_ohlc[x_cols], y=oos_y)
        for char in x_cols:
            res_df.loc[
                char, (f"I{ws}/R{pw}", f"({i})")
            ] = f"{mod_res['params'][char]:.2f}"

        res_df.loc[
            oos_r2_name, (f"I{ws}/R{pw}", f"({i})")
        ] = f"{regr.is_r2() * 100:.2f}"

    print(res_df)
    res_df.to_csv(
        os.path.join(
            cf.LIGHT_CSV_RES_DIR,
            f"regression_on_market_data_combined_{regression_type}_{ws}d_is_mean_per_stock_v2.csv",
        )
    )
    right_csv_name = f"regress_ret_on_cnn_pred_and_raw_image_data_{regression_type}_{ws}d_is_mean_per_stock_v2.csv"
    try:
        ret_reg_df = pd.read_csv(
            os.path.join(cf.LIGHT_CSV_RES_DIR, right_csv_name), index_col=0
        )
    except FileNotFoundError:
        print("Need to run regress_ret_on_cnn_pred_and_raw_image_data() first")
        regress_ret_on_cnn_pred_and_raw_image_data(
            ws, regression_type, pw_list=[5, 20, 60]
        )
    finally:
        ret_reg_df = pd.read_csv(
            os.path.join(cf.LIGHT_CSV_RES_DIR, right_csv_name), index_col=0
        )
        ret_reg_df.columns = pd.MultiIndex.from_tuples(ret_reg_col_index)
        for idx in ["CNN"] + x_cols:
            for col in ret_reg_df.columns:
                if ret_reg_df.loc[idx, col] == np.nan:
                    continue
                ret_reg_df.loc[idx, col] = f"{float(ret_reg_df.loc[idx, col]):.2f}"

        res_df[ret_reg_col_index] = ret_reg_df[ret_reg_col_index]
        res_df.loc[oos_r2_name, ret_reg_col_index] = ret_reg_df.loc[
            oos_r2_name, ret_reg_col_index
        ].astype(float)

    pd.options.display.float_format = "{:,.2f}".format
    latex = res_df.to_latex(
        column_format="l" + len(res_df.columns) * "c",
        escape=False,
        multicolumn_format="c",
        multirow=True,
        na_rep="",
    )
    for i in range(4):
        latex = latex.replace((f"(del{i}, del{i})"), "")
        latex = latex.replace(f"del{i}", "")
        latex = latex.replace("nan", "")

    print(latex)

    return res_df


def main():
    pass


if __name__ == "__main__":
    main()
