import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from numpy.polynomial.polynomial import polyfit

from Data.dgp_config import FREQ_DICT, CACHE_DIR, PORTFOLIO, INTERNATIONAL_COUNTRIES
from Misc.config import OOS_YEARS
from Portfolio.portfolio import PortfolioManager
from Misc import utilities as ut


def portfolio_performance_helper(ws: int, pw: int):
    assert ws in [5, 20, 60] and pw in [5, 20, 60]
    freq = FREQ_DICT[pw]
    signal_df = pd.read_csv(CACHE_DIR / f"{freq}ly_prediction_with_rets.csv")
    signal_df["Date"] = pd.to_datetime(signal_df["Date"], dayfirst=True)
    signal_df["StockID"] = signal_df["StockID"].astype(str)
    signal_df = signal_df.set_index(["Date", "StockID"])
    df = signal_df.rename({f"CNN{ws}D{pw}P": "up_prob"}, axis="columns")
    df = df[["up_prob", "MarketCap"]].copy()
    portfolio_dir = PORTFOLIO / f"cnn_{freq}ly" / f"CNN{ws}D{pw}P"
    portfolio = PortfolioManager(df, freq=freq, portfolio_dir=portfolio_dir)
    portfolio.generate_portfolio()


def load_cnn_and_monthly_stock_char(data_type):
    assert data_type in ["is", "oos"]
    save_path = CACHE_DIR / f"cnn_and_monthly_stock_char_{data_type}.parquet"
    print(f"loading from {save_path}")
    df = pd.read_parquet(save_path)
    return df


def corr_between_cnn_pred_and_stock_chars():
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
    df = pd.DataFrame(columns=stock_chars)
    stock_char_df = load_cnn_and_monthly_stock_char("oos")
    for ws in [5, 20, 60]:
        for pw in [5, 20, 60]:
            for c in stock_chars:

                def up_prob_char_rank_corr(df):
                    prob_rank = df[f"I{ws}/R{pw}"].rank(
                        method="average", ascending=False
                    )
                    char_rank = df[c].rank(method="average", ascending=False)
                    return char_rank.corr(prob_rank, method="spearman")

                corr_series = stock_char_df.groupby("Date").apply(
                    up_prob_char_rank_corr
                )
                df.loc[f"I{ws}/R{pw}", c] = f"{corr_series.mean():.2f}"
    return df


def cnn_vs_linear_table_ols(pw):
    pf_dir = CACHE_DIR / "cnn1d_and_linear_model_portfolio_returns"
    col_index = [
        (f"I5/R{pw}", "ew"),
        (f"I5/R{pw}", "vw"),
        ("del0", "del0"),
        (f"I20/R{pw}", "ew"),
        (f"I20/R{pw}", "vw"),
        ("del1", "del1"),
        (f"I60/R{pw}", "ew"),
        (f"I60/R{pw}", "vw"),
    ]
    col_index = pd.MultiIndex.from_tuples(col_index)
    idx = [
        "Linear (image scale)",
        "Linear (cum. ret. scale)",
        "Linear (devol. ret. scale)",
        "CNN1D (image scale)",
        "CNN1D (cum. ret. scale)",
        "CNN1D (devol. ret. scale)",
    ]
    df = pd.DataFrame(columns=col_index, index=idx)
    for ws in [5, 20, 60]:
        for weight_type in ["ew", "vw"]:
            for model in ["Linear", "CNN1D"]:
                pf_df = pd.read_csv(
                    pf_dir
                    / f"{model.lower()}_I{ws}R{pw}_image_scale_{weight_type}.csv",
                    index_col=0,
                )
                df.loc[
                    f"{model} (image scale)", (f"I{ws}/R{pw}", weight_type)
                ] = calculate_portfolio_sr(pf_df, pw)

                pf_df = pd.read_csv(
                    pf_dir / f"{model.lower()}_I{ws}R{pw}_ret_scale_{weight_type}.csv",
                    index_col=0,
                )
                df.loc[
                    f"{model} (cum. ret. scale)", (f"I{ws}/R{pw}", weight_type)
                ] = calculate_portfolio_sr(pf_df, pw)

                pf_df = pd.read_csv(
                    pf_dir / f"{model.lower()}_I{ws}R{pw}_vol_scale_{weight_type}.csv",
                    index_col=0,
                )
                df.loc[
                    f"{model} (devol. ret. scale)", (f"I{ws}/R{pw}", weight_type)
                ] = calculate_portfolio_sr(pf_df, pw)
    return df


def load_international_portfolio_returns(horizon, country, weight_type, transfer=True):
    assert horizon in [5, 20]
    assert country in INTERNATIONAL_COUNTRIES + ["global"]
    assert weight_type in ["ew", "vw"]
    if transfer:
        portfolio_path = (
            CACHE_DIR
            / f"international_portfolio_decile_returns/{country}_I{horizon}R{horizon}_us_transfer_{weight_type}.csv"
        )
    else:
        portfolio_path = (
            CACHE_DIR
            / f"international_portfolio_decile_returns/{country}_I{horizon}R{horizon}_{weight_type}.csv"
        )
    pf = pd.read_csv(portfolio_path, index_col=0)
    return pf


def calculate_portfolio_sr(portfolio_ret, horizon):
    assert horizon in [5, 20, 60]
    avg = portfolio_ret["H-L"].mean()
    std = portfolio_ret["H-L"].std()
    horizons_per_year = 52 if horizon == 5 else 12 if horizon == 20 else 4
    return (avg * horizons_per_year) / (std * np.sqrt(horizons_per_year))


def glb_ctry_stock_number():
    stock_num_save_path = CACHE_DIR / f"glb_stocks_num.csv"
    df = pd.read_csv(stock_num_save_path, index_col=0)
    df["Average"] = df.mean(axis=1)
    df: pd.DataFrame = df.fillna(0).astype(int)
    df.sort_values(by="Average", ascending=False, inplace=True)
    return df


def international_sr_table(horizon):
    col_index = [("del2", "Stock Count")]
    for i, wt in enumerate(["ew", "vw"]):
        col_index += [
            (f"del{i}", f"del{i}"),
            (wt, "Re-train"),
            (wt, "Direct Transfer"),
            (wt, "Transfer-Re-train"),
        ]
    col_index = pd.MultiIndex.from_tuples(col_index)
    GLOBAL = "global"
    countries = INTERNATIONAL_COUNTRIES + [GLOBAL]
    df = pd.DataFrame(columns=col_index, index=countries)
    df[("del2", "Stock Count")] = (
        glb_ctry_stock_number()[[str(i) for i in range(1993, 2001)]]
        .mean(axis=1)
        .astype(int)
    )
    df.loc[GLOBAL, ("del2", "Stock Count")] = df[("del2", "Stock Count")].sum()
    df[("del2", "Stock Count")] = df[("del2", "Stock Count")].astype(int)
    for ctry in countries:
        for wt in ["ew", "vw"]:
            rt_df = load_international_portfolio_returns(
                horizon, ctry, wt, transfer=False
            )
            dt_df = load_international_portfolio_returns(
                horizon, ctry, wt, transfer=True
            )

            df.loc[ctry, (wt, "Re-train")] = calculate_portfolio_sr(
                rt_df, horizon=horizon
            )
            df.loc[ctry, (wt, "Direct Transfer")] = calculate_portfolio_sr(
                dt_df, horizon=horizon
            )

            df.loc[ctry, (wt, "Transfer-Re-train")] = float(
                df.loc[ctry, (wt, "Direct Transfer")]
            ) - float(df.loc[ctry, (wt, "Re-train")])
    df = df.sort_values(by=("del2", "Stock Count"), ascending=False)
    df.loc["Average"] = df.mean()
    df.loc["Average (excluding Global)"] = df.loc[INTERNATIONAL_COUNTRIES].mean()
    for ctry in countries:
        for wt in ["ew", "vw"]:
            df.loc[ctry, (wt, "Transfer-Re-train Value")] = float(
                df.loc[ctry, (wt, "Direct Transfer")]
            ) - float(df.loc[ctry, (wt, "Re-train")])
            df.loc[
                ctry, (wt, "Transfer-Re-train")
            ] = ut.star_significant_value_by_sample_num(
                float(df.loc[ctry, (wt, "Direct Transfer")])
                - float(df.loc[ctry, (wt, "Re-train")]),
                sample_num=len(OOS_YEARS),
                one_sided=True,
            )

    pd.options.display.float_format = "{:.2f}".format

    df[("del2", "Stock Count")] = df[("del2", "Stock Count")].astype(int)

    tl_columns = ["Re-train", "Direct Transfer", "Transfer-Re-train"]
    ndf = df[
        [("del2", "Stock Count"), ("del0", "del0")]
        + [("ew", c) for c in tl_columns]
        + [("del1", "del1")]
        + [("vw", c) for c in tl_columns]
    ]
    latex = ndf.to_latex(
        escape=False,
        column_format="l" + "c" * len(col_index),
        multicolumn_format="c",
        multirow=True,
        na_rep="",
    )
    for i in range(3):
        latex = latex.replace(f"del{i}", "")
    print(latex)
    return df


def glb_plot_sr_gain_vs_stocks_num(horizon):
    sr_df = international_sr_table(horizon)
    sr_df = sr_df[sr_df.index.isin(INTERNATIONAL_COUNTRIES)]
    for i, weight_type in enumerate(["ew", "vw"]):
        fig, ax = plt.subplots()
        stock_number, sr_gain = (
            sr_df[("del2", "Stock Count")],
            sr_df[(weight_type, "Transfer-Re-train Value")],
        )
        ax.scatter(stock_number, sr_gain)
        texts = []
        for ctry in INTERNATIONAL_COUNTRIES:
            texts.append(ax.text(stock_number[ctry], sr_gain[ctry], ctry))
        stock_number_np, sr_gain_np = stock_number.to_numpy(
            dtype="float"
        ), sr_gain.to_numpy(dtype="float")
        b, m = polyfit(stock_number_np, sr_gain_np, 1)
        plt.plot(stock_number_np, b + m * stock_number_np, "-")
        plt.xlabel("Stock Count", fontsize=16)
        plt.ylabel("Sharpe Ratio Gain", fontsize=16)
        plt.grid()
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="r", lw=0.5))
        plt.subplots_adjust(
            top=0.99, bottom=0.13, right=0.99, left=0.13, hspace=0, wspace=0
        )
        plt.savefig(
            f"./{horizon}d{horizon}p_sr_gain_Direct-Retrain_{weight_type}_ensem5.eps"
        )
        plt.show()
        plt.clf()


def time_scale_transfer_portfolio_helper(scale_size: int):
    assert scale_size in [20, 60]
    scaleDT_df = pd.read_csv(CACHE_DIR / f"scaleDT_{scale_size}_df.csv")
    scaleDT_df["Date"] = pd.to_datetime(scaleDT_df["Date"], dayfirst=True)
    scaleDT_df["StockID"] = scaleDT_df["StockID"].astype(str)
    scaleDT_df = scaleDT_df.set_index(["Date", "StockID"])
    portfolio_dir = PORTFOLIO / "cnn_timescale" / f"CNN{scale_size}D{scale_size}P"
    portfolio = PortfolioManager(
        scaleDT_df, freq=FREQ_DICT[scale_size], portfolio_dir=portfolio_dir
    )
    portfolio.generate_portfolio()


def main():
    pass


if __name__ == "__main__":
    main()
