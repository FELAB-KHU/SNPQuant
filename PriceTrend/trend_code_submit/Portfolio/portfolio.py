import os
import os.path as op
import pdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Misc import utilities as ut
from Data import equity_data as eqd


class PortfolioManager(object):
    def __init__(
        self,
        signal_df: pd.DataFrame,
        freq: str,
        portfolio_dir: str,
        start_year=2022,
        end_year=2023,
        country="USA",
        delay_list=None,
        load_signal=True,
        custom_ret=None,
        transaction_cost=False,
    ):
        assert freq in ["week", "month", "quarter"]
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.start_year = start_year
        self.end_year = end_year
        self.country = country
        self.no_delay_ret_name = f"next_{freq}_ret"
        self.custom_ret = custom_ret
        self.delay_list = [0] if delay_list is None else delay_list
        self.transaction_cost = transaction_cost
        if load_signal:
            assert "up_prob" in signal_df.columns
            self.signal_df = self.get_up_prob_with_period_ret(signal_df)

    def __add_period_ret_to_us_res_df_w_delays(self, signal_df):
        period_ret = eqd.get_period_ret(self.freq, country=self.country)
        columns = ["MarketCap"] + [
            f"next_{self.freq}_ret_{dl}delay" for dl in self.delay_list
        ]
        if self.custom_ret is not None:
            columns += [self.custom_ret]
        signal_df = signal_df.copy()
        signal_df[columns] = period_ret[columns]
        signal_df[self.no_delay_ret_name] = signal_df[f"next_{self.freq}_ret_0delay"]
        signal_df.dropna(inplace=True)
        for dl in self.delay_list:
            dl_ret_name = f"next_{self.freq}_ret_{dl}delay"
            print(
                f"{len(signal_df)} samples, {dl} delay \
                    nan values {np.sum(signal_df[dl_ret_name].isna())},\
                    zero values {np.sum(signal_df[dl_ret_name] == 0.)}"
            )
        return signal_df.copy()

    def get_up_prob_with_period_ret(self, signal_df):
        signal_df = signal_df[
            signal_df.index.get_level_values("Date").year.isin(
                range(self.start_year, self.end_year + 1)
            )
        ]
        signal_df = self.__add_period_ret_to_us_res_df_w_delays(signal_df)
        if self.country not in ["future", "new_future"]:
            signal_df.MarketCap = signal_df.MarketCap.abs()
            signal_df = signal_df[~signal_df.MarketCap.isnull()].copy()
        return signal_df

    def calculate_portfolio_rets(self, weight_type, cut=10, delay=0):
        assert weight_type in ["ew", "vw"]
        assert delay in self.delay_list
        if self.custom_ret:
            print(f"Calculating portfolio using {self.custom_ret}")
            ret_name = self.custom_ret
        else:
            ret_name = (
                self.no_delay_ret_name
                if delay == 0
                else f"next_{self.freq}_ret_{delay}delay"
            )
        df = self.signal_df.copy()

        def __get_decile_df_with_inv_ret(reb_df, decile_idx):
            rebalance_up_prob = reb_df.up_prob
            low = np.percentile(rebalance_up_prob, decile_idx * 100.0 / cut)
            high = np.percentile(rebalance_up_prob, (decile_idx + 1) * 100.0 / cut)
            if decile_idx == 0:
                pf_filter = (rebalance_up_prob >= low) & (rebalance_up_prob <= high)
            else:
                pf_filter = (rebalance_up_prob > low) & (rebalance_up_prob <= high)
            _decile_df = reb_df[pf_filter].copy()
            if weight_type == "ew":
                stock_num = len(_decile_df)
                _decile_df["weight"] = (
                    pd.Series(
                        np.ones(stock_num), dtype=np.float64, index=_decile_df.index
                    )
                    / stock_num
                )
            else:
                value = _decile_df.MarketCap
                _decile_df["weight"] = pd.Series(value, dtype=np.float64) / np.sum(value)
            _decile_df["inv_ret"] = _decile_df["weight"] * _decile_df[ret_name]
            return _decile_df

        dates = np.sort(np.unique(df.index.get_level_values("Date")))
        print(f"Calculating portfolio from {dates[0]}, {dates[1]} to {dates[-1]}")
        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        prev_to_df = None
        prob_ret_corr = []
        prob_ret_pearson_corr = []
        prob_inv_ret_corr = []
        prob_inv_ret_pearson_corr = []

        for i, date in enumerate(dates):
            rebalance_df = df.loc[date].copy()
            rank_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="spearman"
            )
            pearson_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="pearson"
            )
            prob_ret_corr.append(rank_corr)
            prob_ret_pearson_corr.append(pearson_corr)

            low = np.percentile(rebalance_df.up_prob, 10)
            high = np.percentile(rebalance_df.up_prob, 90)
            if low == high:
                print(f"Skipping {date}")
                continue
            for j in range(cut):
                decile_df = __get_decile_df_with_inv_ret(rebalance_df, j)
                if self.transaction_cost:
                    if j == cut - 1:
                        decile_df["inv_ret"] -= (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                    elif j == 0:
                        decile_df["inv_ret"] += (
                            decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                portfolio_ret.loc[date, j] = np.sum(decile_df["inv_ret"])

            sell_decile = __get_decile_df_with_inv_ret(rebalance_df, 0)
            buy_decile = __get_decile_df_with_inv_ret(rebalance_df, cut - 1)
            buy_sell_decile = pd.concat([sell_decile, buy_decile])
            prob_inv_ret_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="spearman")
            )
            prob_inv_ret_pearson_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="pearson")
            )

            sell_decile[["weight", "inv_ret"]] = sell_decile[["weight", "inv_ret"]] * (
                -1
            )
            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                tto_df = pd.DataFrame(
                    index=np.unique(list(to_df.index) + list(prev_to_df.index))
                )
                try:
                    tto_df["cur_weight"] = to_df["weight"]
                except ValueError:
                    pdb.set_trace()
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[
                    ["weight", ret_name, "inv_ret"]
                ]
                tto_df.fillna(0, inplace=True)
                denom = 1 + np.sum(tto_df["inv_ret"])
                turnover[i - 1] = np.sum(
                    (
                        tto_df["cur_weight"]
                        - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom
                    ).abs()
                )
                turnover[i - 1] = turnover[i - 1] * 0.5
            prev_to_df = to_df

        portfolio_ret = portfolio_ret.fillna(0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]
        print(
            f"Spearman Corr between Prob and Stock Return is {np.nanmean(prob_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Stock Return is {np.nanmean(prob_ret_pearson_corr):.4f}"
        )
        print(
            f"Spearman Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_pearson_corr):.4f}"
        )
        
        return portfolio_ret, np.mean(turnover)

    @staticmethod
    def _ret_to_cum_log_ret(rets):
        log_rets = np.log(rets.astype(float) + 1)
        return log_rets.cumsum()

    def make_portfolio_plot(
        self, portfolio_ret, cut=None, weight_type=None, save_path=None, plot_title=None
    ):
        for weight_type in ["ew", "vw"]:
            pf_name = self.get_portfolio_name(weight_type, delay=0, cut=10)
            print(f"Calculating {pf_name}")
            portfolio_ret, turnover = self.calculate_portfolio_rets(weight_type, cut=10, delay=0)
        
            save_path = f"./{weight_type}.png"
            plot_title = f'{self.freq} "plot_title" {weight_type} {cut}'
            ret_name = "nxt_freq_ewret" if weight_type == "ew" else "nxt_freq_vwretx"
            df = portfolio_ret.copy()
            df.columns = ["Low(L)"] + [str(i) for i in range(2, cut)] + ["High(H)", "H-L"]
            bench = eqd.get_bench_freq_rets(self.freq)
            spy = eqd.get_spy_freq_rets(self.freq)
            df["SPY"] = spy[ret_name]
            df["Benchmark"] = bench[ret_name]
            df.dropna(inplace=True)
            top_col_name, bottom_col_name = ("High(H)", "Low(L)")
            log_ret_df = pd.DataFrame(index=df.index)
            for column in df.columns:
                log_ret_df[column] = self._ret_to_cum_log_ret(df[column])
            plt.figure()
            log_ret_df = log_ret_df[[top_col_name, bottom_col_name, "H-L", "SPY", "Benchmark"]]
            prev_year = pd.to_datetime(log_ret_df.index[0]).year - 1
            prev_day = pd.to_datetime("{}-12-31".format(prev_year))
            log_ret_df.loc[prev_day] = [0, 0, 0, 0, 0]
            plot = log_ret_df.plot(
                style={"SPY": "y", "Benchmark": "g", top_col_name: "b", bottom_col_name: "r", "H-L": "k"},
                lw=1,
                title=plot_title,
            )
            plot.legend(loc=2)
            plt.grid()
            plt.savefig(save_path)
            plt.close()

    def portfolio_res_summary(self, portfolio_ret, turnover, cut=10):
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        res = np.zeros((cut + 1, 3))
        period = 52 if self.freq == "week" else 12 if self.freq == "month" else 4
        res[:, 0] = avg * period
        res[:, 1] = std * np.sqrt(period)
        res[:, 2] = res[:, 0] / res[:, 1]

        summary_df = pd.DataFrame(res, columns=["ret", "std", "SR"])
        summary_df = summary_df.set_index(
            pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L"])
        )
        summary_df.loc["Turnover", "SR"] = turnover / (
            1 / 4
            if self.freq == "week"
            else 1
            if self.freq == "month"
            else 3
            if self.freq == "quarter"
            else 12
        )
        print(summary_df)
        return summary_df

    def generate_portfolio(self, cut=10, delay=0):
        assert delay in self.delay_list
        for weight_type in ["ew", "vw"]:
            pf_name = self.get_portfolio_name(weight_type, delay, cut)
            print(f"Calculating {pf_name}")
            portfolio_ret, turnover = self.calculate_portfolio_rets(
                weight_type, cut=cut, delay=delay
            )
            data_dir = ut.get_dir(op.join(self.portfolio_dir, "pf_data"))
            portfolio_ret.to_csv(op.join(data_dir, f"pf_data_{pf_name}.csv"))

            summary_df = self.portfolio_res_summary(portfolio_ret, turnover, cut)
            smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
            print(f"Summary saved to {smry_path}")
            summary_df.to_csv(smry_path)
            with open(os.path.join(self.portfolio_dir, f"{pf_name}.txt"), "w+") as file:
                summary_df = summary_df.astype(float).round(2)
                file.write(ut.to_latex_w_turnover(summary_df, cut=cut))

    def get_portfolio_name(self, weight_type, delay, cut):
        assert weight_type.lower() in ["ew", "vw"]
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_surfix = "" if cut == 10 else f"_{cut}cut"
        custom_ret_surfix = "" if self.custom_ret is None else self.custom_ret
        tc_surfix = "" if not self.transaction_cost else "_w_transaction_cost"
        pf_name = f"{delay_prefix}{weight_type.lower()}{cut_surfix}{custom_ret_surfix}{tc_surfix}"
        return pf_name

    def load_portfolio_ret(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        try:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}.csv"),
                index_col=0,
            )
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}_100.csv"),
                index_col=0,
            )
        df.index = pd.to_datetime(df.index)
        return df

    def load_portfolio_summary(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, delay, cut)
        try:
            df = pd.read_csv(op.join(self.portfolio_dir, f"{pf_name}.csv"), index_col=0)
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, f"{pf_name}_100.csv"), index_col=0
            )
        return df


def main():
    pass


if __name__ == "__main__":
    main()
