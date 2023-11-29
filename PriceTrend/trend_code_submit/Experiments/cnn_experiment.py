import copy

from tqdm import tqdm
import itertools
import math
import sys
import time
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset, random_split

from Model import cnn_model
from Portfolio import portfolio as pf
from Misc import config as cf
from Data import dgp_config as dcf
from Misc import utilities as ut
from Data.chart_dataset import EquityDataset, TS1DDataset
from Data import equity_data as eqd


class Experiment(object):
    def __init__(
        self,
        ws,
        pw,
        model_obj: cnn_model.Model,
        train_freq,
        ensem=5,
        lr=1e-5,
        drop_prob=0.50,
        device_number=0,
        max_epoch=50,
        enable_tqdm=True,
        early_stop=True,
        has_ma=True,
        has_volume_bar=True,
        is_years=cf.IS_YEARS,
        oos_years=cf.OOS_YEARS,
        country="USA",
        transfer_learning=None,
        annual_stocks_num="all",
        tstat_threshold=0,
        ohlc_len=None,
        pf_freq=None,
        tensorboard=False,
        weight_decay=0,
        loss_name="cross_entropy",
        margin=1,
        train_size_ratio=0.7,
        ts_scale="image_scale",
        chart_type="bar",
        delayed_ret=0,
    ):
        self.ws = ws
        self.pw = pw
        self.model_obj = model_obj
        self.train_freq = train_freq
        assert self.train_freq in ["week", "month", "quarter", "year"]
        self.ensem = ensem
        self.lr = lr
        self.drop_prob = drop_prob
        self.device_number = device_number if device_number is not None else 0
        self.device = torch.device(
            "cuda:{}".format(self.device_number) if torch.cuda.is_available() else "cpu"
        )
        self.max_epoch = max_epoch
        self.enable_tqdm = enable_tqdm
        self.early_stop = early_stop
        self.has_ma = has_ma
        self.has_volume_bar = has_volume_bar
        self.is_years = is_years
        self.oos_years = oos_years
        self.country = country
        if self.country == "China":
            self.oos_years = list(range(2001, 2019))
        elif (
            self.country in cf.START_YEAR_DICT.keys()
        ):
            self.oos_years = list(range(cf.START_YEAR_DICT[self.country] + 8, 2024))
        self.oos_start_year = self.oos_years[0]
        assert transfer_learning in [None, "ft", "usa", "scaleDT"]
        self.tl = transfer_learning
        self.annual_stocks_num = annual_stocks_num
        self.tstat_threshold = tstat_threshold
        self.ohlc_len = self.ws if ohlc_len is None else ohlc_len
        assert self.ohlc_len in [5, 20, 60]
        self.mean_std = None
        self.pf_freq = dcf.FREQ_DICT[self.pw] if pf_freq is None else pf_freq
        assert self.pf_freq in ["week", "month", "quarter", "year"]
        self.tensorboard = tensorboard
        self.weight_decay = weight_decay
        self.margin = margin
        self.loss_name = loss_name if model_obj.regression_label is None else "MSE"
        self.train_size_ratio = train_size_ratio
        self.ts_scale = ts_scale
        assert self.ts_scale in ["image_scale", "ret_scale", "vol_scale"]
        self.label_dtype = (
            torch.long if self.model_obj.regression_label is None else torch.float
        )
        self.chart_type = chart_type
        assert self.chart_type in ["bar", "pixel", "centered_pixel"]
        self.delayed_ret = delayed_ret
        assert self.delayed_ret in [0, 1, 2, 3, 4, 5]

        model_name = (
            self.model_obj.name
            if self.model_obj is not None
            else cf.BENCHMARK_MODEL_NAME_DICT[ws]
        )
        self.exp_name = self.get_exp_name()
        self.pf_dir = self.get_portfolio_dir()
        self.model_dir = ut.get_dir(os.path.join(cf.EXP_DIR, model_name, self.exp_name))
        self.ensem_res_dir = ut.get_dir(os.path.join(self.model_dir, "ensem_res"))
        self.tb_dir = ut.get_dir(os.path.join(self.model_dir, "tensorboard_res"))
        self.oos_metrics_path = os.path.join(
            self.ensem_res_dir, "oos_metrics_no_delay.pkl"
        )

    def get_model_checkpoint_path(self, model_num):
        model_dir = self.model_dir
        if self.country != "USA" and self.tl == "usa":
            print(f"Using pretrained USA model for {self.country}-usa")
            model_dir = model_dir.replace(f"-{self.country}-usa", "")

        if self.ws != self.ohlc_len and self.tl == "scaleDT":
            ohlc_freq = self.ws / self.ohlc_len
            print(
                f"Applying pretrained {int(self.ws / ohlc_freq)}d{int(self.pw / ohlc_freq)}p model to evaluate {self.ws}d{self.pw}p"
            )
            model_dir = model_dir.replace(
                f"{self.ws}d{self.pw}p",
                f"{int(self.ws / ohlc_freq)}d{int(self.pw / ohlc_freq)}p",
            )
            model_dir = model_dir.replace(f"-{self.ohlc_len}ohlc", "")
            model_dir = model_dir.replace(f"-scaleDT", "").replace(
                f"{self.train_freq}lyTrained",
                f"{dcf.FREQ_DICT[self.pw / ohlc_freq]}lyTrained",
            )
            if not os.path.exists(model_dir):
                raise ValueError(f"{model_dir} not trained yet")
        return os.path.join(model_dir, f"checkpoint{model_num}.pth.tar")

    def load_model_state_dict_from_save_path(self, model_save_path):
        print(f"Loading model state dict from {model_save_path}")
        model_state_dict = torch.load(model_save_path, map_location=self.device)[
            "model_state_dict"
        ]
        return model_state_dict

    def get_exp_name(self):
        exp_setting_list = [
            f"{self.ws}d{self.pw}p-lr{self.lr:.0E}-dp{self.drop_prob:.2f}",
            f"ma{self.has_ma}-vb{self.has_volume_bar}-{self.train_freq}lyTrained",
        ]
        if self.delayed_ret == 0:
            exp_setting_list.append("noDelayedReturn")
        else:
            exp_setting_list.append(f"{self.delayed_ret}DelayedReturn")
        if not self.model_obj.batch_norm:
            exp_setting_list.append("noBN")
        if not self.model_obj.xavier:
            exp_setting_list.append("noXavier")
        if not self.model_obj.lrelu:
            exp_setting_list.append("noLRelu")
        if self.weight_decay != 0:
            exp_setting_list.append(f"WD{self.weight_decay:.0E}")
        if self.loss_name != "cross_entropy":
            exp_setting_list.append(self.loss_name)
            if self.loss_name == "multimarginloss":
                exp_setting_list.append(f"margin{self.margin:.0E}")

        if self.annual_stocks_num != "all":
            exp_setting_list.append(f"top{self.annual_stocks_num}AnnualStock")
        if self.tstat_threshold != 0:
            exp_setting_list.append(f"{self.tstat_threshold}tstat")
        if self.ohlc_len != self.ws:
            exp_setting_list.append(f"{self.ohlc_len}ohlc")
        if self.train_size_ratio != 0.7:
            exp_setting_list.append(f"tv_ratio{self.train_size_ratio:.1f}")
        if self.model_obj.regression_label is not None:
            exp_setting_list.append(self.model_obj.regression_label)
        if self.ts_scale != "image_scale":
            ts_name = "raw_ts1d" if self.ts_scale == "ret_scale" else "vol_scale"
            exp_setting_list.append(ts_name)
        if self.chart_type != "bar":
            exp_setting_list.append(self.chart_type)
        if self.country != "USA":
            exp_setting_list.append(str(self.country))
        if self.tl is not None:
            exp_setting_list.append(str(self.tl))
        exp_name = "-".join(exp_setting_list)
        return exp_name

    def get_portfolio_dir(self):
        name_list = [self.country]
        if self.model_obj.name not in cf.BENCHMARK_MODEL_NAME_DICT.values():
            name_list.append(self.model_obj.name)
        name_list.append(self.exp_name)
        name_list.append(f"ensem{self.ensem}")
        if (self.oos_years[0] != 2001) or (self.oos_years[-1] != 2019):
            name_list.append("{}-{}".format(self.oos_years[0], self.oos_years[-1]))
        if self.pf_freq != dcf.FREQ_DICT[self.pw]:
            name_list.append(f"{self.pf_freq}ly")
        if self.delayed_ret == 0:
            name_list.append("noDelayedReturn")
        else:
            name_list.append(f"{self.delayed_ret}DelayedReturn")
        name = "_".join(name_list)
        pf_dir = cf.get_dir(os.path.join(cf.PORTFOLIO_DIR, name))
        return pf_dir

    def get_train_validate_dataloaders_dict(self):
        if self.model_obj.ts1d_model:
            tv_datasets = {
                year: TS1DDataset(
                    self.ws,
                    self.pw,
                    self.train_freq,
                    year,
                    country=self.country,
                    remove_tail=(year == self.oos_start_year - 1),
                    ohlc_len=self.ohlc_len,
                    ts_scale=self.ts_scale,
                    regression_label=self.model_obj.regression_label,
                )
                for year in self.is_years
            }
        else:
            tv_datasets = {
                year: EquityDataset(
                    self.ws,
                    self.pw,
                    self.train_freq,
                    year,
                    country=self.country,
                    has_volume_bar=self.has_volume_bar,
                    has_ma=self.has_ma,
                    annual_stocks_num=self.annual_stocks_num,
                    tstat_threshold=self.tstat_threshold,
                    stockid_filter=None,
                    remove_tail=(year == self.oos_start_year - 1),
                    ohlc_len=self.ohlc_len,
                    regression_label=self.model_obj.regression_label,
                    chart_type=self.chart_type,
                    delayed_ret=self.delayed_ret,
                )
                for year in self.is_years
            }
        tv_dataset = ConcatDataset([tv_datasets[year] for year in self.is_years])
        train_size = int(len(tv_dataset) * self.train_size_ratio)
        validate_size = len(tv_dataset) - train_size
        print(
            f"Training and validation data from {self.is_years[0]} to {self.is_years[-1]} \
                with training set size {train_size} and validation set size {validate_size}"
        )
        train_dataset, validate_dataset = random_split(
            tv_dataset,
            [train_size, validate_size],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cf.BATCH_SIZE,
            shuffle=True,
            num_workers=cf.NUM_WORKERS,
        )
        validate_dataloader = DataLoader(
            validate_dataset, batch_size=cf.BATCH_SIZE, num_workers=cf.NUM_WORKERS
        )
        dataloaders_dict = {"train": train_dataloader, "validate": validate_dataloader}
        return dataloaders_dict

    def calculate_true_up_prob_from_dataloader(self, dataloader):
        df = self.get_df_from_dataloader(dataloader)
        assert np.sum(df.label == 1) + np.sum(df.label == 0) == len(df)
        return len(df), np.sum(df.label == 1) / len(df)

    def get_df_from_dataloader(self, dataloader):
        df_columns = ["StockID", "ending_date", "label", "ret_val", "MarketCap"]
        df_dtypes = [object, "datetime64[ns]", np.int, np.float64, np.float64]
        df_list = []
        for batch in dataloader:
            batch_df = ut.df_empty(df_columns, df_dtypes)
            batch_df["StockID"] = batch["StockID"]
            batch_df["ending_date"] = pd.to_datetime(
                [str(t) for t in batch["ending_date"]]
            )
            batch_df["label"] = np.nan_to_num(batch["label"].numpy()).reshape(-1)
            batch_df["ret_val"] = np.nan_to_num(batch["ret_val"].numpy()).reshape(-1)
            batch_df["MarketCap"] = np.nan_to_num(batch["MarketCap"].numpy()).reshape(
                -1
            )
            df_list.append(batch_df)
        df = pd.concat(df_list)
        df.reset_index(drop=True)
        return df

    def train_empirical_ensem_model(self, ensem_range=None, pretrained=True):
        val_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])
        train_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])
        if ensem_range is None:
            ensem_range = range(self.ensem)
        for model_num in ensem_range:
            print(f"Start Training Ensem Number {model_num}")
            model_save_path = self.get_model_checkpoint_path(model_num)
            if os.path.exists(model_save_path) and pretrained:
                print("Found pretrained model {}".format(model_save_path))
                validate_metrics = torch.load(model_save_path)
            else:
                dataloaders_dict = self.get_train_validate_dataloaders_dict()
                train_metrics, validate_metrics, _ = self.train_single_model(
                    dataloaders_dict, model_save_path, model_num=model_num
                )
                for column in train_metrics.keys():
                    train_df.loc[model_num, column] = train_metrics[column]

            for column in validate_metrics.keys():
                if column == "model_state_dict":
                    continue
                val_df.loc[model_num, column] = validate_metrics[column]

        val_df = val_df.astype(np.float64).round(3)
        val_df.loc["Mean"] = val_df.mean()
        val_df.to_csv(
            os.path.join(
                cf.LOG_DIR,
                f"{self.model_obj.name}-{self.exp_name}-ensem{self.ensem}.csv",
            ),
            index=True,
        )

        with open(
            os.path.join(
                cf.LATEX_DIR,
                f"{self.model_obj.name}-{self.exp_name}-ensem{self.ensem}.txt",
            ),
            "w+",
        ) as file:
            file.write(val_df.to_latex())

    def load_mean_validation_metrics(self):
        df = pd.read_csv(
            os.path.join(
                cf.LOG_DIR,
                f"{self.model_obj.name}-{self.exp_name}-ensem{self.ensem}.csv",
            ),
            index_col=0,
        )
        return df.loc["Mean"]

    def load_mean_train_metrics(self):
        try:
            df = pd.read_csv(
                os.path.join(
                    cf.LOG_DIR,
                    f"{self.model_obj.name}-{self.exp_name}-ensem{self.ensem}_train.csv",
                ),
                index_col=0,
            )
            return df.loc["Mean"]
        except FileNotFoundError:
            return None

    @staticmethod
    def _update_running_metrics(loss, labels, preds, running_metrics):
        running_metrics["running_loss"] += loss.item() * len(labels)
        running_metrics["running_correct"] += (preds == labels).sum().item()
        running_metrics["TP"] += (preds * labels).sum().item()
        running_metrics["TN"] += ((preds - 1) * (labels - 1)).sum().item()
        running_metrics["FP"] += (preds * (labels - 1)).sum().abs().item()
        running_metrics["FN"] += ((preds - 1) * labels).sum().abs().item()

    @staticmethod
    def _generate_epoch_stat(epoch, learning_rate, num_samples, running_metrics):
        TP, TN, FP, FN = (
            running_metrics["TP"],
            running_metrics["TN"],
            running_metrics["FP"],
            running_metrics["FN"],
        )
        epoch_stat = {"epoch": epoch, "lr": "{:.2E}".format(learning_rate)}
        epoch_stat["diff"] = 1.0 * ((TP + FP) - (TN + FN)) / num_samples
        epoch_stat["loss"] = running_metrics["running_loss"] / num_samples
        epoch_stat["accy"] = 1.0 * running_metrics["running_correct"] / num_samples
        epoch_stat["MCC"] = (
            np.nan
            if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0
            else 1.0
            * (TP * TN - FP * FN)
            / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        )
        return epoch_stat

    def evaluate(self, model, dataloaders_dict, new_label=None):
        assert new_label in [None, 0, 1]
        print("Evaluating model on device: {}".format(self.device))
        model.to(self.device)
        res_dict = {}
        for subset in dataloaders_dict.keys():
            since = time.time()
            if "tqdm" in sys.modules and self.enable_tqdm:
                data_iterator = tqdm(dataloaders_dict[subset], leave=True, unit="batch")
                data_iterator.set_description("Evaluation: ")
            else:
                data_iterator = dataloaders_dict[subset]
            model.eval()
            running_metrics = {
                "running_loss": 0.0,
                "running_correct": 0.0,
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
            }
            for batch in data_iterator:
                inputs = batch["image"].to(self.device, dtype=torch.float)
                if new_label is not None:
                    labels = (
                        torch.Tensor([new_label])
                        .repeat(inputs.shape[0])
                        .to(self.device, dtype=self.label_dtype)
                    )
                else:
                    labels = batch["label"].to(self.device, dtype=self.label_dtype)
                outputs = model(inputs)
                loss = self.loss_from_model_output(labels, outputs)
                _, preds = torch.max(outputs, 1)
                self._update_running_metrics(loss, labels, preds, running_metrics)
                del inputs, labels
            num_samples = len(dataloaders_dict[subset].dataset)
            epoch_stat = self._generate_epoch_stat(-1, -1, num_samples, running_metrics)
            if "tqdm" in sys.modules and self.enable_tqdm:
                data_iterator.set_postfix(epoch_stat)
                data_iterator.update()
            print(epoch_stat)
            time_elapsed = time.time() - since
            print(
                "Evaluation on {} complete in {:.0f}m {:.0f}s".format(
                    subset, time_elapsed // 60, time_elapsed % 60
                )
            )
            res_dict[subset] = {
                metric: epoch_stat[metric] for metric in ["loss", "accy", "MCC", "diff"]
            }
        del model
        torch.cuda.empty_cache()
        return res_dict

    def loss_from_model_output(self, labels, outputs):
        if self.loss_name == "kldivloss":
            log_prob = nn.LogSoftmax(dim=1)(outputs)
            target = ut.binary_one_hot(labels.view(-1, 1), self.device)
            target = target.to(torch.float)
            loss = torch.nn.KLDivLoss()(log_prob, target)
        elif self.loss_name == "multimarginloss":
            loss = torch.nn.MultiMarginLoss(margin=self.margin)(outputs, labels)
        elif self.loss_name == "cross_entropy":
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        elif self.loss_name == "MSE":
            loss = torch.nn.MSELoss()(outputs.flatten(), labels)
        else:
            loss = None
        return loss

    def train_single_model(self, dataloaders_dict, model_save_path, model_num=None):
        if self.country != "USA" and self.tl is not None:
            us_model_save_path = model_save_path.replace(
                f"-{self.country}-{self.tl}", ""
            )
            model_state_dict = self.load_model_state_dict_from_save_path(
                us_model_save_path
            )
            model = self.model_obj.init_model_with_model_state_dict(
                model_state_dict=model_state_dict, device=self.device
            )
            if self.tl == "usa":
                validate_metrics = self.evaluate(
                    model, {"validate": dataloaders_dict["validate"]}
                )["validate"]
                validate_metrics["epoch"] = 0
                self.release_dataloader_memory(dataloaders_dict, model)
                return None, validate_metrics, None
            elif self.tl == "ft":
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, 2)
                model.fc.apply(cnn_model.init_weights)
                optimizer = optim.Adam(
                    model.fc.parameters(), lr=self.lr, weight_decay=self.weight_decay
                )
                model.to(self.device)
            else:
                raise ValueError(f"{self.tl} on {self.country} not supported")
        else:
            model = self.model_obj.init_model(device=self.device, state_dict=None)
            optimizer = optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        print("Training on device {} under {}".format(self.device, model_save_path))
        print(model)
        cudnn.benchmark = True
        since = time.time()
        best_validate_metrics = {"loss": 10.0, "accy": 0.0, "MCC": 0.0, "epoch": 0}
        best_model = copy.deepcopy(model.state_dict())
        train_metrics = {"prev_loss": 10.0, "pattern_accy": -1}
        prev_weight_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name[-8:] == "0.weight":
                prev_weight_dict[name] = param.data.clone()
        for epoch in range(self.max_epoch):
            for phase in ["train", "validate"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                if "tqdm" in sys.modules and self.enable_tqdm:
                    data_iterator = tqdm(
                        dataloaders_dict[phase],
                        leave=True,
                        unit="batch",
                        postfix={
                            "epoch": -1,
                            "loss": 10.0,
                            "accy": 0.0,
                            "MCC": 0.0,
                            "diff": 0,
                        },
                    )
                    data_iterator.set_description("Epoch {}: {}".format(epoch, phase))
                else:
                    data_iterator = dataloaders_dict[phase]

                running_metrics = {
                    "running_loss": 0.0,
                    "running_correct": 0.0,
                    "TP": 0,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0,
                }
                for i, batch in enumerate(data_iterator):
                    inputs = batch["image"].to(self.device, dtype=torch.float)
                    labels = batch["label"].to(self.device, dtype=self.label_dtype)
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = self.loss_from_model_output(labels, outputs)
                        _, preds = torch.max(outputs, 1)
                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    self._update_running_metrics(loss, labels, preds, running_metrics)
                    del inputs, labels
                num_samples = len(dataloaders_dict[phase].dataset)
                epoch_stat = self._generate_epoch_stat(
                    epoch, self.lr, num_samples, running_metrics
                )
                if "tqdm" in sys.modules and self.enable_tqdm:
                    data_iterator.set_postfix(epoch_stat)
                    data_iterator.update()
                print(epoch_stat)
                if phase == "validate":
                    if epoch_stat["loss"] < best_validate_metrics["loss"]:
                        for metric in ["loss", "accy", "MCC", "epoch", "diff"]:
                            best_validate_metrics[metric] = epoch_stat[metric]
                        best_model = copy.deepcopy(model.state_dict())

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name[-8:] == "0.weight":
                        pct_chg = (
                            param.data - prev_weight_dict[name]
                        ).norm() / prev_weight_dict[name].norm()
                        print("{} pct chg: {:.4f}".format(name, pct_chg))
                        prev_weight_dict[name] = param.data.clone()

            if self.early_stop and (epoch - best_validate_metrics["epoch"]) >= 2:
                break
            print()
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print(
            "Best val loss: {:4f} at epoch {}, ".format(
                best_validate_metrics["loss"], best_validate_metrics["epoch"]
            )
        )
        model.load_state_dict(best_model)
        best_validate_metrics["model_state_dict"] = model.state_dict().copy()
        torch.save(best_validate_metrics, model_save_path)
        train_metrics = self.evaluate(model, {"train": dataloaders_dict["train"]})[
            "train"
        ]
        train_metrics["epoch"] = best_validate_metrics["epoch"]
        self.release_dataloader_memory(dataloaders_dict, model)
        del best_validate_metrics["model_state_dict"]
        return train_metrics, best_validate_metrics, model

    @staticmethod
    def release_dataloader_memory(dataloaders_dict, model):
        for key in list(dataloaders_dict.keys()):
            dataloaders_dict[key] = None
        del model
        torch.cuda.empty_cache()

    def load_ensemble_model(self):
        model_list = [self.model_obj.init_model() for _ in range(self.ensem)]
        try:
            for i in range(self.ensem):
                model_list[i].load_state_dict(
                    self.load_model_state_dict_from_save_path(
                        self.get_model_checkpoint_path(i)
                    )
                )
        except FileNotFoundError:
            print("Failed to load pretrained models")
            return None

        return model_list

    def _ensemble_results(self, model_list, dataloader):
        print(
            "Getting ensemble results on {} with {} models".format(
                self.device, len(model_list)
            )
        )
        df_columns = ["StockID", "ending_date", "up_prob", "ret_val", "MarketCap"]
        df_dtypes = [object, "datetime64[ns]", np.float64, np.float64, np.float64]
        df_list = []
        for batch in dataloader:
            image = batch["image"].to(self.device, dtype=torch.float)
            if self.model_obj.regression_label is None:
                total_prob = torch.zeros(len(image), 2, device=self.device)
            else:
                total_prob = torch.zeros(len(image), 1, device=self.device)
            for model in model_list:
                model.to(self.device)
                model.eval()
                with torch.set_grad_enabled(False):
                    outputs = model(image)
                    if self.model_obj.regression_label is None:
                        outputs = nn.Softmax(dim=1)(outputs)
                total_prob += outputs
            del image
            batch_df = ut.df_empty(df_columns, df_dtypes)
            batch_df["StockID"] = batch["StockID"]
            batch_df["ending_date"] = pd.to_datetime(
                [str(t) for t in batch["ending_date"]]
            )
            batch_df["ret_val"] = np.nan_to_num(batch["ret_val"].numpy()).reshape(-1)
            batch_df["MarketCap"] = np.nan_to_num(batch["MarketCap"].numpy()).reshape(
                -1
            )
            if self.model_obj.regression_label is None:
                batch_df["up_prob"] = total_prob[:, 1].cpu()
            else:
                batch_df["up_prob"] = total_prob.flatten().cpu()
            df_list.append(batch_df)
        df = pd.concat(df_list)
        df["up_prob"] = 1.0 * df["up_prob"] / len(model_list)
        df.reset_index(drop=True)
        return df

    def generate_ensem_res(self, freq, load_saved_data, year_list=None):
        year_list = list(self.oos_years) if year_list is None else year_list
        year_models_path_dict = {
            y: [self.get_model_checkpoint_path(i) for i in range(self.ensem)]
            for y in year_list
        }
        model_list = [self.model_obj.init_model() for _ in range(self.ensem)]
        for year in year_list:
            freq_surfix = f"_{freq}" if freq != dcf.FREQ_DICT[self.pw] else ""
            ensem_res_path = os.path.join(
                self.ensem_res_dir, f"ensem{self.ensem}_res_{year}{freq_surfix}.csv"
            )
            if os.path.exists(ensem_res_path) and load_saved_data:
                print(f"Found {ensem_res_path}")
                continue
            else:
                print(
                    f"Generating {self.ws}d{self.pw}p ensem results for year {year} with freq {freq}"
                )
                print(
                    "Loading saved model from: {} to {}".format(
                        year_models_path_dict[year][0], year_models_path_dict[year][-1]
                    )
                )
                for i, model in enumerate(model_list):
                    model.load_state_dict(
                        self.load_model_state_dict_from_save_path(
                            year_models_path_dict[year][i]
                        )
                    )

                year_dataloader = self._get_dataloader_for_year(
                    year,
                    freq,
                    remove_tail=(year == self.oos_start_year - 1) and freq != "year",
                )

                df = self._ensemble_results(model_list, year_dataloader)
                df.to_csv(ensem_res_path)

    def _oos_ensem_stat(self):
        ensem_res = self.load_ensem_res_w_period_ret(
            year=self.oos_years, freq=self.pf_freq
        )
        ret_name = "period_ret"

        def _prob_and_ret_rank_corr(df):
            prob_rank = df["up_prob"].rank(method="average", ascending=False)
            ret_rank = df[ret_name].rank(method="average", ascending=False)
            return ret_rank.corr(prob_rank, method="spearman")

        def _prob_and_ret_pearson_corr(df):
            return pd.Series(df["up_prob"]).corr(df[ret_name], method="pearson")

        pred_prob = ensem_res.up_prob.to_numpy()
        label = np.where(ensem_res[ret_name].to_numpy() > 0, 1, 0)
        if self.model_obj.regression_label is not None:
            pred_prob += 0.5
        oos_metrics = ut.calculate_test_log(pred_prob, label)

        rank_corr = ensem_res.groupby("Date").apply(_prob_and_ret_rank_corr)
        pearson_corr = ensem_res.groupby("Date").apply(_prob_and_ret_pearson_corr)
        oos_metrics["Spearman"] = rank_corr.mean()
        oos_metrics["Pearson"] = pearson_corr.mean()
        ut.save_pkl_obj(oos_metrics, self.oos_metrics_path)
        return oos_metrics

    def load_oos_ensem_stat(self):
        oos_metrics_path = self.oos_metrics_path

        if os.path.exists(oos_metrics_path):
            print(f"Loading oos metrics from {oos_metrics_path}")
            oos_metrics = ut.load_pkl_obj(oos_metrics_path)
        else:
            oos_metrics = self._oos_ensem_stat()
        return oos_metrics

    def calculate_portfolio(
        self, load_saved_data=True, delay_list=[0], is_ensem_res=True, cut=10
    ):
        ensem_res_year_list = (
            list(self.is_years) + list(self.oos_years)
            if is_ensem_res
            else list(self.oos_years)
        )
        self.generate_ensem_res(
            self.pf_freq, load_saved_data, year_list=ensem_res_year_list
        )

        oos_metrics = self.load_oos_ensem_stat()
        print(oos_metrics)

        if self.delayed_ret != 0:
            delay_list = delay_list + [self.delayed_ret]
        pf_obj = self.load_portfolio_obj(delay_list)
    
        for delay in delay_list:
            pf_obj.generate_portfolio(delay=delay, cut=cut)
        
        pf_obj.make_portfolio_plot(cut=cut, portfolio_ret=None)

    def load_portfolio_obj(
        self, delay_list=[0], load_signal=True, custom_ret=None, transaction_cost=False
    ):
        if load_signal:
            whole_ensemble_res = self.load_ensem_res(self.oos_years, multiindex=True)
        else:
            whole_ensemble_res = None
        pf_obj = pf.PortfolioManager(
            whole_ensemble_res,
            self.pf_freq,
            self.pf_dir,
            country=self.country,
            delay_list=delay_list,
            load_signal=load_signal,
            custom_ret=custom_ret,
            transaction_cost=transaction_cost,
        )
        return pf_obj

    def load_ensem_res(self, year=None, multiindex=False, freq=None):
        if freq is None:
            freq = self.pf_freq
        assert (year is None) or isinstance(year, int) or isinstance(year, list)
        year_list = (
            self.oos_years
            if year is None
            else [year]
            if isinstance(year, int)
            else year
        )
        df_list = []
        for y in year_list:
            ohlc_str = f"{self.ohlc_len}ohlc" if self.ohlc_len != self.ws else ""
            print(
                f"Loading {self.ws}d{self.pw}p{ohlc_str} ensem results for year {y} with freq {freq}"
            )
            freq_surfix = f"_{freq}" if freq != dcf.FREQ_DICT[self.pw] else ""
            ensem_res_path = os.path.join(
                self.ensem_res_dir, f"ensem{self.ensem}_res_{y}{freq_surfix}.csv"
            )
            if os.path.exists(ensem_res_path):
                print(f"Loading from {ensem_res_path}")
                df = pd.read_csv(
                    ensem_res_path,
                    parse_dates=["ending_date"],
                    index_col=0,
                    engine="python",
                )
                df.StockID = df.StockID.astype(str)
            else:
                self.generate_ensem_res(
                    freq,
                    load_saved_data=True,
                    year_list=list(self.is_years) + list(self.oos_years),
                )
                df = pd.read_csv(
                    ensem_res_path,
                    parse_dates=["ending_date"],
                    index_col=0,
                    engine="python",
                )
                df.StockID = df.StockID.astype(str)
            df_list.append(df)
        whole_ensemble_res = pd.concat(df_list, ignore_index=True)
        whole_ensemble_res.rename(columns={"ending_date": "Date"}, inplace=True)
        whole_ensemble_res.set_index(["Date", "StockID"], inplace=True)
        if self.country == "USA":
            whole_ensemble_res = whole_ensemble_res[["up_prob", "MarketCap"]]
        if not multiindex:
            whole_ensemble_res.reset_index(inplace=True, drop=False)
        whole_ensemble_res.dropna(inplace=True)
        return whole_ensemble_res

    def load_ensem_res_w_period_ret(self, year=None, freq=None):
        if freq is None:
            freq = self.pf_freq
        ensem_res = self.load_ensem_res(year=year, multiindex=True, freq=freq)
        period_ret = eqd.get_period_ret(freq, country=self.country)
        print(f"Loading ensem res with {freq} return of no delay")
        ensem_res["period_ret"] = period_ret[f"next_{freq}_ret"]
        ensem_res.dropna(inplace=True)
        return ensem_res

    def _get_dataloader_for_year(self, year, freq, remove_tail=False):
        if self.model_obj.ts1d_model:
            year_dataset = TS1DDataset(
                self.ws,
                self.pw,
                self.train_freq,
                year,
                country=self.country,
                remove_tail=(year == self.oos_start_year - 1),
                ohlc_len=self.ohlc_len,
                ts_scale=self.ts_scale,
                regression_label=self.model_obj.regression_label,
            )
        else:
            year_dataset = EquityDataset(
                self.ws,
                self.pw,
                freq,
                year,
                country=self.country,
                has_volume_bar=self.has_volume_bar,
                has_ma=self.has_ma,
                annual_stocks_num=self.annual_stocks_num,
                tstat_threshold=self.tstat_threshold,
                stockid_filter=None,
                remove_tail=remove_tail,
                ohlc_len=self.ohlc_len,
                regression_label=self.model_obj.regression_label,
                chart_type=self.chart_type,
                delayed_ret=self.delayed_ret,
            )
        year_dataloader = DataLoader(year_dataset, batch_size=cf.BATCH_SIZE)
        return year_dataloader

    def get_pf_res_path(
        self, weight_type, delay=0, cut=10, file_ext="csv", rank_weight=None
    ):
        assert weight_type in ["vw", "ew"]
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_surfix = "" if cut == 10 else f"_{cut}cut"
        rank_weight_surfix = "" if rank_weight is None else f"_rank_{rank_weight}"
        pf_name = f"{delay_prefix}{weight_type}_{cut_surfix}{rank_weight_surfix}"
        return os.path.join(self.pf_dir, f"{pf_name}.{file_ext}")

    def get_pf_data(
        self, weight_type, value_filter=100, delay=0, cut=10, rank_weight=None
    ):
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_surfix = "" if cut == 10 else f"_{cut}cut"
        rank_weight_surfix = "" if rank_weight is None else f"_rank_{rank_weight}"
        pf_name = f"{delay_prefix}{weight_type}_{value_filter}{cut_surfix}{rank_weight_surfix}"
        pf_data_path = os.path.join(self.pf_dir, "pf_data", f"pf_data_{pf_name}.csv")
        print(f"Loading portfolio data from {pf_data_path}")
        df = pd.read_csv(pf_data_path, index_col=0, parse_dates=True)
        return df

    def calculate_oos_up_prob(self):
        df = pd.DataFrame()
        for y in self.oos_years:
            try:
                ensem_res = self.load_ensem_res_w_period_ret(y)
            except FileNotFoundError:
                continue
            df.loc[y, "Sample Number"] = len(ensem_res)
            df.loc[y, "True Up Pct"] = np.sum(ensem_res.period_ret > 0.0) / len(
                ensem_res
            )
            df.loc[y, "Pred Up Pct"] = np.sum(ensem_res.up_prob > 0.5) / len(ensem_res)
            df.loc[y, "Mean Up Prob"] = ensem_res.up_prob.mean()
        df.loc["Mean"] = df.mean(axis=0)
        df = df.round(2)
        df["Sample Number"] = df["Sample Number"].astype(int)
        df.to_csv(os.path.join(self.ensem_res_dir, f"oos_up_prob.csv"))
        return df.loc["Mean"]

    def summarize_true_up_label(self):
        tv_df = self._df_true_up_label(list(range(2017, 2022)), "In Sample")
        test_df = self._df_true_up_label(list(range(2022, 2024)), "OOS")
        df = pd.concat([tv_df, test_df])
        df.to_csv(
            os.path.join(
                cf.LOG_DIR,
                f"{self.ws}d{self.pw}p_vb{self.has_volume_bar}_ma{self.has_ma}_oos_up_prob.csv",
            )
        )
        with open(
            os.path.join(
                cf.LOG_DIR,
                f"{self.ws}d{self.pw}p_vb{self.has_volume_bar}_ma{self.has_ma}_oos_up_prob.txt",
            ),
            "w+",
        ) as f:
            f.write(df.to_latex())

    def _df_true_up_label(self, year_list, datatype):
        df = pd.DataFrame(
            index=year_list,
            columns=[
                "Sample Number",
                "True Up Pct",
                "Accy",
                "Pred Up Pct",
                "Mean Up Prob",
                "Accy (Pred Up)",
                "Accy (Pred Down)",
            ],
        )
        for y in year_list:
            try:
                ensem_res = self.load_ensem_res_w_period_ret(year=y)
            except FileNotFoundError:
                continue

            print(
                f"{np.sum(np.isnan(ensem_res.period_ret))}/{len(ensem_res)} of ret_val is Nan"
            )
            label = np.where(ensem_res.period_ret > 0, 1, 0)
            pred = np.where(ensem_res.up_prob > 0.5, 1, 0)
            df.loc[y, "Sample Number"] = len(ensem_res)
            df.loc[y, "True Up Pct"] = np.sum(ensem_res.period_ret > 0.0) / len(
                ensem_res
            )
            df.loc[y, "Accy"] = np.sum(label == pred) / len(label)
            df.loc[y, "Pred Up Pct"] = np.sum(ensem_res.up_prob > 0.5) / len(ensem_res)
            df.loc[y, "Mean Up Prob"] = ensem_res.up_prob.mean()
            df.loc[y, "Accy (Pred Up)"] = np.sum(
                label[pred == 1] == pred[pred == 1]
            ) / len(pred[pred == 1])
            df.loc[y, "Accy (Pred Down)"] = np.sum(
                label[pred == 0] == pred[pred == 0]
            ) / len(pred[pred == 0])
        df.loc[f"{datatype} Mean"] = df.mean(axis=0)
        df = df.astype(float).round(2)
        df["Sample Number"] = df["Sample Number"].astype(int)
        return df


def get_exp_obj_by_spec(
    ws,
    pw,
    train_freq=None,
    ensem=5,
    dn=0,
    country="USA",
    transfer_learning=None,
    is_years=cf.IS_YEARS,
    oos_years=cf.OOS_YEARS,
    layer_number=None,
    inplanes=cf.TRUE_DATA_CNN_INPLANES,
    filter_size=None,
    max_pooling=None,
    stride=None,
    dilation=None,
    filter_size_list=None,
    stride_list=None,
    dilation_list=None,
    max_pooling_list=None,
    batch_norm=True,
    drop_prob=0.5,
    xavier=True,
    lrelu=True,
    has_ma=True,
    has_volume_bar=True,
    bn_loc="bn_bf_relu",
    conv_layer_chanls=None,
    weight_decay=0,
    loss_name="cross_entropy",
    lr=1e-5,
    tensorboard=False,
    margin=1,
    train_size_ratio=0.7,
    pf_freq=None,
    ohlc_len=None,
    tstat_filter=0,
    stocks_for_train="all",
    regression_label=None,
    chart_type="bar",
    delayed_ret=0,
    ts1d_model=False,
    ts_scale="image_scale",
):
    train_freq = dcf.FREQ_DICT[pw] if train_freq is None else train_freq
    ohlc_len_ = ohlc_len if ohlc_len is not None else ws
    layer_number = (
        cf.BENCHMARK_MODEL_LAYERNUM_DICT[ohlc_len_]
        if layer_number is None
        else layer_number
    )

    filter_size_list = (
        cf.EMP_CNN_BL_SETTING[ohlc_len_][0]
        if filter_size_list is None
        else filter_size_list
    )
    stride_list = (
        cf.EMP_CNN_BL_SETTING[ohlc_len_][1] if stride_list is None else stride_list
    )
    dilation_list = (
        cf.EMP_CNN_BL_SETTING[ohlc_len_][2] if dilation_list is None else dilation_list
    )
    max_pooling_list = (
        cf.EMP_CNN_BL_SETTING[ohlc_len_][3]
        if max_pooling_list is None
        else max_pooling_list
    )

    model_obj = cnn_model.Model(
        ohlc_len_,
        layer_number,
        inplanes=inplanes,
        drop_prob=drop_prob,
        filter_size=filter_size,
        stride=stride,
        dilation=dilation,
        max_pooling=max_pooling,
        filter_size_list=filter_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        max_pooling_list=max_pooling_list,
        batch_norm=batch_norm,
        xavier=xavier,
        lrelu=lrelu,
        bn_loc=bn_loc,
        conv_layer_chanls=conv_layer_chanls,
        regression_label=regression_label,
        ts1d_model=ts1d_model,
    )
    exp_obj = Experiment(
        ws,
        pw,
        model_obj,
        train_freq=train_freq,
        ensem=ensem,
        lr=lr,
        drop_prob=drop_prob,
        device_number=dn,
        max_epoch=50,
        enable_tqdm=True,
        early_stop=True,
        has_ma=has_ma,
        has_volume_bar=has_volume_bar,
        is_years=is_years,
        oos_years=oos_years,
        weight_decay=weight_decay,
        loss_name=loss_name,
        tensorboard=tensorboard,
        margin=margin,
        train_size_ratio=train_size_ratio,
        country=country,
        transfer_learning=transfer_learning,
        pf_freq=pf_freq,
        ohlc_len=ohlc_len,
        tstat_threshold=tstat_filter,
        annual_stocks_num=stocks_for_train,
        chart_type=chart_type,
        delayed_ret=delayed_ret,
        ts_scale=ts_scale,
    )

    return exp_obj


def get_bl_exp_obj(
    ws,
    pw,
    dn=0,
    train_freq=None,
    drop_prob=0.5,
    train_size_ratio=0.7,
    ensem=5,
    is_years=cf.IS_YEARS,
    oos_years=cf.OOS_YEARS,
    country="USA",
    inplanes=cf.TRUE_DATA_CNN_INPLANES,
    transfer_learning=None,
    has_ma=True,
    has_volume_bar=True,
    pf_freq=None,
    ohlc_len=None,
    tstat_filter=0,
    stocks_for_train="all",
    batch_norm=True,
    chart_type="bar",
    delayed_ret=0,
    ts1d_model=False,
    ts_scale="image_scale",
    regression_label=None,
    lr=1e-5,
) -> Experiment:
    ohlc_len_ = ohlc_len if ohlc_len is not None else ws

    if ts1d_model:
        (
            filter_size_list,
            stride_list,
            dilation_list,
            max_pooling_list,
        ) = cf.EMP_CNN1d_BL_SETTING[ohlc_len_]
        layer_number = cf.TS1D_LAYERNUM_DICT[ohlc_len_]
    else:
        (
            filter_size_list,
            stride_list,
            dilation_list,
            max_pooling_list,
        ) = cf.EMP_CNN_BL_SETTING[ohlc_len_]
        layer_number = cf.BENCHMARK_MODEL_LAYERNUM_DICT[ohlc_len_]

    exp_obj = get_exp_obj_by_spec(
        ws,
        pw,
        train_freq=dcf.FREQ_DICT[pw]
        if ohlc_len is None
        else train_freq
        if train_freq is not None
        else "month",
        ensem=ensem,
        dn=dn,
        country=country,
        is_years=is_years,
        oos_years=oos_years,
        transfer_learning=transfer_learning,
        layer_number=layer_number,
        inplanes=inplanes,
        filter_size=(5, 3),
        max_pooling=(2, 1),
        stride=(1, 1),
        dilation=(1, 1),
        filter_size_list=filter_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        max_pooling_list=max_pooling_list,
        batch_norm=batch_norm,
        drop_prob=drop_prob,
        xavier=True,
        lrelu=True,
        has_ma=has_ma,
        has_volume_bar=has_volume_bar,
        bn_loc="bn_bf_relu",
        conv_layer_chanls=None,
        weight_decay=0,
        loss_name="cross_entropy",
        lr=lr,
        tensorboard=False,
        margin=1,
        train_size_ratio=train_size_ratio,
        pf_freq=pf_freq,
        ohlc_len=ohlc_len,
        tstat_filter=tstat_filter,
        stocks_for_train=stocks_for_train,
        chart_type=chart_type,
        delayed_ret=delayed_ret,
        ts1d_model=ts1d_model,
        ts_scale=ts_scale,
        regression_label=regression_label,
    )

    return exp_obj


def run_arch_comparison(
    ws=20,
    pw=20,
    ensem=5,
    load_saved_data=True,
    dn=None,
    ohlc_len=None,
    tl=None,
    pretrained=True,
    chart_type="bar",
    total_worker=1,
    worker_idx=0,
):
    torch.set_num_threads(1)
    if total_worker > 1:
        assert 0 < (worker_idx + 1) <= total_worker
        dn = worker_idx % 2 if dn is None else dn
    else:
        worker_idx = 0
        dn = dn if dn is not None else 0

    _ohlc_len = ws if ohlc_len is None else ohlc_len
    param_dict_list = [
        dict(drop_prob=0),
        dict(drop_prob=0.25),
        dict(drop_prob=0.75),
        dict(inplanes=32),
        dict(inplanes=128),
        dict(layer_number=cf.BENCHMARK_MODEL_LAYERNUM_DICT[_ohlc_len] + 1),
        dict(layer_number=cf.BENCHMARK_MODEL_LAYERNUM_DICT[_ohlc_len] - 1),
        dict(batch_norm=False),
        dict(xavier=False),
        dict(lrelu=False),
        dict(max_pooling_list=[(2, 2)] * 10),
        dict(filter_size_list=[(3, 3)] * 10),
        dict(filter_size_list=[(7, 3)] * 10),
    ]

    if ws != 5:
        param_dict_list.append(dict(stride_list=[(1, 1)] * 10))
        param_dict_list.append(dict(dilation_list=[(1, 1)] * 10))
        param_dict_list.append(
            dict(dilation_list=[(1, 1)] * 10, stride_list=[(1, 1)] * 10)
        )

    print(
        f"Worker {worker_idx} from {total_worker} workers for {len(param_dict_list)} jobs"
    )
    worker_setting_list = [
        param_dict_list[i]
        for i in range(worker_idx, len(param_dict_list), total_worker)
    ]
    for i, param_dict in enumerate(worker_setting_list):
        print(
            f"\n\n################\n{i}/{len(worker_setting_list)} job\n################\n\n"
        )
        exp_obj = get_exp_obj_by_spec(
            ws,
            pw,
            train_freq=None,
            ensem=ensem,
            dn=dn,
            ohlc_len=ohlc_len,
            transfer_learning=tl,
            chart_type=chart_type,
            **param_dict,
        )
        print(exp_obj.model_dir)
        ew_pf_path, vw_pf_path = exp_obj.get_pf_res_path("ew"), exp_obj.get_pf_res_path(
            "vw"
        )
        if os.path.exists(ew_pf_path) and os.path.exists(vw_pf_path):
            print(f"Found {ew_pf_path} and {vw_pf_path}")
            print(exp_obj.load_oos_ensem_stat())
            continue
        exp_obj.train_empirical_ensem_model(pretrained=pretrained)
        exp_obj.calculate_portfolio(load_saved_data=load_saved_data, is_ensem_res=False)


def train_us_model(
    ws_list,
    pw_list,
    dp=0.50,
    ensem=5,
    total_worker=1,
    dn=None,
    from_ensem_res=True,
    ensem_range=None,
    train_size_ratio=0.7,
    is_ensem_res=True,
    vb=True,
    ma=True,
    tstat_filter=0,
    stocks_for_train="all",
    batch_norm=True,
    chart_type="bar",
    delayed_ret=0,
    calculate_portfolio=False,
    ts1d_model=False,
    ts_scale="image_scale",
    regression_label=None,
    pf_delay_list=[0],
    lr=1e-5,
):
    torch.set_num_threads(1)
    if total_worker > 1:
        worker_idx = int(sys.argv[1])
        assert 0 < (worker_idx + 1) <= total_worker
    else:
        worker_idx = 0

    setting_list = list(itertools.product(ws_list, pw_list))

    print(
        f"Worker {worker_idx} from {total_worker} workers for {len(setting_list)} jobs"
    )

    worker_setting_list = [
        setting_list[i] for i in range(worker_idx, len(setting_list), total_worker)
    ]
    if dn is None:
        dn = (worker_idx + 0) % 2

    for ws, pw in worker_setting_list:
        exp_obj = get_bl_exp_obj(
            ws,
            pw,
            dn=dn,
            drop_prob=dp,
            train_size_ratio=train_size_ratio,
            ensem=ensem,
            has_volume_bar=vb,
            has_ma=ma,
            tstat_filter=tstat_filter,
            stocks_for_train=stocks_for_train,
            batch_norm=batch_norm,
            chart_type=chart_type,
            delayed_ret=delayed_ret,
            ts1d_model=ts1d_model,
            ts_scale=ts_scale,
            regression_label=regression_label,
            lr=lr,
        )

        exp_obj.train_empirical_ensem_model(ensem_range=ensem_range)
        if calculate_portfolio:
            exp_obj.calculate_portfolio(
                load_saved_data=from_ensem_res,
                is_ensem_res=is_ensem_res,
                delay_list=pf_delay_list,
            )
        del exp_obj


def main():
    pass


if __name__ == "__main__":
    main()
