from Experiments.cnn_experiment import train_us_model
from Data.generate_chart import GenerateStockData
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


if __name__ == "__main__":
    # Generate Image Data
    year_list = list(range(2017, 2024))
    chart_type = "bar"
    ws = 20
    freq = "week"
    ma_lags = [ws]
    vb = True
    for year in year_list:
        print(f"{ws}D {freq} {chart_type} {year}")
        dgp_obj = GenerateStockData(
            "USA",
            year,
            ws,
            freq,
            chart_freq=1,  # for time-scale I20/R20 to R5/R5, set ws=20 and chart_freq=4
            ma_lags=ma_lags,
            volume_bar=vb,
            need_adjust_price=True,
            allow_tqdm=True,
            chart_type=chart_type,
        )
        # generate CNN2D Data
        dgp_obj.save_annual_data()
        # generate CNN1D Data
        # dgp_obj.save_annual_ts_data()

    # Train CNN Models for US
    # CNN2D
    train_us_model(
        [20],
        [5],
        total_worker=1,
        calculate_portfolio=True,
        ts1d_model=False,
        ts_scale="image_scale",
        regression_label=None,
        pf_delay_list=[0],
        lr=1e-4,
    )
    # train_us_model(
    #     [20],
    #     [20],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    # train_us_model(
    #     [20],
    #     [60],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    # # CNN1D
    # train_us_model(
    #     [20],
    #     [20],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=True,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    # # Timescale
    # train_us_model(
    #     [20],
    #     [20],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-5,
    # )
    # train_us_model(
    #     [60],
    #     [60],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-5,
    # )
