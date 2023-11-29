import os


def get_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return dir

WORK_DIR = get_dir("../WORK_SPACE")

EXP_DIR = get_dir(os.path.join(WORK_DIR, "new_model_res"))
PORTFOLIO_DIR = get_dir(os.path.join(EXP_DIR, "portfolio"))
LOG_DIR = get_dir(os.path.join(EXP_DIR, "log"))
LATEX_DIR = get_dir(os.path.join(EXP_DIR, "latex"))
LIGHT_CSV_RES_DIR = get_dir(os.path.join(WORK_DIR, "torch_ta/ta/csv_res/"))

BATCH_SIZE = 128
TRUE_DATA_CNN_INPLANES = 64
BENCHMARK_MODEL_LAYERNUM_DICT = {5: 2, 20: 3, 60: 4}
EMP_CNN_BL_SETTING = {
    5: ([(5, 3)] * 10, [(1, 1)] * 10, [(1, 1)] * 10, [(2, 1)] * 10),
    20: (
        [(5, 3)] * 10,
        [(3, 1)] + [(1, 1)] * 10,
        [(2, 1)] + [(1, 1)] * 10,
        [(2, 1)] * 10,
    ),
    60: (
        [(5, 3)] * 10,
        [(3, 1)] + [(1, 1)] * 10,
        [(3, 1)] + [(1, 1)] * 10,
        [(2, 1)] * 10,
    ),
}

TS1D_LAYERNUM_DICT = {5: 1, 20: 2, 60: 3}
EMP_CNN1d_BL_SETTING = {
    5: ([3] * 1, [1] * 1, [1] * 1, [2] * 1),
    20: ([3] * 2, [1] * 2, [1] * 2, [2] * 2),
    60: ([3] * 3, [1] * 3, [1] * 3, [2] * 3),
}

NUM_WORKERS = 1


START_YEAR_DICT = {
    "Russia": 1999,
    "Greece": 1999,
    "Finland": 1999,
    "Ireland": 1999,
    "Sweden": 1999,
}
IS_YEARS = list(range(2017, 2022))
OOS_YEARS = list(range(2022, 2024))

BENCHMARK_MODEL_NAME_DICT = {
    5: "D5L2F53S1F53S1C64MP11",
    20: "D20L3F53S1F53S1F53S1C64MP111",
    60: "D60L4F53S1F53S1F53S1F53S1C64MP1111",
}

latex_figw, latex_figh = 20.0, 15.0
latex_font_size = 18
latex_subtitle_font = 16
latex_pad, latex_w_pad, latex_h_pad = 10.0, 2.0, 2.0
latex_legend_x, latex_legend_y, latex_legend_font = 0.5, 0.92, 18
latex_line_width = 2
hl_style = {
    "SPY (EW)": "y",
    "H (EW)": "b",
    "L (EW)": "r",
    "H-L (EW)": "k",
    "P/L (EW)": "b",
    "SPY (VW)": "y:",
    "H (VW)": "b:",
    "L (VW)": "r:",
    "H-L (VW)": "k:",
    "P/L (VW)": "b:",
}

hl_style_w_marker = {
    "SPY (EW)": "yo-",
    "H (EW)": "bo-",
    "L (EW)": "ro-",
    "H-L (EW)": "ko-",
    "P/L (EW)": "bo-",
    "SPY (VW)": "yo:",
    "H (VW)": "bo:",
    "L (VW)": "ro:",
    "H-L (VW)": "ko:",
    "P/L (VW)": "bo:",
}

bar_color = {
    "H (EW)": "b",
    "L (EW)": "r",
    "H-L (EW)": "k",
    "P/L (EW)": "b",
    "H (VW)": "b",
    "L (VW)": "r",
    "H-L (VW)": "k",
    "P/L (VW)": "b",
}

bar_color_6 = ["b", "r", "k", "cornflowerblue", "salmon", "dimgray"]
bar_color_2 = ["b", "cornflowerblue"]
