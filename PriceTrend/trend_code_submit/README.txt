########################################################################################################################
Create Conda Virtual Environment
########################################################################################################################

To install the conda virtual environment, run command:
  conda env create -f cnn_env.yml -p ../cnn_env
It will create a folder named "cnn_env" for the virtual environment
To activate the virtual environment, under the code directory (trend_submit), run command:
  conda activate ../cnn_env


########################################################################################################################
Code Index (See sample_codes.py for sample codes to generate images and train CNN models)
########################################################################################################################

############################################################
Data:
############################################################
  dgp_config.py: configuration file for the data generation
  equity_data.py: preprocess raw US return data downloaded from CRSP
  chart_library.py: helper class of drawing chart images from raw data
  chart_dataset.py: pytorch dataset objects of CNN/CNN1D
  generate_chart.py: class of image data generation
      save_annual_data() to generate CNN data by year
      save_annual_ts_data() to generate CNN1D data by year

############################################################
Experiments:
############################################################
  cnn_experiment.py: primary script for running CNN experiments under different model settings and training setups

############################################################
Misc:
############################################################
  cache_manager.py: helper class for cache storage
  config.py: general configuration file
  utilities.py: utility functions for calculating performance metrics and generating latex texts.

############################################################
Model:
############################################################
  cnn_model.py: file that defines CNNModel and CNN1DModel torch.nn classes, which are the primary models for prediction

############################################################
Portfolio:
############################################################
  portfolio.py: class for generating long-short portfolio based on prediction signals

############################################################
Analysis:
############################################################
  helper functions generating tables in the paper, see instructions below


########################################################################################################################
Generate paper tables and figures

CACHE_DIR should be placed under the same parent directory as trend_submit, i.e. (~/trend_submit/trend_code_submit/ and ~/trend_submit/CACHE_DIR_SUBMIT)
All data under CACHE_DIR has been capped and only the first 50 and last 50 rows are kept.
########################################################################################################################

Table I to IV
Analysis.analysis_lib:
portfolio_performance_helper(ws: int, pw: int)
Note:
ws=20, pw=5 corresponds to I20/R5 model
For Table 3, 20-day returns are split into day 1 to day 5 and day 6 to day 20. The portfolio return are generated
based on the split returns.
For Table 4, a filter of 500 stocks with the largest market cap is applied on the prediction before generating portfolios.

Table V Correlation Between CNN Predictions and Stock Characteristics
Analysis.analysis_lib:
corr_between_cnn_pred_and_stock_chars()

Table VI: CNN Predictions and Standard Stock Characteristics
Analysis.regression_tables:
cnn_pred_on_monthly_stock_char(pw=5, regression_type="logit")

Table VII: CNN, Future Returns, and Standard Stock Characteristics
Analysis.regression_tables:
cnn_and_ret_and_stock_char_regression(pw=5, ws_list=[5, 20, 60], regression_type="logit")

Table VIII: Logistic Regressions Using Market Data With Image Scaling
Analysis.regression_tables:
regress_ret_on_cnn_pred_and_raw_image_data(
   ws=5, regression_type="logit", pw_list=[5, 20, 60]
)
regression_on_market_data_combined(
   ws=5, regression_type="logit", pw_list=[5, 20, 60]
)

Table IX: Portfolio Performance of CNN versus Logistic Model and CNN1D
Analysis.analysis_lib:
cnn_vs_linear_table_ols(5)
cnn_vs_linear_table_ols(20)
cnn_vs_linear_table_ols(60)

Table X: International Transfer and H-L Decile Portfolio Sharpe Ratios (I5/R5)
Analysis.analysis_lib:
glb_plot_sr_gain_vs_stocks_num(5)

Table XI and XII: Time Scale Transfer I5/R5 to I20/R20 and I60/R60
Analysis.analysis_lib:
time_scale_transfer_portfolio_helper(20)
time_scale_transfer_portfolio_helper(60)

Figure 5: Cumulative volatility-adjusted returns of equal-weight portfolios.
The figure can be plotted using the portfolios generated from weekly predictions ("weekly_prediction_with_rets.csv")
The portfolio returns can be found from CACHE_DIR/PORTFOLIO/ after running portfolio_performance_helper(ws, pw)

Figure 6: Prediction accuracy by decile.
The figure can be plotted using weekly predictions ("weekly_prediction_with_rets.csv") and future returns ("us_week_ret.pq")

Figure 7: Prediction accuracy of CNN and logistic models by decile.
The figure can be plotted using weekly predictions ("CACHE_DIR/cnn1d_and_linear_model_portfolio_returns/") and future returns (us_week_ret.pq)

Figure 8: Comparison with traditional technical indicators.
Equal-weight and value-weight long-short portfolio returns for all 7,846 technical indicators are stored in
   CACHE_DIR/technical_indicators_portfolio_ret_combined_weekly.csv
   CACHE_DIR/technical_indicators_portfolio_ret_combined_monthly.csv
   CACHE_DIR/technical_indicators_portfolio_ret_combined_quarterly.csv

Figure 9: Sharpe ratio gains from international transfer.
The figure can be plotted using functions to generate Table 10
