import pandas as pd

def load_market_data():
    path = './_SNP_dataset/market_by_exchange_FullPricingDate'
    df = pd.read_parquet(f'{path}/market_FullPricingDate.parquet', engine='fastparquet')
    return df