import pandas as pd
import numpy as np
import os

from config import *

def find_exchange():
    df = load_market_data()
    df['exchangeSymbol'] = df['exchangeSymbol'].str.upper()
    exchanges = pd.read_csv("exchanges_cap.csv")