import pandas as pd
import numpy as np
import os

from config import *

def find_exchange():
    df = load_market_data()
    df['exchangeSymbol'] = df['exchangeSymbol'].str.upper()
    exchanges = pd.read_csv("exchanges_cap.csv")
    
    # 찾지 못한 symbol을 저장할 리스트 초기화
    found_symbols = []
    not_found_symbols = []

    # df의 exchangeSymbol 컬럼에서 각 고유한 값에 대해 검사
    for symbol in df['exchangeSymbol'].unique():
        # exchanges의 symbol 컬럼에서 일치하는 것이 있는지 확인
        if symbol not in exchanges['symbol'].values:
            # exchanges의 goog_symbol 컬럼에서 일치하는 것이 있는지 확인
            if symbol not in exchanges['goog_symbol'].values:
                # 둘 다 일치하지 않으면 not_found_symbols 리스트에 추가
                not_found_symbols.append(symbol)
            else:
                # 일치하면 found_symbols 리스트에 추가
                found_symbols.append(symbol)
        else:
            # 일치하면 found_symbols 리스트에 추가
            found_symbols.append(symbol)

    # 결과 출력
    print("Symbols found:", found_symbols)
    print("Symbols not found:", not_found_symbols)
    
    return found_symbols, not_found_symbols