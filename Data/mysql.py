import pymysql
from sqlalchemy import create_engine
import pandas as pd

def mySQL_read(table_name: str, query: str = f"SELECT * FROM {table_name}"):
    user = 'username' # 본인 상황에 맞게 수정하세요
    password = 'userpassword' # 본인 상황에 맞게 수정하세요
    host = '192.168.0.2'  # 포트 번호를 여기에 포함하지 않습니다.
    port = 9999  # 포트 번호를 별도의 변수로 설정합니다.
    database = 'db'
    table_name = 'test'

    # SQLAlchemy 엔진 생성 시 호스트와 포트를 정확하게 지정합니다.
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

    # pymysql 연결 시에도 마찬가지로 호스트와 포트를 정확하게 지정합니다.
    connection = pymysql.connect(host=host, port=port, user=user, password=password, database=database)

    # 예시 호출
    table_name = 'estimates'
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    
    return df

def mySQL_tosql(table_name: str):
    user = 'username' # 본인 상황에 맞게 수정하세요
    password = 'userpassword' # 본인 상황에 맞게 수정하세요
    host = '192.168.0.2'  # 포트 번호를 여기에 포함하지 않습니다.
    port = 9999  # 포트 번호를 별도의 변수로 설정합니다.
    database = 'db'
    table_name = 'test'

    # SQLAlchemy 엔진 생성 시 호스트와 포트를 정확하게 지정합니다.
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")

    # pymysql 연결 시에도 마찬가지로 호스트와 포트를 정확하게 지정합니다.
    connection = pymysql.connect(host=host, port=port, user=user, password=password, database=database)

    # 예시 호출
    file = 'company'
    df = pd.read_parquet(f'{file}.parquet')

    try:
        # MySQL에 테이블 생성
        df.to_sql(file, con=engine, if_exists='replace', index=False)
        print(f'MySQL에 {file} 데이터 저장 완료')
        return True
    
    except Exception as e:
            print(f"Error processing file {file}: {e}")

def main():
    pass

if __name__ == "__main__":
    main()