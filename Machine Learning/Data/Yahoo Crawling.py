### Yahoo Data 수집방법
### KRX에서 시장 데이터와 구성 종목 받아오고,
### 그 데이터 바탕으로 Yahoo에서 데이터 긁어오기

import pandas as pd
import pandas_datareader as pdr
import datetime
import os

# 종목 타입에 따라 download url이 다름. 종목코드 뒤에 .KS .KQ등이 입력, Download Link 구분
stock_type = {
'kospi': 'stockMkt',
'kosdaq': 'kosdaqMkt'
}

# 회사명으로 주식 종목 코드를 획득할 수 있도록 하는 함수
def get_code(df, name):
    code = df.query("name=='{}'".format(name))['code'].to_string(index=False)
    # 위와같이 code명을 가져오면 sript() 하여 공백 제거
    code = code.strip()
    return code

# download url 조합
def get_download_stock(market_type=None):
    market_type = stock_type[market_type]
    download_link = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
    download_link = download_link + '?method=download'
    download_link = download_link + '&marketType=' + market_type
    df = pd.read_html(download_link, header=0)[0]
    return df

# kospi 종목코드 목록 다운로드
def get_download_kospi():
    df = get_download_stock('kospi')
    df.종목코드 = df.종목코드.map('{:06d}.KS'.format)
    return df

# kosdaq 종목코드 목록 다운로드
def get_download_kosdaq():
    df = get_download_stock('kosdaq')
    df.종목코드 = df.종목코드.map('{:06d}.KQ'.format)
    return df

# kospi, kosdaq 종목코드 각각 다운로드
kospi_df = get_download_kospi()
kosdaq_df = get_download_kosdaq()

# data frame정리
kospi_df = kospi_df[['회사명', '종목코드']]
kosdaq_df = kosdaq_df[['회사명', '종목코드']]

# data frame title 변경 '회사명' = name, 종목코드 = 'code'
kospi_df = kospi_df.rename(columns={'회사명': 'name', '종목코드': 'code'})
kosdaq_df = kosdaq_df.rename(columns={'회사명': 'name', '종목코드': 'code'})


#데이터 시간 길이
start_date = datetime.datetime(2001,1,1)
end_date = datetime.datetime(2021,10,14)

#디렉토리 생성
directory = "KOSPI"
directory2 = "KOSDAQ"
path = os.getcwd()
os.mkdir(path + "/" + directory)
os.mkdir(path + "/" + directory2)

#오류발생시 체크 (ETF에서 가끔)
for i in range(len(kospi_df)):
    try:
        df = pdr.get_data_yahoo(kospi_df.iloc[i,1], start_date, end_date)
        df.to_csv(f'{directory}\{kospi_df.iloc[i,0]}.csv')
    except:
        pass

#메인실행
Kospi_index = pdr.get_data_yahoo('^KS11', start_date, end_date)
Kospi_index.to_csv('KOSPI Index.csv')

# 삼성전자의 종목코드 획득. data frame에는 이미 XXXXXX.KX 형태로 조합이 되어있음
#code = get_code(code_df, '삼성전자')

# get_data_yahoo API를 통해서 yahho finance의 주식 종목 데이터를 가져온다.
#df = pdr.get_data_yahoo(code)

