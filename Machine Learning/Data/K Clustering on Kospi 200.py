### Clustering을 활용한 주식분석
### Kospi 200 기업중, 10년간 데이터가 존재하는 주식으로 분류함
### 기존의 11개의 분류에서, 보다 비슷하게 움직이는 분류로 세밀하게 나누고자 함.
### 결과적으로 기존 섹터끼리 움직임은 비슷하기도 하나, 그 이외 자회사나, 특정 분야에서
### 같은 섹터로 분류된 것이 있는 걸로 봐서 이를 토대로 투자전략을 수립할 가능성이 있어보임

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

stock_type = {
'kospi': 'stockMkt',
'kosdaq': 'kosdaqMkt'
}

# 회사명으로 주식 종목 코드를 획득할 수 있도록 하는 함수
def get_code(df, name):
    code = df.query("name=='{}'".format(name))['code'].to_string(index=False)
    # 위와같이 code명을 가져오면 앞에 공백이 붙어있는 상황이 발생하여 앞뒤로 sript() 하여 공백 제거
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

kospi_df = get_download_kospi()

path_dir = "../K_Clustering/KOSPI Data"
file_list = os.listdir(path_dir)

#KOSPI 200 종목들 섹터별로
#건설
sec1 = ['대림건설', '대우건설', '삼성엔지니어링', '쌍용양회', '아이에스동서', '태영건설', '포스코케미칼', '한전기술', '한일현대시멘트', '현대건설', 'GS건설', 'HDC', 'HDC현대산업개발', 'LG하우시스']
#금융
sec2 = ['기업은행', '메리츠증권', '미래에셋증권', '삼성생명', '삼성증권', '삼성카드', '삼성화재', '신한지주', '우리금융지주', '키움증권', '하나금융지주', '한국금융지주', '한화생명', '현대해상', 'BNK금융지주', 'DB손해보험', 'KB금융', 'NH투자증권']
#경기소비재
sec3 = ['강원랜드', '금호타이어', '기아차', '넥센타이어', '더블유게임즈', '락앤락', '롯데관광개발', '롯데쇼핑', '롯데하이마트', '만도', '삼성물산', '세방전지', '신세계', '신세계인터내셔날', '영원무역', '지누스', '코웨이', '쿠쿠홀딩스', '쿠쿠홈시스', '한국앤컴퍼니', '한국타이어앤테크놀로지', '한샘', '한섬', '한세실업', '한온시스템', '현대모비스', '현대백화점', '현대위아', '현대차', '현대홈쇼핑', '호텔신라', '화승엔터프라이즈', '휠라홀딩스', 'F&F', 'GKL', 'SNT모티브']
#산업재
sec4 = ['대한항공', '두산', '두산퓨얼셀', '아시아나항공', '에스원', '팬오션', '포스코인터내셔널', '한국항공우주', '한전KPS', '한진칼', '한화시스템', '한화에어로스페이스', '현대글로비스', 'CJ대한통운', 'HMM', 'LX인터내셔널', 'LIG넥스원', 'LS', 'LS ELECTRIC', 'SK네트웍스']
#생활소비재
sec5 = ['농심', '대상', '동서', '동원F&B', '롯데지주', '롯데칠성', '빙그레', '삼양사', '삼양식품', '삼양홀딩스', '아모레G', '아모레퍼시픽', '애경산업', '오뚜기', '오리온', '오리온홀딩스', '이마트', '코스맥스', '하이트진로', '한국가스공사', '한국전력', '한국콜마', '현대그린푸드', 'BGF리테일', 'CJ', 'CJ제일제당', 'GS리테일', 'KT&G', 'LG생활건강', 'SPC삼립']
#에너지/화학
sec6 = ['금호석유', '대한유화', '롯데케미칼', '롯데정밀화학', '코오롱인더', '태광산업', '한솔케미칼', '한화', '한화솔루션', '효성', '후성', '휴켐스', 'GS', 'KCC', 'LG화학', 'OCI', 'S-Oil', 'SK', 'SKC', 'SK디스커버리', 'SK이노베이션', 'SK케미칼']
#정보기술
sec7 = ['삼성SDI', '삼성SDS', '삼성전기', '삼성전자', '일진머티리얼즈', 'DB하이텍', 'LG', 'LG디스플레이', 'LG이노텍', 'LG전자', 'SK하이닉스']
#중공업
sec8 = ['대우조선해양', '두산인프라코어', '두산밥캣', '두산중공업', '삼성중공업', '씨에스윈드', '한국조선해양', '현대로템', '현대미포조선', '현대엘리베이', '현대중공업지주']
#철강/소재
sec9 = ['고려아연', '남선알미늄', '동국제강', '동원시스템즈', '영풍', '풍산', '현대제철', 'KG동부제철', 'POSCO']
#커뮤니케이션서비스
sec10 = ['넷마블', 'HYBE', '엔씨소프트', '이노션', '제일기획', '카카오', 'CJ CGV', 'KT', 'LG유플러스', 'NAVER', 'SK텔레콤']
#헬스케어
sec11 = ['녹십자', '녹십자홀딩스', '대웅', '대웅제약', '보령제약', '부광약품', '삼성바이오로직스', '셀트리온', '신풍제약', '영진약품', '유한양행', '일양약품', '종근당', '한미사이언스', '한미약품', '한올바이오파마', 'JW중외제약', 'SK바이오팜']

#코스피200 종목 리스트
sec = []
for i in range(1,12):
    a = globals()[f'sec{i}']
    sec = sec + a

#날짜인덱스 맞추기
start_date = datetime.strptime('2001-01-02', '%Y-%m-%d')
end_date = datetime.strptime('2021-10-02', '%Y-%m-%d')
date_list = pd.date_range(start_date, end_date)
df = pd.DataFrame(index=date_list)

#5년 기준 주가 데이터 수정종가
for k in sec:
    try:
        tmp = pd.read_csv(f'{path_dir}/{k}.csv').iloc[:,[0,6]]
        if datetime.strptime(tmp.at[0, 'Date'], '%Y-%m-%d') > datetime.strptime('2011-01-01', '%Y-%m-%d'):
            pass
        else:
            sdate = datetime.strptime(tmp.at[0, 'Date'], '%Y-%m-%d')
            edate = datetime.strptime(tmp.at[len(tmp) - 1, 'Date'], '%Y-%m-%d')
            #날짜인덱스 맞추기
            tmp.rename(columns={'Date': 'Date_fix'}, inplace=True)
            tmp['Date'] = pd.to_datetime(tmp["Date_fix"])
            tmp.set_index('Date', inplace=True)
            tmp.drop('Date_fix', axis=1, inplace=True)
            date_list = pd.date_range(start_date, end_date)
            df['Adj Close'] = tmp['Adj Close']
            df = df.rename(columns={'Adj Close':f'{k}'})
    except:
        pass
df = df.dropna()

#회사명 인덱스, 수익률, 전치행렬 이용
df = df.pct_change().iloc[1:].T
stock_name = list(df.index)
r_a = df.values


#데이터 정규화
normalize = Normalizer()
array_norm = normalize.fit_transform(df)
df_norm = pd.DataFrame(array_norm, columns=df.columns)
result_df = df_norm.set_index(df.index)

#K설정 별 클러스터링
k = 24
clusters = range(2,k)
error = []

for p in clusters:
    clustering = KMeans(p)
    clustering.fit(result_df)
    error.append(clustering.inertia_/100)

df2 = pd.DataFrame({'K_cluster': clusters, "Err_Term":error})

#최적K 체크

plt.figure(figsize=(20,15))
plt.plot(df2.K_cluster, df2.Err_Term, marker='o', color='Blue')
plt.xlabel("Number of K")
plt.ylabel("SSE")
plt.savefig('Best Cluster Finding.png')

##### 사진에서 보면 그냥 완만하게 내려가므로 마땅한 값 찾기가 어려움

#임의의 k할당 (16)
cluster = KMeans(16)
cluster.fit(result_df)
cluster.labels_

labels = cluster.predict(r_a)
final_df = pd.DataFrame({'labels': labels, 'Stock': stock_name})
final_df.sort_values('labels')
final_df.to_csv('Clustering Result.csv', encoding='euc-kr')

#9: 대기업 전자제품, 12: 쇼핑몰, 등등..

# #마지막 실제 섹터값 넣어주기
#
# # for q in range(1,12):
# #     a = globals()[f'sec{q}']
# #     for j in a:
# #         b = str(j)
# #         if final_df['Stock'].str.contain(b)] == True:
# #             final_df['Real Sector'] = q
# #         else:
# #             pass
