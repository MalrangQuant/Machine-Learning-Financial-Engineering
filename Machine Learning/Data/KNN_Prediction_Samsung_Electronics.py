##KNN을 통해
##주식의 [금일 시가-종가 차이, 고가-저가 차이], 시장의 [금일 시가-종가 차이, 고가-저가 차이]
##4 정보를 이용해 다음날 주가상승 여부를 바탕으로 KNN수익률 예측모형 만들어보기
##일치 비율과, 샤프비율을 이용해 모델 평가
##결과적으로 수익이 나지는 못함... 다른 종목이나 변수를 좀 더 등락에 연관성 있는 변수로 바꾸면 수정가능성 있어보임


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
import mglearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from datetime import datetime
from sklearn.model_selection import train_test_split
from itertools import compress

asset = pd.read_csv('../KNN/KOSPI Data/삼성전자.csv')

#날짜 맞추고 인덱싱
start_date = datetime.strptime(asset.at[0, 'Date'], '%Y-%m-%d')
end_date = datetime.strptime(asset.at[len(asset)-1, 'Date'], '%Y-%m-%d')
asset.rename(columns = {'Date':'Date_fix'}, inplace=True)
asset['Date'] = pd.to_datetime(asset["Date_fix"])
asset.set_index('Date', inplace=True)
asset.drop('Date_fix', axis=1, inplace=True)

date_list = pd.date_range(start_date, end_date)
df = pd.DataFrame(index=date_list)
df[['Open','High','Low','Close']] = asset[['Open','High','Low','Close']]
df = df.dropna()


#데이터 전처리 + 금일대비 다음날 주가 상승여부 체크
df2 = pd.DataFrame(index=date_list)
df2['Open - Close'] = df['Open'] - df['Close']
df2['High - Low'] = df['High'] - df['Low']
df2 = df2.dropna()


## 다음날 주가가 올랐는지 아닌지 1로 체크
up_down_tmp = []
for i in range(len(df2)-1):
    a = df.iloc[(i+1),3]
    b = df.iloc[(i+1),0]
    if ((a-b) > 0):
        c = 1
    elif (a==b):
        c = 0
    else:
        c = -1
    up_down_tmp.append(c)
up_down_tmp.append(np.nan) #예측치 없으므로 끝자리 채워주기
df2 = df2.assign(Up_Down=up_down_tmp)
df2 = df2.dropna()


#베타값 가져오기
df3 = pd.read_csv('KOSPI_Beta_df.csv', encoding='euc-kr')
df2_beta = df3[df3['Stock'].str.contains('삼성전자')].iloc[0,2]
df2['Beta'] = df2_beta


#날짜 인덱스 맞추기
index = pd.read_csv('KOSPI Index.csv')
start_date = datetime.strptime(index.at[0, 'Date'], '%Y-%m-%d')
end_date = datetime.strptime(index.at[len(index)-1, 'Date'], '%Y-%m-%d')
index.rename(columns = {'Date':'Date_fix'}, inplace=True)
index['Date'] = pd.to_datetime(index["Date_fix"])
index.set_index('Date', inplace=True)
index.drop('Date_fix', axis=1, inplace=True)
date_list = pd.date_range(start_date, end_date)

#인덱스 등락여부
df4 = pd.DataFrame(index=date_list)
df4['Mkt_Open-Close'] = index['Open'] - index['Close']
df4['Mkt_High-Low'] = index['High'] - index['Low']
df4[['Open','High','Low','Close']] = index[['Open','High','Low','Close']]
df4 = df4.dropna()

up_down_tmp2 = []
for i in range(len(df4)):
    a = (df4.iat[i,3] - df4.iat[i,0])
    if (a > 0):
        c = 1
    elif (a == 0):
        c = 0
    else:
        c = -1
    up_down_tmp2.append(c)

df4 = df4.assign(Up_Down=up_down_tmp2)
df4 = df4.dropna()
df2[['Index_Up_Down', 'Mkt_Open-Close', 'Mkt_High-Low']] = df4[['Up_Down', 'Mkt_Open-Close', 'Mkt_High-Low']]

#df2로 모으기
df2[['Open','High','Low','Close']] = df[['Open','High','Low','Close']]


#결측치 제거
df2 = df2.dropna()


#변수 나누고 KNN 적용
X = df2[['Open - Close', 'High - Low','Mkt_Open-Close', 'Mkt_High-Low',]] #'Index_Up_Down']]
Y = df2['Up_Down']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=777, stratify=Y)
preprocessor = preprocessing.Normalizer()
X_train_kr = preprocessor.fit_transform(X_train)
X_test_kr = preprocessor.transform(X_test)


#최적의 k 찾기
training_accuracy = []
test_accuracy = []
k_settings = range(1, 40)

for k in k_settings:
    ploan_knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    ploan_knn.fit(X_train, Y_train)
    training_accuracy.append(ploan_knn.score(X_train, Y_train))
    test_accuracy.append(ploan_knn.score(X_test, Y_test))

plt.plot(k_settings, training_accuracy, label="Training Accuracy")
plt.plot(k_settings, test_accuracy, label="Testing Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("# of Neighbors")
plt.legend()
plt.show()

best_k_temp = test_accuracy==max(test_accuracy)
best_k = list(compress(k_settings, best_k_temp))[0]
print('최적의 K값:', best_k)

#최적 값에서 모델의 Y예측

best_ploan_knn = neighbors.KNeighborsClassifier(n_neighbors=best_k)
best_ploan_knn.fit(X_train, Y_train)
best_ploan_knn.predict(X_test)

n_test = len(Y_test)
Y_predict = best_ploan_knn.predict(X_test)

print('테스트 데이터 개수:', n_test)
print('예측과 일치한 데이터 개수:', sum(Y_test == Y_predict))
print('일치 비율:',round(sum(Y_test == Y_predict)/n_test*100,2),'%')

#도표로 수익률 검증해보기
train_pct = 0.7
split = int(train_pct*len(df2))

df2['Predicted Signal'] = best_ploan_knn.predict(X)

df2['asset_ret'] = np.log(df2['Close'] / df2['Close'].shift(1))
cum_df_ret = df2[split:]['asset_ret'].cumsum() * 100

df2['st_ret'] = df2['asset_ret'] * df2['Predicted Signal'].shift(1)
cum_st_ret = df2[split:]['st_ret'].cumsum() * 100

plt.figure(figsize=(10,5))
plt.plot(cum_df_ret, color='r', label='asset_ret')
plt.plot(cum_st_ret, color='g', label='st_ret')
plt.legend()
plt.savefig('Return Compare on KNN modeling_Samsung Electronics.png')
plt.show()

#수익검사
std = cum_st_ret.std()
sharpe = (cum_st_ret - cum_df_ret) / std
sharpe = sharpe.mean()
print('sharpe ratio : %.2f' % sharpe)

#승률이 낮아 수익이 안나옴.
#일치비율이 상당히 낮게 나옴 (예측력이 떨어짐...?, 손해봄...)
#시장의 등락 정보를 포함해도 큰 변화는 없음.
#활용가능한 변수들 고민해보기.