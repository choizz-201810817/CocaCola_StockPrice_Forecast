#%%
import pandas as pd
import numpy as np

import pymysql
from sqlalchemy import create_engine

from tensorflow.python.client import device_lib
import tensorflow as tf

# 딥러닝 디폴트 임포트
from keras import models, Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import os
import warnings
warnings.filterwarnings('ignore')

# %%
# db에 data 저장
cokeDf = pd.read_csv(r'data\stock.csv')

def df2tbl(id='', pswd='', host='', dbName='', tblName=''):
    dbPath = f'mysql+pymysql://{id}:{pswd}@{host}/{dbName}'
    dbConn = create_engine(dbPath)
    cokeDf.to_sql(name='cokeTbl',con=dbConn, if_exists='fail', index=False)

#%%
# db에서 데이터 불러오기
conn = pymysql.connect(host='localhost', user='root', passwd='949700', db='cokestockdb', charset='utf8')
df = pd.read_sql('SELECT * FROM cokeTbl', con=conn)

# %%
### 결측치 처리
### 파생변수 생성
### 데이터 스케일링
### eda
### 모델 정의 및 학습
### 예측
### 결과

### 참고 : 시계열 데이터는 시간을 index로 빼야함
### open 시가, high 고가, low 저가, close , volume 거래량, dividends 배당, stock splits 주식분할
### close와 adj close가 있다면 adj close를 사용
### matplotlib에서 index에 들어가는 것은 무조건 x축 값(set index로 index로 넘어간 경우만 해당)
### 시계열은 plotly를 주로 사용
### 주식 그래프는 lineplot 사용
#%%
# 결측치 확인 및 info 확인 및 shape확인
print("="*20, ' 결측치 확인 ', "="*20)
print(df.isna().sum())
print("="*20, ' info 확인 ', "="*20)
print(df.info())
print("="*20, ' shape 확인 ', "="*20)
print("coke data shape:", df.shape)

# Date컬럼 datetype으로 변경 및 컬럼명 소문자로 변경
df['Date'] = pd.to_datetime(df['Date'])
df.columns = df.columns.str.lower()

# date 컬럼을 인덱스로 설정
df = df.set_index('date')

# %%
# 시계열 데이터 확인
plt.figure(figsize=(12,5))
df.plot()

df.plot(subplots=True)

# stock splits의 value_counts를 통해 주가 분할이 얼마나 이루어 졌는지 확인
df['stock splits'].value_counts()

# %%
fig = make_subplots(rows=7, cols=1)

for i, col in enumerate(df.columns):
    fig.add_trace(go.Line(x=df.index, y=df[col], name =  f'{col} Graph'), row=i+1, col=1)
    fig.update_layout(autosize=False, width=700, height=1800, title=dict(text="CoCaCola Stock"))
    fig.show()

#%%
fig = go.Figure()
fig.add_trace(go.Line(x=df.index, y=df.close))
fig.show()

# %%
# 주가 30일치만 확인
fig = go.Figure(data=[go.Candlestick(x=df[-30:].index, open=df.open, high=df.high, low= df.low, close=df.close)])
fig

# %%
# 그래프 추이를 알기 위해 이동평균 기법을 진행
# 이동 평균이란? 평균을 이동시킴
# ex) 30일의 평균을 내겠다.. 정하고 이동평균을 진행하면 꼬불꼬불한 그래프가 펴져서 그래프의 추이를 확인할 수 있게 됨
# 1~30일의 평균, 2~31일의 평균, 3~32일의 평균... 이런식
# 여기서는 300, 500, 700, 900일의 이동 평균을 진행
# 롤링이라는 함수 사용

df1 = pd.DataFrame(df.close)
for days in [3, 300, 500, 700, 900]:
    df1[f'mm_{days}days'] = df1.close.rolling(days).mean()


# 이동평균 단위가 가장 큰 900에 맞춰 앞부분 자르기(null값 제거)
mmDf = df1.iloc[899:]

fig = go.Figure()
for col in mmDf.columns:
    fig.add_trace(go.Line(x=mmDf.index, y=mmDf[col]))
fig.update_layout
fig.show()


# %%
# 일일 수익률(주가 상승량)이 어느 정도인가?
# 리스크가 얼마나 되는가?
# 일일 수익률이 고르면 안전한 주가, 왔다갔다 한다면 위험성이 큰 주가
# 표준편차가 크면 왔다갔다 많이 한다는 뜻이므로 위험성 큼
# 표준편차가 작으면 폭이 작다는 의미이므로 비교적 안정적

# 일일 수익률 -> pct_change활용
mmDf['pct_change'] = mmDf.close.pct_change()
mmDf.head()

fig = go.Figure()
fig.add_trace(go.Line(x=mmDf.index, y=mmDf['pct_change']))
fig.update_layout(title=dict(text="CoCaCola Close Stock's pct_change"))
fig.show()

# pct_change의 분포확인
# 히스토그램 : 해당 값의 특정 구간의 빈도수를 나타내는 것
# 히스토그램 그릴 때는 x값만 필요
fig = go.Figure()
fig.add_trace(go.Histogram(x=mmDf['pct_change']))
fig.update_layout(title=dict(text="CoCaCola Close Stock's pct_change's hist"))
fig.show()

# %%
## 리스크 확인
# 리스크는 pct_change의 표준편차를 의미
# 그래프에서 y값이 y축에 가까울 수록 안전하고 영점에서 떨어져 있을 수록 위험하다는 뜻
fig = go.Figure()
fig.add_trace(go.Scatter(x=[mmDf['pct_change'].mean()], y=[mmDf['pct_change'].std()]))
fig.update_layout(title=dict(text="CoCaCola's Risk"))
fig.show()


# %%
df2 = pd.DataFrame(df1.close)

mmSc = MinMaxScaler()
df2.close = mmSc.fit_transform(df2)
print(df2.shape)


# %%
# 85%의 데이터로 학습시키고 나머지  15%의 데이터로 모델을 test함
# lstm의 경우 t-1시점이 t시점에 영향을 주는 모델
# 따라서 X 데이터 셋을 1일~60일, 2일~61일, 3일~62일.. 이런 식으로 맞물리게 만들어야함
# y 데이터 셋은 한 단위 날짜의 마지막 날의 다음 날 값을 사용 (즉, 61일, 62일, 63일... 이 타겟값)
# 주식에서는 3개월을 한 단위로 보는데 그 중 주말 제외 한 달을 20일로 보고 총 60일 단위로 끊음

trainSet = df2[:int(len(df2)*.85)+1].close.values
testSet = df2[int(len(df2)*.85)+1-60:].close.values

# print("trainSet's shape :", trainSet.shape)
# print("testSet's shape :", testSet.shape)

def trainTestSplit(trainSet=np.array([]), testSet=np.array([])):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(trainSet)-60):
        X_train.append(trainSet[i:i+60])
        y_train.append(trainSet[i+60])

    for i in range(len(testSet)-60):
        X_test.append(testSet[i:i+60])
        y_test.append(testSet[i+60])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

X_train, X_test, y_train, y_test = trainTestSplit(trainSet, testSet)

print("X_train's data shape:", X_train.shape)
print("y_train의 data 수:", y_train.shape)
print("X_test의 data 수:", X_test.shape)
print("y_test의 data 수:", y_test.shape)

#%%
print(device_lib.list_local_devices())
tf.config.list_physical_devices('GPU')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# %%
## LSTM의 input shape : 첫 번째는 batch_size, 두 번째 인자는 시점, 세 번째는 feature 수..
## batch_size는 한 번에 학습하는 데이터의 수..
# batch_size를 키우면 한 번에 여러개씩 학습하며, 이를 몇 개로 해야 최적의 값을 찾을 수 있는지 모르기 때문에 적지 않음.(None으로 자동적으로 들어감)
## 여기서 시점이란? 과거 몇 개의 시간(시간단위(월, 일, 시간 등등..))으로 하나를 예측하느냐
## 우리의 input train data는 (60,)의 데이터가 총 12794개가 있음.. 이를 (12794,60,1)로 만들어 줘야함

X_train = X_train.reshape(12794,60,1)
print('input data의 shape :', X_train.shape)

#%%
# lstm 특성상 여러개의 입력을 받고 각 노드들이 다음 노드에 영향을 주면서 옆으로 밀려 하나의 값을 출력하는데,
# 다음 lstm을 쌓는 경우 여러개의 입력을 받을 준비를 하고 있는데 하나면 받을 수 없음..
# 따라서 첫 lstm의 각 노드마다 출력값이 나올 수 있게 return_sequences를 True로 설정..

HIDDENUNITS = 256

model = Sequential()
model.add(LSTM(HIDDENUNITS, input_shape=(60,1), return_sequences=True))
model.add(LSTM(int(HIDDENUNITS/2), activation='elu', return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='Adam', loss='mean_squared_error')
model.summary()
model.fit(X_train, y_train)
history = model.fit(X_train, y_train, epochs=10, verbose=1)

# %%
sns.lineplot(history.history['loss'])
# %%
pred = model.predict(X_test)
invPred = mmSc.inverse_transform(pred)
invYtest = mmSc.inverse_transform(y_test.reshape(2268,-1))
#%%
predDf = pd.DataFrame({'close':invYtest.reshape(2268,), 'pred': invPred.reshape(2268,)})
# %%
predDf.index = df1.index[-2268:]
predDf

#%%
predDf = predDf.astype('float32')
plt.figure()
predDf.plot()

# %%
from sklearn.metrics import mean_squared_error
mean_squared_error(invYtest, invPred)
# %%
