---
layout: post
title: 머신러닝, 딥러닝 요약
subtitle: TIL Day 28
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, ML, DL]
author: polaris0208
---

# 지도학습 모델(supervised)
### 선형회귀(Linear Regression)

```py
import numpy as np # 연산 
import pandas as pd # 데이터 처리
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.linear_model import LinearRegression # 선형 모델의 선형 회귀
from sklearn.metrics import mean_squared_error, r2_score # 성능 평가

X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5],[6,6]]) # 독립변수
y = np.array([1, 2, 3, 4, 5, 6]) # 종속 변수

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# x 는 독립 변수, 테스트 사이즈 20퍼센트, random_state는 시드값, 시드값이 달라지면 결과가 달라짐
# 예시가 42인 이유는 더글러스 애덤스의 은하수를 여행하는 히치하이커를 위한 안내서의 궁극의 해답 42로 추측

model = LinearRegression()
model.fit(X_train, y_train) # 테스트 데이터와 테스트 정답 - 학습 진행

y_pred = model.predict(X_test) # 예측
y_pred # array([1., 2.])
X_test # array([[1, 1], [2, 2]])

mse = mean_squared_error(y_test, y_pred) # 0에 근접할 수록 우수
r2 = r2_score(y_test, y_pred) # 1에 근접할 수록 우수
```

### 다항회귀(Polynomial Features)

```py
from sklearn.preprocessing import PolynomialFeatures # 다항회귀 # 차수선택
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1, 4, 9, 16, 25, 36])

poly = PolynomialFeatures(degree=2) # 2차항까지
X_poly = poly.fit_transform(X) # 데이터 전달
X_poly # 0차항, 1차항, 2차항

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Logistic 회귀

```py
import numpy as np # 연산 
import pandas as pd # 데이터 처리
from sklearn.model_selection import train_test_split # 데이터 분할

from sklearn.datasets import load_breast_cancer # 데이터 셋 가져오기
from sklearn.preprocessing import StandardScaler # 표준화 도구
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
X = data.data 
y = data.target # 레이블 # 이진데이터

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler() # 평균이 0 분산 1
X_train = scaler.fit_transform(X_train) # 평균과 분산 찾아서 정규화
X_test = scaler.transform(X_test) # fit 제거 -테스트 데이터의 평균과 분산을 사용하면 안됨

# 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

### SVM, KNN, Naive Bayes

```py
import numpy as np # 연산 
import pandas as pd # 데이터 처리
import seaborn as sns
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 로드
titanic = sns.load_dataset('titanic')

# 필요한 열 선택 및 결측값 처리
titanic = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].dropna()

# 성별과 탑승한 곳 인코딩
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})
titanic['embarked'] = titanic['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 특성과 타겟 분리
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### SVM 분류모델 (Support Vector Machine)

```py
from sklearn.svm import SVC # 오류 처리 방식에 약간 차이가 있는 svm

# 모델 생성 및 학습
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

### KNN 분류모델(K Nearest Neighbor)

```py
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

### Naive Bayes 분류 모델

```py
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

### Decision Tree 의사결정 나무

```py
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

# 비지도 학습 모델(Unsuperviesd)
* 정답이 없는 문제를 푼다
## 군집화 모델(Clustering)

```py
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler # 표준화


data = pd.read_csv('/Users/사용자이름/myenv/studynote/kaggle/Mall_Customers.csv')
# 거리를 구하는 것에 의미 있는 데이터만 남김(숫자)
data = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
# 데이터 스케일링
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

### K-means clustering

```py
# 클러스터링 작업
from sklearn.cluster import KMeans # kmeans 
# k값을 증가시켜 가면서 inertia 값을 추출

inertia = []
K = range(1, 11)
for k in K:
  kmeans = KMeans(n_clusters=k, random_state = 42) 
  kmeans.fit(data_scaled) 
  inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt # 시각화 도구
import seaborn as sns # 시각화 도구

plt.figure(figsize=(10, 8))
plt.plot(K, inertia, 'bx-') #"bx-" 는 스타일 https://guebin.github.io/DV2023/posts/01wk-2.html
plt.xlabel('k') # x축 이름
plt.ylabel('Inertia') # y축 이름
plt.title('Elbow Method For Optimal k') # 제목
plt.show() # 시각화 후 엘보 포인트 정하기

# 모델 생성
# 엘보 포인트를 k값으로 모델 생성 및 학습
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(data_scaled)

# 군집 결과 할당 # 각 데이터 포인트가 속한 군집 레이블을 반환
data['Cluster'] = kmeans.labels_

# 군집 시각화
# 2차원으로 군집 시각화 (연령 vs 소득)
# hue(색조)=data : 해당 데이터에 따라 색을 다르게 하라
# palette : 색 옵션

plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['Age'], y=data['Annual Income (k$)'], hue=data['Cluster'], palette='viridis')
plt.title('Clusters of customers (Age vs Annual Income)')
plt.show()
```

### Hierarchical Clustering
* 계층적 군집화

```py
import scipy.cluster.hierarchy as sch # 덴드로그램
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(data_scaled, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering # 병합 군집

hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')

# 모델 학습 및 예측
y_hc = hc.fit_predict(data_scaled)

# 결과 시각화

plt.figure(figsize=(10, 7))
plt.scatter(data_scaled[y_hc == 0, 0], data_scaled[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(data_scaled[y_hc == 1, 0], data_scaled[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(data_scaled[y_hc == 2, 0], data_scaled[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(data_scaled[y_hc == 3, 0], data_scaled[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(data_scaled[y_hc == 4, 0], data_scaled[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()

from sklearn.metrics import silhouette_score

# 실루엣 점수 계산 (-1 ~ 1)
silhouette_avg = silhouette_score(data_scaled, y_hc)
print(f'Silhouette Score: {silhouette_avg}')
#
Silhouette Score: 0.39002826186267214
```

### DBSCAN
**Density-Based Spatial Clustering of Applications with Noise**

```py
from sklearn.cluster import DBSCAN

# DBSCAN 모델 생성
dbscan = DBSCAN(eps=5, min_samples=5)

# 모델 학습 및 예측
data['Cluster'] = dbscan.fit_predict(X)

# 군집화 결과 시각화
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis')
plt.title('DBSCAN Clustering of Mall Customers')
plt.show()

eps_values = [3, 5, 7, 10]
min_samples_values = [3, 5, 7, 10]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        data['Cluster'] = dbscan.fit_predict(X)
        
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='viridis')
        plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
        plt.show()
```

## 차원축소 모델(Dimension Reduction)

```py
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_openml

# MNIST 데이터셋 불러오기
mnist = fetch_openml('mnist_784', version=1)

# 데이터와 레이블 분리
X = mnist.data
y = mnist.target

# 데이터 프레임의 첫 5행 출력
print(X.head())
print(y.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### PCA
**Principal Component Analysis, 주성분 분석**

```py
from sklearn.decomposition import PCA

# PCA 모델 생성
pca = PCA(n_components=0.95)  # 전체 분산의 95%를 설명하는 주성분 선택

# PCA 학습 및 변환
X_pca = pca.fit_transform(X_scaled)
```

```py
# 선택된 주성분의 : 332
print(f'선택된 주성분의 수: {pca.n_components_}')

# 각 주성분이 설명하는 분산 비율
print(f'각 주성분이 설명하는 분산 비율: {pca.explained_variance_ratio_}')

# 누적 분산 비율 #cumsum 누적합
print(f'누적 분산 비율: {pca.explained_variance_ratio_.cumsum()}')

# 2차원 시각화 # legend 범례
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', legend=None)
plt.title('PCA of MNIST Dataset (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

### t-SNE
**t-Distributed Stochastic Neighbor Embedding**

```py
# 진짜 오래걸림
from sklearn.manifold import TSNE

# t-SNE 모델 생성
tsne = TSNE(n_components=2, random_state=42)

# t-SNE 학습 및 변환
X_tsne = tsne.fit_transform(X_scaled)

# 변환된 데이터의 크기 확인
print(X_tsne.shape)
#
(70000, 2)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis', legend=None)
plt.title('t-SNE of MNIST Dataset (2D)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

## LDA
**Linear Discriminant Analysis, 선형 판별 분석**

```py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA 모델 생성
lda = LinearDiscriminantAnalysis(n_components=9)  # 클래스의 수 - 1 만큼의 선형 판별 축 선택

# LDA 학습 및 변환
X_lda = lda.fit_transform(X_scaled, y)

# 변환된 데이터의 크기 확인
print(X_lda.shape)
# 
(70000, 9)

# 2차원 시각화
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=y, palette='viridis', legend=None)
plt.title('LDA of MNIST Dataset (2D)')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.show()
```

## 앙상블 학습 **Ensemble Learning**

```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_breast_cancer

# 유방암 데이터 로드
cancer_data = load_breast_cancer()
X, y = cancer_data.data, cancer_data.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

1. 배깅 **Bagging**

```py
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
# 배깅 모델 생성
bagging_model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=100, random_state=42)

# 모델 학습
bagging_model.fit(X_train_scaled, y_train)

# 예측
y_pred_bagging = bagging_model.predict(X_test_scaled)

# 평가
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
print(f'배깅 모델의 MSE: {mse_bagging}')
```

2. 부스팅 **Boosting**

```py
from sklearn.ensemble import GradientBoostingRegressor

# 부스팅 모델 생성
boosting_model = GradientBoostingRegressor(n_estimators=100, random_state=42) # estimator가 없음 / 순차 수행이기 때문에 선택지가 적기 떄문에

# 모델 학습
boosting_model.fit(X_train_scaled, y_train)

# 예측
y_pred_boosting = boosting_model.predict(X_test_scaled)

# 평가
mse_boosting = mean_squared_error(y_test, y_pred_boosting)
print(f'부스팅 모델의 MSE: {mse_boosting}')
```

4. **Random Tree**

```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 랜덤 포레스트 모델 생성
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 모델 학습
rf_model.fit(X_train_scaled, y_train)

# 예측
y_pred_rf = rf_model.predict(X_test_scaled)

# 평가
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'랜덤 포레스트 모델의 MSE: {mse_rf}')
```

4. 그래디언트 부스팅 머신 **Gradient Boosting Machine, GBM**

```py
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# GBM 모델 생성
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # 약한 학습기로 시작하기 위해 depth는 3

# 모델 학습
gbm_model.fit(X_train_scaled, y_train)

# 예측
y_pred_gbm = gbm_model.predict(X_test_scaled)

# 평가
mse_gbm = mean_squared_error(y_test, y_pred_gbm)
print(f'GBM 모델의 MSE: {mse_gbm}')
```

5. **XGBoost(eXtreme Gradient Boosting)**

```py
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# XGBoost 모델 생성
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 모델 학습
xgb_model.fit(X_train_scaled, y_train)

# 예측
y_pred_xgb = xgb_model.predict(X_test_scaled)

# 평가
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f'XGBoost 모델의 MSE: {mse_xgb}')
```

# Deep Learning Model
## MNIST 예제

```py
transform = transforms.Compose([
    # 
    transforms.ToTensor(),
    # 이미지를 Tensor(파이토치 자료구조)로 전환
    transforms.Normalize((0.5,), (0.5,)) 
    # 이미지 정규화(평균, 표준편차)
    ])

# MNIST 데이터셋 로드
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 경로 # train(학습용) 데이터셋 여부 # 다운로드 여부 #transform 전달-전처리한 상태로
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# 토치.유틸.데이터 기능.데이터 로더(데이터 셋, batch_size(쪼갠단위로 학습), suffle(섞어서 쪼갬))

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

for (X_train, Y_train) in trainloader:
    print(f"X_train: {X_train.size()} type: {X_train.type()}")
    print(f"Y_train: {Y_train.size()} type: {Y_train.type()}")
    break
# batch_size, channel, width, height
# X_train: torch.Size([64, 1, 28, 28]) type: torch.FloatTensor
# Y_train: torch.Size([64]) type: torch.LongTensor
```

### ANN

```py
class SimpleANN(nn.Module): # 상속으로 기능 가져오기
    # init 설정
    def __init__(self):
        # 부모 클래스의 init 가져오기
        super(SimpleANN, self).__init__() 
        # fc : fully connected module ; 서로가 서로에게 연결된 레이어 
        self.fc1 = nn.Linear(28 * 28, 128)  # 입력층에서 은닉층으로
        # nn.Linear: ANN모델 생성 함수
        # 입출력 지정
        # 28 * 28 데이터셋의 크기 / 10 0~9 10개로 출력
        self.fc2 = nn.Linear(128, 64)       # 은닉층에서 은닉층으로
        self.fc3 = nn.Linear(64, 10)        # 은닉층에서 출력층으로

    def forward(self, x): # 레이어간 전달
        x = x.view(-1, 28 * 28)  # 입력 이미지를 1차원 벡터로 변환
        # view 함수 -1 
        # 전체 요소 개수에서 28*28 을 제외한 성분의 수 
        # 예시) 전체 16 (-1, 4) # (4,4)로 생성
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # 최종출력 레이어는 relu 처리가 필요 없음
        return x # 결과는 (64개 데이터 * 10차원)

# 모델 초기화
model = SimpleANN()

correct = 0
total = 0
with torch.no_grad():
    # 평가 단계에서는 기울기 계산 필요가 없음, 생략
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # 10개의 레이브은 각각의 가능성, 각 레이블에서 가능성이 가장 큰것만 추출
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%') 
# 학습 전 = 9.35%

criterion = nn.CrossEntropyLoss() # 분류모델 손실함수 계산
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 최적화, lr- 학습률(적당히 작은값), momentum - 치후 설명

# 모델 학습
for epoch in range(10):  # 10 에포크 동안 학습 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # i = index, 
        inputs, labels = data

        # 기울기 초기화 - 연쇄법칙 수행으로 남아있는 로그 제거
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 업데이트를 진행할 방향, 기울기를 찾는 과정
        optimizer.step() 
        # 기울기를 바탕으로 가중치를 업데이트

        # 손실 출력
        running_loss += loss.item()
        if i % 100 == 99:  # 매 100 미니배치마다 출력
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

#모델평가
correct = 0
total = 0
with torch.no_grad():
    # 평가 단계에서는 기울기 계산 필요가 없음, 생략
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # 10개의 레이브은 각각의 가능성, 각 레이블에서 가능성이 가장 큰것만 추출
        total += labels.size(0)
        # 배치 크기 
        correct += (predicted == labels).sum().item()
        # 에측값과 실제 값이 일치하는 샘플의 수를 계산

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
# 97.61%
```

### KNN

```py
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        # 입력 채널(1: 흑백, 3: RGB), 출력 채널(입력채널과 별도), 커널(필터) 크기 3x3
        self.pool = nn.MaxPool2d(2, 2)               
        # 풀링 크기 2x2 # kernel, stride
        # Pooling 크기가 (2, 2) 라면 출력 데이터 크기는 입력 데이터의 행과 열 크기를 2로 나눈 몫
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        # 입력 채널(이전 출력 채널과 동일) 32, 출력 채널 64, 커널 크기 3x3
        self.fc1 = nn.Linear(64 * 7 * 7, 512)        # 완전 연결 층
        self.fc2 = nn.Linear(512, 10)                # 출력 층 (10개의 클래스)
        # ANN 레이어 여러개 사용 가능

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 플래튼 # ANN에 전달하기 위해 변환
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화, 학습
model = SimpleCNN()

# 손실 함수와 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 최적화, lr- 학습률(적당히 작은값), momentum - 치후 설명

# 모델 학습
for epoch in range(10):  # 10 에포크 동안 학습
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # 기울기 초기화
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # 업데이트를 진행할 방향, 기울기를 찾는 과정
        optimizer.step()
        # 기울기를 바탕으로 가중치를 업데이트
        
        # 손실 출력
        running_loss += loss.item()
        if i % 100 == 99:  # 매 100 미니배치마다 출력
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# 모델 평가
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
# Accuracy of the network on the 10000 test images: 99.27%
```

## sine data

```py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# 시각화 도구

# sine 데이터 생성
# Sine 함수 데이터 생성
def create_sine_wave_data(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand()
        x = np.linspace(start, start + 2 * np.pi, seq_length)
        X.append(np.sin(x))
        y.append(np.sin(x + 0.1))
    return np.array(X), np.array(y)

seq_length = 50
num_samples = 1000
X, y = create_sine_wave_data(seq_length, num_samples)

# 데이터셋을 PyTorch 텐서로 변환 / tensor = 기울기를 계산 할 수 있는 데이터 형태
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
```

### RNN 모델 설정

```py
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) # 출력 레이어이기 때문에 output_size를 정해줌

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)  # 초기 은닉 상태/제일 처음의 값은 은닉 상태가 없기 떄문에 지정해중
        out, _ = self.rnn(x, h0)
        out = self.fc(out) # 전체
        # out = self.fc(out[:, -1, :])  # 마지막 시간 단계의 출력
        return out

input_size = 1
hidden_size = 32
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)
```

### LSTM 모델 설정

```py
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)  # 초기 은닉 상태
        c0 = torch.zeros(1, x.size(0), hidden_size)  # 초기 셀 상태
        # RNN 모델 에 초기 셀 상태 추가
        out, _ = self.lstm(x, (h0, c0)) # 셀 상태 추가
        out = self.fc(out[:, -1, :])  # 마지막 시간 단계의 출력
        return out

model = SimpleLSTM(input_size, hidden_size, output_size)
```

### 모델 초기화, 학습

```py
# 손실 함수와 최적화 알고리즘 정의
criterion = nn.MSELoss()
# 평균 제곱 오차 손실함수
optimizer = optim.Adam(model.parameters(), lr=0.01)
# ADAM 최적화 방식

# 모델 학습
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    optimizer.zero_grad() 
    # 이전 단계에서 계산된 기울기 초기화
    loss = criterion(outputs, y)
    loss.backward()
    # 역전파를 통한 기울기 계산
    optimizer.step()
    # 가중치 업데이트

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')

# 모델 평가
model.eval()
with torch.no_grad():
    predicted = model(X).detach().numpy()

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(y.numpy().flatten()[:100], label='True')
plt.plot(predicted.flatten()[:100], label='Predicted')
plt.legend()
plt.show()
```