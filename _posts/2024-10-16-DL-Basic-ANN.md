---
layout: post
title: 딥러닝 기초와 ANN 모델 
subtitle: TIL Day 24
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, DL]
author: polaris0208
---

## 딥러닝의 개념
**인공신경망(Artificial Neural Networks) 기반 머신러닝**
* 다층 신경망 사용 - 자동으로 학습, 복잡한 문제 해결
* 중요한 패턴 추출, 예측,분류, 생성 등 다양한 작업 수행
> 중요한 패턴 : 작업 수행에 큰 영향을 주는 요소 - 가장 잘 동작하는 특징

1. 딥러닝의 특징
* 비선형 추론 가능
* 다층 구조 - 고차원 특징 학습
* 자동 특징 추출: 별도의 공학 과정이 필요 없음

2. 딥러닝의 역사와 활용 방안
- 역사와 발전
    - 1980s 역전파 알고리즘(Backpropagation) 제안, 다층 신경망 학습 가능
    - 2000s 대규모 데이터셋 등장, 딥러닝 연구 활성화
    - AlexNet, VGGNest, ResNet, 등 다양한 딥러닝 모델 개발, 혁신적인 성과
    <br>
- 인공지능-머신러닝-딥러닝의 관계
  - AI: 인간의 지능을 모방, 문제를 해결하는 기술 / 규칙기반 학습~자율 학습 시스템
  - ML: 데이터를 이용해 모델을 학습하고 예측이나 결정을 내리는 기술
  - DL: 다층신경망 사용, 대규모 데이터 활용 복잡한 문제를 다룸
  <br>
- 최근의 할용 방안
  - 이미지 인식: 이미지 분류, 검출, 생성 - 자율 주행의 도로상황 인식
  - 자연어 처리: 번역, 요약, 감정분석 - 구글 번역기
  - 음성 인식
  - 의료 분야: 종합적

3. 배워야 하는 이유
* 높은 수요
* 혁신적인 성과
* 지속적인 발전
* 실용적인 응용
* **창의적인 가능성; 입력에 대한 제한이 줄어들면서 접근성이 높아짐 ; 다양한 가능성이 열림**

## 신경망의 기본원리
### 퍼셉트론(Perceptron)
- XOR 문제포함
1. 단일 퍼셉트론의 원리
  - 단일 퍼셉트론 개념  
    - Perceptron: 인공 신경망의 가장 기본적인 단위; 하나의 뉴런을 모델링
    - 입력값에 가중치(weight)를 곱하고 이를 모두 더한 후 활성화 함수(activateion function)을 통해 출력 값을 결정

2. 다층 퍼셉트론(MLP) - Multi Layer Perceptron
  - 다층 퍼셉트론의 개념
    - 여러층의 퍼셉트론을 쌓아올린 신경망 구조
  - 레이어의 개념
    - 입력층(input layer): 외부 데이터가 신경망에 입력되는 부분 / 입력 레이어의 뉴런 수 = 입력되는 특징 수
    - 은닉층(hidden layer): 입력 데이터를 처리하고 특징을 추출하는 역할 / 은닉층의 뉴런 수, 층 수는 모델의 복잡성과 층수에 영향
  - XOR 문제와 MLP
    - 단일 퍼셉트론은 선형 분류
    - XOR 문제는 두 입력 값이 다를 때만 1을 출력 - 단일 퍼셉트론으로는 대응 불가
    - MLP로는 비선형성을 해석 가능
### 활성화 함수 
1. 필요성
  - 입력값을 출력값으로 변환
  - 활성화 함수가 없으면 신경만은 단순 선형 변환만 수행, 복잡한 패턴 학습 불가
  - 비선형성 도입 복잡한 패턴 학습 가능
- 단점: 수학적 문제 발생

2. 활성화 함수의 종류
  - ReLU(Rectified Linear Unit) : 0과 x 사이의 max값 반환 계산이 가능
    - 기울기 소실문제 완화 / 죽은 ReLU 문제 발생: 음수 입력에 대해 기울기가 0이 됨
  - Sigmoid: 출력 값이 0과 1 사이로 제한 - 확률 표기에 적합
    - 기울기 소실 문제. 값이 0과 1에 가까워질 때 학습이 느려짐
  - Tanh (Hyperbolic Tangent):출력값 -1, 1사이 - 중심이 0애 가까어짐
    - 기울기 소실 문제

### 손실 함수와 최적화 알고리즘
- 용도에 따라 사용하는 종류가 다르다
1. 손실함수의 역할
  - 예측값과 실제값의 차이를 측정
  - 모델의 성능을 평가, 최적화 알고리즘을 통해 모델의 학습을 수행

2. 손실 함수의 종류
  - MSE: 회귀 문제
    - 에측값과 실제값의 차이를 제곱하여 평균을 구함
  - Cross-Entropy : 분류문제
    - 예측 확률과 실제 클래스 간의 차이를 측정
3. 최적화 알고리즘의 개념과 종류
  - 손실함수가 줄어들도록 가중치를 조절
  - 오차함수를 미분 해서 수렴하는 가중치를 찾음
  - local optimum 문제: 항상 최적값을 찾을 수 있는 것이 아님, 상황에 따라 
  - SGD(Stochastic Gradient Descent) 
    - 전체 데이터셋이 아닌 선택된 일부 데이터 사용하여 기울기 계산. 가중치 없데이트
    - 계산이 빠르고, 효율적
    - 최적점에 도달하기 전에 진동이 발생 가능
  - Adam(Adaptive Moment Estimation)
    - 모멘텀과 RMSProp을 결합한 알고리즘, 학습룰을 적응저으로 조정
    - 빠른 수렴속도, 안정적인 학습
    - 하이퍼 파라미터 설정이 복잡

### 역전파(Backpropagation)
1. 역전파 개념과 수학적 원리
- 역전파 알고리즘의 개념
  - 신경망의 가중치를 학습시키기 위한 알고리즘
  - 출력에서 입력 방향으로 손실함수의 기울기 계산, 가중치 업데이트
- 수학적 원리
  - 연쇄법칙(Chaine Rule) 여러개의 레이어의 기울기 계산하는 법칙
  - 각 층의 기울기는 이전 층의 기울기와 현재 층의 기울기를 곱하여 계산

## 인공 신경망의 기본 구조와 동작원리
### ANN(Artificial Neural Networks) 구성요소
- 생물학적 신경망을 모방하여 설계된 컴퓨팅 시스템
1. ANN의 기본 구성요소
- 입력층, 은닉층, 출력층
- 각 층은 뉴런으로 구성
- 출력층부터 설계: 문제의 종류에 따라 출력이 달라야 함 
- 입력층: 입력 데이터를 은닉층이 받아들이는 형태로 변환하여 전달
- 은닉층: 입력 데이터를 처리하고 특징을 추출 / 연산 파트
2. 동작 방식
- 순전파(Foward Propagation)
  - 입력을 통해 각층 뉴런 활성화
  - 최종 출력 값을 계산
  - 각 뉴런은 입력값에 가중치(weight)를 곱하고, 바이어스(bias)를 더함
  - 활성화 함수(activation)을 통해 출력 값을 결정
- 손실계산(Loss Calculation)
  - 예측값과 실제 값의 차이를 계산
- 역전파(Backpropagation)
  - 손실함수의 기울기를 출력층에서 입력층 발향으로 계산
  - 계산을 바탕으로 가중치 업데이트
  - Parameter: 업데이트 하기 위한 가중치
  - hyperparameter: parameter값을 조정해서 global optimum을 찾아감
- 출력 레이어 선택
  - 획귀문제: 출력 레이어의 뉴런 수는 예측하려는 연속적인 값의 차원과 동일
    - 활성화 함수 = 주로 선형 함수 
  - 이진 분류 문제: 출력 레이어의 뉴런 수는 1
    - 활성화 함수 = sigmoid; 0과 1사이의 확률
  - 다중 클래스 분류 문제: 예측하려는 클래스 수와 동일
    - 활성화 함수 = sofdmax; 각 클래스에 대한 확률


### MNIST 예제
- 모듈 `import`

```py
import torch # 핵심 라이브러리
import torch.nn as nn # neural networks 신경망 구성
import torch.optim as optim # 최적화(함수를 최소 또는 최대로 맞추는 변수를 찾는 것)
import torchvision # 이미지 처리
import torchvision.transforms as transforms # 이미지 처리 전처리
```

- 데이터셋 전처리
  * train = True : 학습용 데이터
  * transform = 생성한 전처리 설정(transform)
  * batch_size : 쪼갠 데이터의 크기
  * suffle : 쪼갠 데이터 섞기 ; 그대로 사용하면 순서의 상관이 작용 할 가능성이 있음

```py
# 데이터셋 전처리
transform = transforms.Compose([
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
```

- 상속으로 SimpleANN 클래스 기능 가져오기

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
```

- 모델 초기화

```py
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
```

- 모델 학습
  * epoch : 한번의 학습 싸이클 ; 적절하게 설정 필요

```py
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
```

- 모델 평가

```py
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