---
layout: post
title: CNN과 RNN 모델, NLP 기본 개념
subtitle: TIL Day 25
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, DL]
author: polaris0208
---

## CNN 
>합성곱 신경망(Convolutional neural network)
>이미지와 같은 2차원 데이터의 특징을 효과적으로 추출

### 개요
- 풀링 층(Pooling)
- 완전 연경층(Fully Connected Layer)
  - CNN은 출력에 적합하지 않음, ANN으로 출력

#### ANN 과의 차이점
- 이미지 데이터를 ANN에 입력할 경우 데이터가 거대해짐 
  - 예시) 1000 x 1000 이미지 -> 1 x 1000000 데이터
- 이미지의 경우 픽셀들이 연관성을 가지고 있어서 변환되면 연관이 깨짐
  - 예시) 강아지의 귀 부분을 구성하는 여러 픽셀

### 합성곱 연산의 원리와 필터의 역할
1. 합성곱 연산
- 필터와 이미지의 픽셀을 곱하여 특징 맵 생성
- 필터는 각각 edge, corner, texture 등 다양한 국소적 패턴을 학습

2. Pooling Layer
- 특징 맵의 크기를 줄이고 중요한 특징을 추출
- Max Pooling : 필터 크기 내에서 최대 값을 선택
  - 중요한 특징을 강조, 불필요한 정보 제거
- Average Pooling : 평균 값
  - 특징 맵의 크기를 줄임, 정보의 손실 최소화

3. stride : 필터의 움직임
4. padding
- CNN 처리를 거듭했을 때 이미지가 너무 작아지는 경우 대비
- 다양한 값을 넣어 레이어가 깊어졌을 때 이미지 크기가 너무 작아지는 것 방지

5. CNN 아키텍쳐
- LeNet : 최초, 손글씨 숫자 인식 / 합성곱 - 풀링 - 완전연결
- AlexNet : ReLU 활성화 함수, dropout 도입하여 성능향상
- VGG : 깊고 규칙적인 구조, 3x3 필터 사용하여 깊이 증가 ,VGG16, VGG19

### MNIST 데이터 예제
#### 모듈 `import`

```py
import torch # 핵심 라이브러리
import torch.nn as nn # neural networks 신경망 구성
import torch.optim as optim # 최적화(함수를 최소 또는 최대로 맞추는 변수를 찾는 것)
import torchvision # 이미지 처리
import torchvision.transforms as transforms # 이미지 처리 전처리
```

#### 데이터셋 전처리

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

#### 모델 설정

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
```

#### 모델 초기화, 학습

```py
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
```

#### 모델 평가

```py
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

-------------
## RNN
> 순환 신경망(Recurrent Neural Network) 
> 시계열 데이터나 순차적인 데이터를 처리 = 시퀀스 데이터

### 기본 구조와 동작 원리
1. 기본 구조와 작동 방식
#### 기본 구조
- 이전 시간 단계의 정보를 현재 시간 단계로 전달
- 시퀀스 데이터의 패턴을 학습

``` 
내 이름은 -> 코난 -> 탐정이죠 

다른 모델: [내 이름은] -> 분석
          [코난] -> 분석
            [탐정이죠] -> 분석

RNN : [내 이름은] -> 분석
        [(내 이름은) 코난] -> 분석
          [(내 이름은 코난)탐정이죠] - > 분석
# () = 은닉 데이터
# 모든 시점의 데이터에서 분석
# 역전파 - 모든 시점의 가중치를 업데이트
```

#### 동작 원리
- 순환구조
  - 입력 데이터 + 이전 시간 단계의 은닉 데이터
  - 현재 시간 단계의 은닉 상태 출력
  - 시퀀스 정보를 저장, 다음 시간 단계로 전달
- 동작 원리
  - 각 시간 단계에서 동일한 가중치 공유
  - 시퀀스 학습
  - 순전파(Foward Propagation)
  - 역전파(BPTT): Backpropagation Through Time
    - 기존 역전파: Backpropagation

### LSTM & GRU
- gate를 이용해 정보를 체계적으로 관리
#### 장기 의존성 문제(long-term dependency problem)
  - 시퀀스 데이터는 특성상 데이터가 방대
  - 학습 과정에서 소실되는 데이터 발생
  - 방대한 데이터의 학습 과정에서 기울기 소실 문제 발생

#### LSTM(Long Short-Term Memory)
- cell state: 각 시점 정보의 흐름 저장, 장기적으로 유지
- gate: 정보를 선택적으로 저장하거나 삭제
- input gate: 새로운 정보와 이전 정보가 결합해서 cell 얼마나 반영될지 경정
- reset gate: cell에서 지울 정보 선택
- output gate: cell + 입력 데이터 = 출력

#### GRU(Gated Recurrent Unit)
- hidden state 만 활용(cell과 hidden state 결함)
- reset gate: 이전 정보를 얼마나 무시할지
- update gate: 새로운 정보를 얼마나 반영할지

#### 차이점
- LSTM 은 cell 과 hidden state 모두 사용, 복잡한 gate
- GRU는 hidden state만 사용, 간단한 gate 구조
- GRU가 계산 비용이 적고, 학습이 빠를 수 있음

#### 유의점
- RNN은 역전파를 통해 가중치를 업데이트
- LSTM과 GRU의 경우 gate도 가중치로 역할, 학습에 포함

### RNN 시계열 데이터 처리
- 주식 가격 예측, 날씨 예측, 텍스트 생성
1. 데이터 전처리
- 데이터를 적절한 형태로 변환
- 정규화(nomalization)
  - 데이터의 성격에 따라 방법이 다름
- 입력 시퀀스와 출력 시퀀스 정의
2. 모델 구축
- 모델 정의 
- 입력 크기, 은닉 상태 크기, 출력 크기 등 설정
3. 모델 학습
- 손실 함수, 최적화 알고리즘 정의
- 순전파와 역전파를 통해 모델을 학습
4. 모델 평가
- 테스트 데이터를 사용하여 모델의 성능을 평가

### sine 데이터 예제
#### 모듈 `import`

```py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# 시각화 도구
```

#### sine 데이터 생성

```py
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

#### RNN 모델 설정

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

#### LSTM 모델 설정

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

#### 모델 초기화, 학습

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
```
 
#### 모델 평가

```py
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

![rnn_practice](https://github.com/user-attachments/assets/2538cf42-a6cc-40d3-b4e5-204ee8cd7a45)

## Attention
> 시퀀스 데이터에서 중요한 부분에 더 많은 가중치 할당
> 정보를 더 효율적으로 처리하는 기법

### 개념
1. 기본 구성 요소와 동작 방식
#### Attention 메커니즘
- 자연어 처리(NLP), 시계열 데이터, 기계 번역, 오약, 질의 응답
#### 동작 방식
1. 구성
- 답(key)-질문(query)-최종요약(value) 
- 입력의 sequence를 분석
- 중요도를 파악
- 가중치 부여(일관되게 부여)
2. Attention 스코어(중요도)
- query 와 key 간의 유사도를 측정
- 백터의 내적(dot product) 사용
- softmax = attention스코어를 확률 분포로 변환 = 가중치의 합을  1로
- attention = 가중치 x value
3. sel - Multi-head attention
- self : 시퀀스 내의 각 요소
- multi-head : 여러개의 self attentiondmf 병렬로 수행
  - 모델이 다양한 관점에서 데이터를 처리

### 예시 
> the cat sat on the mat because it was tired

```teacher, "what is 'it'```
```most of students, "cat"```
```the others, "mat"```
- 선생님이 질문한 "it" 은 query
- 대부분의 학생이 대답한 "cat" 은 key 
- 일부 학생이 대답한 "mat"는 value
1. it, cat, mat의 유사도를 확인
2. 가중치를 계산
3. 각 단어의 관계를 확인하여 문장을 더 잘 이해하게 됨

-----------

## NLP
> 자연어 처리 모델
### 워드 임베딩(Word Embedding)
- 단어를 고정된 크기의 백터(숫자)로 변환
- 유사한 단어들은 유사한 숫자로 변환
- Word2Vec, GloVe
1. Word2Vec
- CBOW: 주변 단어(context)를 보고 중심 단어(target)를 예측
- Skip-gram: context를 보고 target 예측
2. GloVe(Global vectors for Word Representation)
- 단어-단어 공기 행렬(word-word co-occurrence matrix)
- 전역적인 통계 정보를 통해 단어 간의 의미적 유사성을 반영

### 시퀀스 모델링(Sequence Modeling)
- 순차적인 데이터 처리
- RNN, LSTM, GRU

### Trnasformer & BERT
1. transformer
- 순차적인 데이터 병렬 처리
- 자연어 처리에 뛰어난 성능
- Encoder-Decoder 구조
  - Encoder: 입력 시퀀스 처리, 인코딩된 표현 생
    - self-attention: 문장 내 관계 학습 
    - Feed-Foward Neural Network -> 새로운 백터로 변환
  - Decoder
    - 시작 토큰 입력
    - taget 단어를 고정된 백터로 변환
    - positional encoding 수행
    - masked multi-head attenrion self attention
      - 이전의 단어들로만 예측하도록 마스킹하여 self attention 수헹
    - 인코더-디코더 attention: 디코더가 인코더 연결- 입력 문자 참조
    - FFNN를 통해 추가로 백터 변환
  - 위 과정을 반복하여 오차를 줄여나감
  - 종료 토큰이 예측되면 번역 종료

2. BERT
(Bidirectional Encoder Representations from Transformers)
- Transformer인코더 기반 사전 학습된 모델
- 양방향으로 문맥을 이해
- 다양한 자연어 처리 가능
- 사전학습(Pre-Training)
  - 대규모 텍스트 코퍼스
  - Masked Language Model과 Next Sentence Prediction 작업
- Fine-tuning
  - 사전 학습된 BERT를 파인 튜닝하여 사용
  - 텍스트의 분류, 질의 응답, 텍스트 생성 등 다양한 자연어 처리 작업
