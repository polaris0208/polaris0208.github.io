---
layout: post
title: Pytorch 기본
subtitle: TIL Day 26
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, DL]
author: polaris0208
---

## **PyTorch** 기본
> 딥러닝 프레임워크
> 유연성과 사용 편의성

### 기본 모델 구축 
`torch.nn.Module`: 모든 신경망 모델의 기본 클래스
- `init`: 구조 만들기
- 부모 클래스의 `init`(초기화 함수)를 가져옴

```py
import torch.nn as nn

class MyModel(nn.Nodule):
  def __init__(self):
    super(MyModel, self).__init__() 
    
    self.layer1 = nn.Linear(10,20) # 입력, 출력
```

- `foward` 데이터의 방향 결정

```py
  def foward(self, x):
    x = self.layer1(x)
    return x
```

### **Custum Dataset** 생성
`torch.utils.data.Dataset`: 사용자 정의 데이터셋 생성
- `torch`의 유틸리티 중, 데이터 기능의, `Dataset` 모듈
- 데이터 불러오기: **학습용, 테스트용** 나눠서 불러옴

```py
from torch.utils.data import Dataset

class MyDataset(Dataset):
  def __init__ (self, data, targets):
    self.data = data
    self.targets = targets
```

- 샘플의 개수 반환: 데이터의 개수를 알려줌

```py
def __len__(self): 
  return len(self.data)
```

- `idx`: 인덱스에 해당하는 샘플을 반환

```py
def __getitem__(self, idx):
  return self.data[idx], self.targets[idx]
```

### **DataLoader**
`torch.utils.data.DataLoader`
-  **mini-batch**학습을 위한 데이터 변환 모듈

```py
from torch.utils.data import DataLoader

dataset = MyDataset(data, targets)
dataloader = Dataloader(dataset, batch-size = 32, shuffle = True)
```

- `Dataset`, `DataLoader` 매개변수
- **MNIST** 예시

```py
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, drop_last = False, timeout = 1)
```

- `Dataset`
  - `root="경로"`: 데이터 경로
  - `download=True`: 경로에 없으면 다운로드
  - `trian = True`: 학습용 데이터, 테스트용인 경우는 `False`
  - `tansform` 데이터 변형, 변형방식은 사용자 정의
- `DataLoader`
  - `trainset`: 변환 할 데이터셋
  - `batch-size`: 배치 하나당 포함 될 샘플의 수
  - `shuffle`: 데이터 순서 무작위 여부
    - 순서가 상관 관계를 가지는 경우 무작위
  - `drop-last`: 배치 사이즈로 나눈 나머지 처리 여부
  - `timeout`: 시간 제한

### 데이터 변환
`torchvision.transforms`: 이미지 데이터 변환 유틸리티
`transforms.ToTensor()`: 이미지를 Tensor(파이토치 자료구조)로 전환
`transforms.Normalize(()`: 이미지 정규화(평균, 표준편차)

```py
transform = transforms.Compose([
    transforms.ToTensor(),
    # 이미지를 Tensor(파이토치 자료구조)로 전환
    transforms.Normalize((0.5,), (0.5,)) 
    # 이미지 정규화(평균, 표준편차)
    ])
```

### GPU로 변경
- **Apple slicon은 GPU** 기반
- 기억해두고 다른 장치에서 작업할 때 사용
- 모델을 GPU로 이동

```py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

- 텐서를 `devise`(GPU)로 이동

```py
inputs, targets = inputs.to(device), targets.to(device)
```

### 유틸리티
- 저장 및 로드: **가장 자주 사용할 기능**, 반드시 기억
1. 저장 및 로드
- 저장

```py
torch.save(model.state_dict(), 'model.pth'
```

- 로드

```py
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

2. 학습 및 평가 모드 설정

```py
# 학습
model.train()
# 평가
model.eval()
```