---
layout: post
title: Quiz LLM 사용자 자료 입력
subtitle: TIL Day 105
cover-img: "/assets/img/background.png"
thumbnail-img: ''
share-img: ''
tags: [TIL, LLM, Tools]
author: polaris0208
---
> 사용자가 유튜브 영상의 내용을 바탕으로 학습하고 싶은 경우 **URL**을 입력하여 데이터를 얻을 수 있도록 기능추가
> 자동생성이 아닌 사람이 제작한 자막의 경우에는 오타가 적고 누락되는 내용도 거의 없어 양질의 데이터가 될 수 있음

## `youtube-transcript-api` 활용
- 유튜브 영상의 스크립트 데이터 수집
   - 스크립트 정보
      - 언어별 스크립트 정보
      - 자동 생성 여부
      - 자막내용, 시작 시간, 지속시간


## 설치 및 임포트
- `youtube-transcript-api==0.6.3`

```py
from youtube_transcript_api import YouTubeTranscriptApi
import re
```

## 비디오 ID 추출
- 해당 **API**는 비디오 **ID**를 기반으로 작동하기 때문에 별도로 추출하는 로직 필요

```py
def get_video_id(url):
    # ?:v= 기본 구조
    # \/ 축소형
    # {11} 11자 구조
    video_id_pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(video_id_pattern, url)
    if match:
        return match.group(1)
    return None
```

## 스크립트 추출
- `languages=['ko', 'en', 'en-US']` : 우선으로 찾을 언어 설정
   - `en-US`: 미국의 일부 영상에서 `en`과 구별하여 등록
- 텍스트 부분만 추출하여 반환

```py
def get_script(video_id):
    # 텍스트, 시작 시점, 자막 지속시간 딕셔너리 구조
    subtitle = ''
    transcription = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en', 'en-US'])
    for content in transcription:
        subtitle += f'{content['text']} \n'
    return subtitle
```