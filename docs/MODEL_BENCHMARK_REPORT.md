# AI 모델 벤치마크 보고서

## 결론

최종 pose extractor는 **YOLO26n-pose (`yolo26n-pose.pt`)**로 결정했다.

이 결정은 YOLO 모델 단독의 FPS, latency, fall-candidate rule count만 보고 내린 결정이 아니다. 최종 목적은 낙상/실신 상황에 가까운 **Faint class를 LSTM이 잘 구분하도록 안정적인 skeleton sequence를 만드는 것**이므로, 최종 선택 기준은 downstream LSTM Normal/Faint 분류 성능이다.

현재 운영 기준 파이프라인은 다음과 같다.

```text
RTSP 입력
-> YOLO26n-pose bbox/keypoint 추출
-> skeleton sequence 생성
-> LSTM Normal/Faint 분류
-> threshold/post-processing
-> MQTT event publishing
```

YOLOv11n-pose도 비교 대상에 포함해 테스트했지만, YOLO26n-pose가 downstream LSTM 기준에서 Faint detection 성능과 repeated-seed 안정성이 더 좋아 최종 후보에서 제외했다. 최신 확장 benchmark는 YOLO26n-pose만 대상으로 수행한다.

## 데이터셋 요약

| 항목 | 값 |
| --- | ---: |
| 전체 metadata row | 215,541 |
| Normal row | 205,594 |
| Faint row | 9,947 |
| source video 수 | 183 |
| Faint가 등장하는 source video 수 | 174 |
| domain | indoor_background, indoor_chromakey, outdoor |

데이터셋은 Normal이 Faint보다 훨씬 많은 불균형 구조다. 그래서 benchmark에서는 단순 row 순서나 global random sampling을 쓰면 안 된다. train/val/test split별로 Normal과 Faint를 같은 수만큼 뽑는 class-balanced sampling을 사용해야 한다.

## 평가 지표 설명

비전공자 기준으로 보면, 이번 benchmark의 핵심 지표는 다음과 같다.

| 지표 | 의미 | 해석 |
| --- | --- | --- |
| Faint recall | 실제 Faint 중 모델이 Faint라고 잡아낸 비율 | 높을수록 실신/낙상 상황을 놓칠 가능성이 낮다 |
| Precision | Faint라고 예측한 것 중 실제 Faint인 비율 | 낮으면 Normal을 Faint로 잘못 알리는 false alarm이 늘어난다 |
| F1-score | recall과 precision의 균형 점수 | recall만 높이고 오탐이 너무 많아지는 상황을 함께 점검한다 |
| Confusion matrix | Normal/Faint를 각각 어떻게 맞히고 틀렸는지 보여주는 표 | 어떤 방향의 실수가 많은지 확인한다 |
| Threshold audit | Faint 확률 기준값을 바꿨을 때 recall/F1이 어떻게 변하는지 확인 | 운영 threshold를 정하는 근거가 된다 |
| Repeated-seed mean/std | 여러 random seed로 학습했을 때 성능 평균과 흔들림 | 특정 seed 운이 아니라 안정적인 성능인지 확인한다 |

안전 이벤트 시스템에서는 Faint recall이 특히 중요하다. 다만 threshold를 너무 낮추면 Normal도 Faint로 많이 잡혀 false alarm이 늘 수 있으므로, recall과 F1-score를 같이 본다.

## Pose-only benchmark와 LSTM benchmark의 차이

Pose-only benchmark는 YOLO가 얼마나 빠르게 사람 bbox/keypoint를 뽑는지 확인하는 단계다. FPS, latency, keypoint missing rate, fall-candidate count 같은 값을 볼 수 있다.

하지만 최종 모델 선택은 pose-only 결과만으로 하지 않는다. 실제 서비스의 판단은 YOLO가 만든 skeleton sequence를 LSTM이 받아 Normal/Faint를 분류한 결과이기 때문이다. 따라서 최종 선택은 다음 순서로 판단한다.

1. Faint recall
2. F1-score
3. threshold별 recall/F1 균형
4. repeated-seed 안정성
5. sequence 생성 안정성
6. runtime feasibility

## 300/class benchmark 결과

300/class benchmark는 train/val/test split마다 Normal 300개, Faint 300개를 선택해 실행했다.

| 항목 | 값 |
| --- | ---: |
| train selected | Normal 300, Faint 300 |
| val selected | Normal 300, Faint 300 |
| test selected | Normal 300, Faint 300 |
| generated sequences | 1,380 |
| zero sequence clips | 108 |
| best balance threshold | 0.4 |
| threshold 0.4 Faint recall | 약 0.672 |
| threshold 0.4 F1-score | 약 0.650 |

threshold 0.4는 Faint를 더 적극적으로 잡으면서도 F1-score 균형이 비교적 좋았다. 이 결과는 운영 threshold 후보로 0.4를 검토할 근거가 된다.

## 1000/class benchmark 결과

1000/class benchmark는 더 큰 규모로 train/val/test split마다 Normal 1000개, Faint 1000개를 선택해 실행했다.

| 항목 | 값 |
| --- | ---: |
| train selected | Normal 1000, Faint 1000 |
| val selected | Normal 1000, Faint 1000 |
| test selected | Normal 1000, Faint 1000 |
| train generated sequences | 5,024 |
| eval generated sequences | 4,676 |
| threshold 0.5 Faint recall | 0.586382 |
| threshold 0.5 F1-score | 0.617608 |
| threshold 0.3 Faint recall | 0.784553 |
| threshold 0.3 F1-score | 0.665661 |
| repeated-seed mean Faint recall | 0.658198 |
| repeated-seed mean F1-score | 0.648263 |

threshold 0.5는 기본 argmax에 가까운 보수적 기준이며 현재 운영 목적에는 너무 보수적이다. threshold 0.3은 Faint recall을 크게 올렸고 F1-score도 더 높았다. 다만 threshold가 낮아질수록 false alarm 가능성이 커질 수 있으므로, 운영에서는 threshold 0.3을 기본 후보로 쓰되 연속 Faint sequence 조건과 camera-level cooldown을 함께 적용한다.

## 현재 해석

YOLO26n-pose는 최종 pose extractor로 채택한다. 특히 1000/class benchmark에서 threshold 0.3은 recall 0.784553, F1 0.665661로 threshold 0.5보다 높은 Faint 탐지 성능을 보였다. 따라서 다음 단계는 추가 모델 비교가 아니라 YOLO26n-pose + LSTM Faint detection의 실시간 추론 준비다.

운영 threshold는 즉시 하나로 고정하기보다 다음 방식으로 정하는 것이 좋다.

- 기본 운영 후보: 0.3
- 비교 후보: 0.4, 0.5
- 실시간 RTSP smoke test에서 false alarm 빈도 확인
- 일정 frame/window 이상 Faint가 유지될 때만 MQTT event 발행
- ByteTrack으로 같은 사람 track의 연속성을 유지
- 이벤트 cooldown/debounce로 중복 알림 방지

## 최종 결정

최종 모델 결정:

```text
Pose extractor: YOLO26n-pose
LSTM task: Normal/Faint classification
Runtime input: keypoint skeleton sequence
Initial runtime threshold candidate: 0.3
Post-processing: consecutive Faint sequences + camera-level cooldown
```

최종 선택 근거는 YOLO26n-pose의 downstream LSTM Faint detection 성능과 repeated-seed 안정성이다.
