# AI 엔진 개발 가이드 (AI_GUIDE)

본 문서는 스마트 안전 관제 시스템의 AI 모델 및 추론 엔진(`strange_ai` / `strange_ai_lstm`) 개발자를 위한 기술 명세서입니다.

---

## 🧠 AI 서버 역할 및 파이프라인 개요

AI 분석 서버는 CCTV 카메라 및 동영상 파일의 RTSP 스트림을 실시간 수신하여 프레임 단위로 인물 포즈를 추출하고, 이들의 시간 흐름을 시퀀스로 분석해 쓰러짐/실신 동작(Faint)을 판단한 후 MQTT Broker를 통해 이를 시스템에 보고하는 역할을 담당합니다.

### 핵심 파이프라인
```text
RTSP 입력 ➡ OpenCV 프레임 캡처 ➡ YOLO26n-pose 객체/관절 추출 ➡ ByteTrack 추적 ➡ Keypoint Sequence Buffer 적재 ➡ LSTM Faint 분류 ➡ Post-Processing (Threshold & Cooldown) ➡ MQTT Publish
```

---

## 📂 주요 폴더 구조

```text
strange_ai/ (GPU PC 분석 경로)
├── ai/
│   ├── action/              # 행동 분류 모델(LSTM) 및 버퍼 관리
│   │   ├── faint_post_processing.py
│   │   └── per_track_sequence_buffer.py
│   ├── detection/           # 객체 탐지 어댑터 (YOLOv8, YOLO26n)
│   ├── evaluation/          # 추론 데이터 기록 및 오프라인 평가지표 튜닝
│   ├── inference/           # 비디오 리더기 및 런타임 빌더
│   ├── publishers/          # MQTT 및 Console 전송 클라이언트
│   └── visualization/       # 실시간 바운딩 박스 & 뼈대 오버레이 렌더링
├── benchmark/               # YOLO/LSTM 모델 벤치마크 및 테스트 툴
│   ├── results/             # 벤치마크 수행 결과 로그 및 플롯 저장소
│   └── compare_lstm_extractors.py
├── configs/                 # 카메라 설정 및 모델 하이퍼파라미터 정의
├── scripts/                 # 로컬 기동 및 배치 스케줄러 스크립트
│   ├── run_rtsp_inference.py       # 단일 RTSP 스트림 실시간 분석 구동기
│   ├── run_registered_cameras.py   # 다중 등록 카메라 멀티 스레드 분석 관리자
│   └── start_simulated_rtsp_from_folder.py # 데모 영상 RTSP 스트림 공급기
├── main.py                  # AI 파이프라인 CLI 메인 엔트리
└── requirements.txt         # 필요 의존성 패키지
```

---

## ⚙️ 주요 실행 파일 및 모델 위치

### 1. 주요 실행 스크립트
* **`scripts/run_rtsp_inference.py`**: 단일 RTSP 스트림을 대상으로 감지, 분류 및 MQTT 전송을 직접 수행하는 핵심 프로세스.
* **`scripts/run_registered_cameras.py`**: 백엔드의 `/api/cameras/active` API를 주기적으로 폴링해, 분석 대상 리스트를 스레드로 동적 기동/관리하는 기동 관리자.
* **`scripts/start_simulated_rtsp_from_folder.py`**: 로컬에 저장된 데모 영상 파일들을 MediaMTX를 거쳐 RTSP 스트림으로 끊임없이 변환/루핑 송출하는 송출기.

### 2. 가중치 모델 파일 위치
* **YOLO Pose 모델:**
  * 기본값: 프로젝트 루트 내 `yolo26n-pose.pt`
* **Downstream LSTM 모델:**
  * 기본값: `benchmark/results/lstm_final_11n_vs_26n_audit/YOLO26n-pose/best.pt`

---

## 🛠️ 핵심 분석 프로세스 상세

### 1. YOLO26n-pose 가중치 선정 이유
* 벤치마크를 진행한 결과, `YOLOv11n-pose`는 추론 FPS 측면에서 우수했으나, Downstream LSTM과 결합했을 때 미세한 노이즈로 인해 Normal/Faint 분류 경계에서 흔들림이 더 컸습니다.
* **`YOLO26n-pose`**는 LSTM 입력으로 제공되는 관절 좌표 시퀀스의 흔들림이 매우 적고 일관성을 보여 최종 LSTM 앙상블에서 높은 Faint Recall 및 시드 결정성(seed stability)을 충족하여 선정되었습니다.

### 2. ByteTrack 기반 다중 객체 추적
* RTSP 파이프라인은 OpenCV의 프레임 읽기 불안정 또는 가림 현상(occlusion)으로 객체를 일시적으로 유실할 때 고유 ID가 뒤바뀌는 것을 막아야 합니다.
* 이를 방지하기 위해 `SimpleTrackAssigner`에 바운딩박스 IoU 매칭을 가미한 **ByteTrack-style fallback tracker**가 작동합니다.
* 감지가 누락되어도 `track_buffer` 설정 시간(기본 45프레임/약 1.5초) 동안 추적 상태를 소멸시키지 않고 좌표 예측으로 유지합니다.

### 3. Keypoint Sequence Buffer 생성
* 각 `track_id`에 할당된 포즈가 입력되면 `PerTrackKeypointSequenceBuffers`에 17개 COCO 포즈 관절의 (x, y, confidence) 정보를 30프레임 동안 적재합니다.
* 버퍼 사이즈: `(30, 17, 3)` (Frames, Keypoints, Channels)
* Stride: 기본 1프레임 단위 슬라이딩 윈도우.

### 4. missing keypoint 처리 (결측치 대체)
* 프레임 내 사람의 신체 일부가 구조물에 가려져 신뢰도(Confidence)가 기준 이하로 떨어진 관절은 이전 프레임의 위치 값을 지수 가중 이동평균(EMA) 필터로 보간 및 완화하여 사용합니다.
* 가중 필터 Alpha 계수: `0.60` (급격한 튀는 현상 제어).

### 5. LSTM Faint/Normal 분류 흐름 및 임계치
* 30프레임 버퍼가 꽉 차는 시점부터 LSTM 분류기가 동작하여 현재 시퀀스가 `Faint`(쓰러짐) 행동인지 여부의 확률(0.0 ~ 1.0)을 추론합니다.
* **추론 임계값 (Action Threshold):** 기본 `0.3` (Faint Recall을 최대화하고 위경보를 줄이기 위한 최적화 지점).

### 6. 디바운스 및 Cooldown 처리
* **연속 감지 디바운싱:** 단발성 예측 오차로 인한 위경보(False Alarm)를 방지하기 위해, 연속적으로 `--min-consecutive-faint` (기본값: 2회) 이상의 분석 루프에서 Faint 판정을 받아야 비로소 경보가 확정됩니다.
* **카메라 쿨다운:** 한 명의 객체로 인해 경보가 중복 발송되는 것을 억제하기 위해, 동일 `track_id`에 대해 `--camera-cooldown-seconds` (기본값: 10초) 동안 중복 이벤트를 억제(Debounce)합니다.

---

## 📡 MQTT Publish 및 Payload 예시

* **토픽명:** `safety/events`
* **형식:** JSON

### JSON Payload Schema
```json
{
  "type": "fall_detected",
  "camera_id": "cam_01",
  "timestamp": "2026-06-16T09:40:00Z",
  "severity": "HIGH",
  "message": "쓰러짐 의심 상황이 감지되었습니다.",
  "source": "edge-ai",
  "track_id": 7,
  "metadata": {
    "bbox": [100, 150, 280, 390],
    "confidence": 0.91,
    "rule_score": 0.87,
    "pose_state": "LYING",
    "model_name": "yolo26n-pose"
  }
}
```

---

## 🪵 로그 및 오류 진단 확인 방법

* **로그 확인:** AI 추론 엔진 실행 시 백그라운드로 작동되도록 리다이렉션을 설정하였다면 다음 로그 파일을 모니터링합니다.
  ```bash
  tail -f rtsp_server.log
  tail -f publisher.log
  tail -f ai_runner.log
  ```
* **오프라인 튜닝 데이터 로그:** `--evaluation-log` 옵션을 활성화하면 런타임 중 프레임 예측별 label, score, confidence 데이터가 JSONL 파일에 정밀 기록되어 `evaluate_prediction_logs.py` 스크립트를 통한 오프라인 최적 임계치 sweeping 분석이 가능해집니다.

---

## ⚠️ AI 파트 수정 시 주의할 점
1. **CUDA 메모리 및 속도 누수 방지:** 비디오 Reader 루프에서 CUDA 캐시를 비우지 않거나, 캡처된 OpenCV Frame을 메모리 해제하지 않고 다중 스레드에 보관하면 GPU PC의 VRAM 부족(OOM)으로 폭사하게 됩니다.
2. **패킷 드롭 전략 엄수:** 프레임 캡처 스레드 큐 사이즈를 항상 작게 설정해 실시간성 지연 누적을 격리시키십시오.
3. **모델 교체 호환성 유지:** 차후 `Fall / Fight` 모델 추가 시, sequence buffer 구조 변경 및 `event_type` 추가를 백엔드와 사전 협의하고 MQTT schema 하위 호환성을 보장해야 합니다.
