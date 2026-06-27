# Frame Sync Debug & MQTT Payload 검증 최종 보고서

> **최종 갱신**: 2026-06-27 (세션 3 변경 내역 반영)

## 1. 개요

### 1.1 Frame Sync Debug 도입 목적

본 프로젝트의 시스템 구조상 CCTV 영상 스트림 경로(RTSP/WebRTC)와 메타데이터 및 분석 이벤트 전송 경로(MQTT)가 물리적으로 이원화되어 있습니다. 이로 인해 네트워크 지연이나 버퍼링이 발생할 경우 영상 프레임과 이벤트 박스의 싱크가 어긋나는(Overlay 밀림) 현상이 발생할 수 있습니다.

이를 방지하고 모니터링하기 위해 영상 원본 프레임의 고유 ID(rameId)와 각 단계별 처리 시각(Captured/Processed/Published)을 메타데이터에 기록하여 종단 간 지연(Latency) 및 정확한 동기화를 추적하기 위한 디버깅 시스템을 검증 및 도입하였습니다.

### 1.2 Overlay 흐름 설계 의도

`
매 프레임 (camera_topic)
├── events[] → 감지된 모든 사람의 bbox 좌표 (항상 전송)
│   ├── type: "tracking"  → LSTM 평가 대기 중 (초록 박스)
│   └── type: "faint"     → LSTM 평가 완료, confidence 포함 (위험도 색상)
│       └── eventTriggered: true → 알람 확정 (빨간 박스)

이벤트 확정시만 (event_topic)
└── { schemaVersion:"1.1", messageType:"event" } → 백엔드 DB 저장
`

---

## 2. 점검 환경

| 항목 | 값 |
| :--- | :--- |
| **작업 디렉터리 (원격 GPU PC)** | /home/welabs/yolo_training/strange_ai_lstm |
| **Python/Venv 경로** | /home/welabs/yolo_training/strange_ai_lstm/.venv/bin/python (Python 3.12.3) |
| **RTSP 입력 URL** | rtsp://127.0.0.1:8554/cam_04 |
| **MQTT 브로커** | AWS EC2 (15.165.248.37:1883) |
| **테스트 카메라 ID** | cam_04 |
| **실행 대상 스크립트** | scripts/serve_ai_overlay.py |
| **메타데이터 스키마 버전** | "1.1" |
| **LSTM 설정** | sequence_length=30, sequence_stride=15, input_size=51 |

---

## 3. 점검 항목 및 결과

| 점검 항목 | 검증 방법 및 명령 | 결과 | 비고 |
| :--- | :--- | :---: | :--- |
| **venv 환경 확인** | which python && python --version | **PASS** | 가상환경 정상 활성화 및 Python 3.12 구동 확인 |
| **RTSP 스트림 검증** | ffprobe -v error rtsp://127.0.0.1:8554/cam_04 | **PASS** | 비디오 스트림 정상 수신 확인 |
| **MediaMTX 포트 확인** | ss -ltnp | grep -E "8554|8888|8889" | **PASS** | 8554, 8888, 8889 포트 활성 상태 확인 |
| **MQTT 원격 접속** | nc -vz 15.165.248.37 1883 | **PASS** | AWS MQTT 서버 포트 접속 확인 성공 |
| **CLI 옵션 검증** | serve_ai_overlay.py --help | **PASS** | --dry-run, --frame-sync-debug, --mqtt-host/port 모두 정상 |
| **Dry-run 로그 검증** | serve_ai_overlay.py ... --dry-run | **PASS** | [frame-sync] 로그 정상 출력 확인 |
| **MQTT Payload 검증** | AWS 브로커 구독 후 수신 덤프 확인 | **PASS** | schemaVersion 1.1 메타데이터 실시간 전송 확인 |
| **MJPEG 디버그 Overlay** | --frame-sync-debug --mjpeg-debug 활성화 후 visual 확인 | **PASS** | Frame ID, Latency 수치 실시간 표출 성공 |
| **LSTM 30/15/51 검증** | 기동 로그 및 sequence 체크 | **PASS** | length=30, stride=15, input_size=51 표준 스펙 적용 확인 |
| **기존 오류 재발 여부** | AttributeError: dry_run, 51/54 dim mismatch | **PASS** | 모두 원천 해소 (아래 §7 상세 기록) |
| **모든 bbox overlay 전송** | overlay payload events[] 내용 확인 | **PASS** | 이번 세션 수정으로 모든 감지 객체가 항상 포함됨 |
| **IoU 기반 track_id 할당** | supervision_postprocessor.py update() 로직 | **PASS** | 인덱스 기반 오매핑 버그 수정 완료 |
| **py_compile 문법 검증** | 11개 핵심 모듈 전수 검사 | **PASS** | ALL_SYNTAX_OK |

---

## 4. Payload Schema 확인

### 4.1 overlay_topic 페이로드 (safety/cameras/overlay)

매 프레임마다 전송. 감지된 모든 사람이 events[]에 포함됨.

`json
{
  "schemaVersion": "1.1",
  "messageType": "overlay",
  "timestampMs": 1782532905623,
  "streamId": "cam_04",
  "cameraLoginId": "cam_04",
  "frameWidth": 1280,
  "frameHeight": 720,
  "events": [
    {
      "type": "tracking",
      "confidence": 0.85,
      "eventTriggered": false,
      "bbox": {"x": 50, "y": 100, "width": 80, "height": 150},
      "boundingBox": {"x": 50, "y": 100, "width": 80, "height": 150},
      "keypoints": [],
      "trackingId": 2,
      "frameId": 45
    },
    {
      "type": "faint",
      "confidence": 0.72,
      "eventTriggered": false,
      "bbox": {"x": 200, "y": 80, "width": 120, "height": 200},
      "boundingBox": {"x": 200, "y": 80, "width": 120, "height": 200},
      "keypoints": [
        {"x": 156.0, "y": 222.0, "confidence": 0.9}
      ],
      "trackingId": 1,
      "frameId": 92
    }
  ],
  "frameId": 92,
  "capturedAtMs": 1782532905622,
  "processedAtMs": 1782532905623,
  "publishedAtMs": 1782532905623,
  "aiLatencyMs": 1,
  "publishLatencyMs": 1
}
`

### 4.2 event_topic 페이로드 (safety/events)

이벤트 확정시에만 전송. DB 저장 대상.

`json
{
  "schemaVersion": "1.1",
  "messageType": "event",
  "eventId": "evt-20260627-cam_04-1782532910000",
  "timestampMs": 1782532910000,
  "streamId": "cam_04",
  "cameraLoginId": "cam_04",
  "type": "faint",
  "event_type": "faint",
  "memoText": "쓰러짐 의심!",
  "severity": "HIGH",
  "confidence": 0.91,
  "trackingId": 1,
  "frameId": 120,
  "capturedAtMs": 1782532909980,
  "processedAtMs": 1782532909995,
  "publishedAtMs": 1782532910000,
  "aiLatencyMs": 15,
  "publishLatencyMs": 20,
  "sequence": {
    "sequenceLength": 30,
    "sequenceStride": 15,
    "sequenceStartFrameId": 91,
    "sequenceEndFrameId": 120,
    "sequenceStartAtMs": 1782532908980,
    "sequenceEndAtMs": 1782532909980
  },
  "boundingBox": {"x": 200, "y": 80, "width": 120, "height": 200},
  "frameWidth": 1280,
  "frameHeight": 720
}
`

### 4.3 events[].type 필드 의미

| type 값 | 조건 | 프론트엔드 렌더링 |
| :--- | :--- | :--- |
| "tracking" | LSTM 평가 대기 중 (첫 30프레임) | 초록 bbox |
| "faint" | LSTM 평가 완료 (faint_probability 존재) | confidence 값에 따라 색상 변환 |
| "faint" + eventTriggered:true | 알람 확정 (event_topic도 발행됨) | 빨간 bbox + 알람 |

---

## 5. 지연 시간 계산 및 LSTM Sequence 동기화

- **aiLatencyMs** = processedAtMs - capturedAtMs (순수 AI 처리 시간)
- **publishLatencyMs** = publishedAtMs - capturedAtMs (총 내부 지연)
- **E2E Latency** = receivedAtMs(프론트) - capturedAtMs (향후 구현 예정)

### LSTM Sequence 동기화

- sequence_length=30, sequence_stride=15, input_size=51
- LSTM은 단일 프레임이 아닌 30프레임 윈도우로 판단하며, 15프레임마다 새 시퀀스를 방출합니다.
- 이벤트 발생 시 sequence.sequenceStartFrameId ~ sequence.sequenceEndFrameId 구간이 기록됩니다.

---

## 6. 발견된 문제 및 수정 내역

### 6.1 이번 세션 수정 (2026-06-27)

**[수정 1] overlay payload가 LSTM 평가된 객체만 포함하던 버그 (CRITICAL)**

파일: ai/publishers/mqtt_payloads.py

원인: build_overlay_payload()의 events[] 컴프리헨션에 _has_overlay_signal() 필터가 적용되어,
LSTM 예측이 준비되지 않은 객체(첫 30프레임 이내 신규 트랙)가 overlay payload에서 완전히 누락.

수정: bbox가 유효한 모든 객체를 포함하도록 변경. type="tracking" vs type="faint" 구분 및
eventTriggered 필드 추가.

**[수정 2] supervision postprocessor에서 인덱스 기반 track_id 오매핑 버그 (CRITICAL)**

파일: ai/postprocess/supervision_postprocessor.py

원인: track_ids[i] → detections[i] 인덱스 매핑 사용. supervision의 update_with_detections()는
low-confidence 디텍션을 필터링하거나 순서를 바꿀 수 있어 잘못된 track_id가 잘못된
사람의 keypoint에 붙는 무결성 오류가 발생할 수 있었음.

수정: tracked bbox와 원본 detection 간 IoU 매칭 방식으로 교체.

### 6.2 이전 세션 수정 (유지됨)

| 항목 | 상태 |
| :--- | :--- |
| --dry-run CLI 옵션 누락 | 수정됨 |
| AttributeError: Namespace has no dry_run | 수정됨 |
| 8010 포트 좀비 프로세스 | 수정됨 |
| 51/54 차원 불일치 | 재발 없음 |

---

## 7. py_compile 검증 결과

다음 11개 핵심 파일 모두 문법 검증 통과 (ALL_SYNTAX_OK):

- scripts/serve_ai_overlay.py
- ai/publishers/event_publisher.py
- ai/publishers/mqtt_payloads.py
- ai/inference/rtsp_runtime.py
- ai/action/classifier.py
- ai/action/keypoint_sequence_buffer.py
- ai/action/per_track_sequence_buffer.py
- ai/action/lstm_contract.py
- ai/frame_sync.py
- ai/postprocess/supervision_postprocessor.py
- ai/visualization/action_overlay.py

---

## 8. Supervision 제약 준수 및 51차원 스키마 정합성

| 제약 조건 | 파일 | 상태 |
| :--- | :--- | :---: |
| LSTM 51차원 독립성 | lstm_contract.py: DEFAULT_KEYPOINT_INPUT_SIZE = 51 상수 고정 | PASS |
| classifier.py에 supervision 미접촉 | classifier.py에 supervision import 0건 | PASS |
| MQTT payload supervision 무의존 | mqtt_payloads.py에 supervision import 0건 | PASS |
| Overlay 텍스트 cv2.putText 사용 | action_overlay.py L183-186 cv2.putText 직접 렌더링 | PASS |
| YOLO Keypoint 소실 방지 | supervision_postprocessor.py match_keypoints_by_iou + IoU 기반 track_id 매핑 | PASS |

---

## 9. 최종 결론 및 후속 과제

Frame Sync Debug 기능이 성공적으로 구축 및 실가동 환경에 통합되었습니다.
이번 세션에서 overlay payload가 감지된 모든 사람의 bbox를 항상 포함하도록 수정되어
프론트엔드가 LSTM 평가 전후 관계없이 모든 사람을 시각화할 수 있습니다.

**후속 과제**:

1. 프론트엔드 receivedAtMs 추가 — E2E Latency 완전 측정
2. WebRTC frameId 동기화 — 화면 프레임 ID와 MQTT frameId 연계 비교
3. 다중 카메라 동시 테스트 — cam_04 외 채널도 동시 구동 테스트
4. 운영 모드 디버그 비활성화 — --frame-sync-debug, --mjpeg-debug 운영 시 기본 off 유지
5. 프론트엔드 type별 bbox 색상 처리 — "tracking"/"faint"/eventTriggered 값으로 색상 코딩
6. keypoint 54차원 확장 — 향후 모션 피처(center_drop, velocity, torso_angle) 추가 예정
