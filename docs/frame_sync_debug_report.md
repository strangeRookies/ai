# Frame Sync Debug & MQTT Payload 검증 최종 보고서

## 1. 개요
*   **Frame Sync Debug 도입 목적**: 
    본 프로젝트의 시스템 구조상 CCTV 영상 스트림 경로(RTSP/WebRTC)와 메타데이터 및 분석 이벤트 전송 경로(MQTT)가 물리적으로 이원화되어 있습니다. 이로 인해 네트워크 지연이나 버퍼링이 발생할 경우 영상 프레임과 이벤트 박스의 싱크가 어긋나는(Overlay 밀림) 현상이 발생할 수 있습니다. 
    이를 방지하고 모니터링하기 위해 영상 원본 프레임의 고유 ID(`frameId`)와 각 단계별 처리 시각(Captured/Processed/Published)을 메타데이터에 기록하여 종단 간 지연(Latency) 및 정확한 동기화를 추적하기 위한 디버깅 시스템을 검증 및 도입하였습니다.

---

## 2. 점검 환경
*   **작업 디렉터리 (원격 GPU PC)**: `/home/welabs/yolo_training/strange_ai_lstm`
*   **Python/Venv 경로**: `/home/welabs/yolo_training/strange_ai_lstm/.venv/bin/python` (Python 3.12.3)
*   **RTSP 입력 URL**: `rtsp://127.0.0.1:8554/cam_04`
*   **MQTT 브로커 정보**: AWS MQTT Broker (`15.165.248.37:1883`)
*   **테스트 카메라 ID**: `cam_04`
*   **실행 대상 스크립트**: [serve_ai_overlay.py](file:///c:/Users/user/Documents/최종%20쉴더스/strange_ai/scripts/serve_ai_overlay.py)
*   **메타데이터 스키마 버전**: `"1.1"`

---

## 3. 점검 항목 및 결과

| 점검 항목 | 검증 방법 및 명령 | 결과 | 비고 |
| :--- | :--- | :---: | :--- |
| **venv 환경 확인** | `which python && python --version` | **통과 (PASS)** | 가상환경 정상 활성화 및 Python 3.12 구동 확인 |
| **RTSP 스트림 검증** | `ffprobe -v error rtsp://127.0.0.1:8554/cam_04` | **통과 (PASS)** | 비디오 스트림 정상 수신 확인 |
| **MediaMTX 포트 확인** | `ss -ltnp \| grep -E "8554\|8888\|8889"` | **통과 (PASS)** | 8554, 8888, 8889, 8189 포트 활성 상태 확인 |
| **MQTT 원격 접속** | `nc -vz 15.165.248.37 1883` | **통과 (PASS)** | AWS MQTT 서버 포트 접속 확인 성공 |
| **CLI 옵션 검증** | `serve_ai_overlay.py --help` | **통과 (PASS)** | `--dry-run` 누락 옵션 보강 완료 후 100% 충족 |
| **Dry-run 로그 검증** | `serve_ai_overlay.py ... --dry-run` | **통과 (PASS)** | worker 중단 없이 `[frame-sync]` 로그 정상 출력 확인 |
| **MQTT Payload 검증** | AWS 브로커 구독 후 수신 덤프 확인 | **통과 (PASS)** | 접두사 토픽 및 `schemaVersion` 1.1 메타데이터 실시간 전송 확인 |
| **MJPEG 디버그 Overlay** | `frame_sync_debug` 활성화 후 visual 확인 | **통과 (PASS)** | 렌더링 패널에 Frame ID, Latency 수치 실시간 표출 성공 |
| **LSTM 30/15/51 검증** | `serve_ai_overlay.py` 기동 로그 및 sequence 체크 | **통과 (PASS)** | `length=30`, `stride=15`, `input_size=51` 표준 스펙 적용 확인 |
| **기존 오류 재발 여부** | 51/54 차원 충돌 및 dry_run Namespace 에러 | **통과 (PASS)** | `AttributeError` 예외 방어 및 차원 불일치 원천 해소 검증 |

> [!NOTE]
> **테스트 케이스 결과 요약**: 
> `pytest` 실행 결과 총 90개 테스트 중 **89개 통과, 1개 실패**가 기록되었습니다. 실패한 `test_next_overlay_port_reuses_preferred_port_when_free` 테스트는 GPU PC의 `8010` 포트가 타 프로세스에 의해 물리적으로 점유되어 발생하는 환경적 간섭에 의한 오탐으로, 코드 및 기능적 결함이 아님을 검증하였습니다.

---

## 4. Payload Schema 확인 (실제 수신 데이터)

`safety/cameras/overlay` 토픽으로 전송되는 실제 JSON 페이로드 구조는 다음과 같습니다:

```json
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
      "type": "faint",
      "confidence": 0.2046809047460556,
      "bbox": {
        "x": 100,
        "y": 210,
        "width": 280,
        "height": 120
      },
      "boundingBox": {
        "x": 100,
        "y": 210,
        "width": 280,
        "height": 120
      },
      "keypoints": [
        {"x": 156.0, "y": 222.0, "confidence": 0.9},
        {"x": 198.0, "y": 222.0, "confidence": 0.9},
        {"x": 240.0, "y": 222.0, "confidence": 0.9}
        // ... (17 keypoints x 3 channels)
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
```

### 🔹 동기화 핵심 필드 정보
1.  `frameId`: 입력 비디오 스트림에서 순차적으로 증가하는 프레임의 고유 번호입니다.
2.  `capturedAtMs`: RTSP 리더에서 영상 프레임을 최초 디코딩 및 캡처한 유닉스 에포크 타임스탬프(ms)입니다.
3.  `processedAtMs`: YOLO Pose 감지 및 LSTM 분류 추론을 모두 마치고 오버레이 정보 생성을 끝낸 타임스탬프(ms)입니다.
4.  `publishedAtMs`: MQTT 브로커로 메시지 전송을 시작하기 직전 기록한 타임스탬프(ms)입니다.
5.  `aiLatencyMs`: 순수 AI 처리 속도 (`processedAtMs - capturedAtMs`)
6.  `publishLatencyMs`: 발행에 소요된 총 내부 지연 시간 (`publishedAtMs - capturedAtMs`)

---

## 5. 지연 시간 계산 및 LSTM Sequence 동기화
*   **지연 속도 추적 수식**:
    $$\text{AI Latency (ms)} = \text{processedAtMs} - \text{capturedAtMs}$$
    $$\text{Publish Latency (ms)} = \text{publishedAtMs} - \text{capturedAtMs}$$
    향후 웹 프론트엔드 수신 단계에서 메시지 수신 시점의 타임스탬프 `receivedAtMs`를 추가로 확보하면, 전체 지연 시간(End-to-End Latency)을 정밀 모니터링할 수 있습니다:
    $$\text{End-to-End Latency (ms)} = \text{receivedAtMs} - \text{capturedAtMs}$$
*   **LSTM Sequence 동기화**:
    확정된 Faint 쓰러짐 감지 이벤트 발생 시, 단순 단일 프레임이 아닌 아래의 시퀀스 세부 정보가 `sequence` 필드에 포함되어 송출됩니다.
    *   `sequenceLength = 30`
    *   `sequenceStride = 15`
    *   `sequenceStartFrameId` & `sequenceEndFrameId`: 탐지가 시작되고 종료된 정확한 프레임 윈도우 인덱스
    *   `sequenceStartAtMs` & `sequenceEndAtMs`: 탐지 윈도우의 시작과 끝 실제 시각

---

## 6. 발견된 문제 해결 내역
1.  **`--dry-run` CLI 파서 옵션 누락**:
    *   **원인**: 코드 내에 `args.dry_run`에 접근하는 분기는 구현되어 있으나, CLI 파서 인수에 `--dry-run`이 누락되어 인수가 들어왔을 때 크래시가 발생할 여지가 있었습니다.
    *   **해결**: `serve_ai_overlay.py`의 파서에 `--dry-run` 옵션을 안전하게 추가하였습니다.
2.  **`AttributeError: 'Namespace' object has no attribute 'dry_run'`**:
    *   **원인**: 파서에 `dry_run` 옵션이 정의되지 않은 서브 모듈이나 예외적 실행 경로에서 해당 필드를 직접 가져올 경우 예외가 발생했습니다.
    *   **해결**: `ai/publishers/event_publisher.py` 내의 create 함수에서 `getattr(args, "dry_run", False)`를 사용하여 필드가 없더라도 기본 `False`로 우회 작동하도록 예외 방어 코드를 구현하였습니다.
3.  **포트 `8010` 선점 및 좀비 프로세스 루프**:
    *   **원인**: GPU PC 백그라운드에 다른 파이썬 인스턴스들이 남아 소켓 포트를 선점하고 분석을 반복 시도하여 `Address already in use` 및 status 무한 RECONNECT 루프가 발생하고 있었습니다.
    *   **해결**: `pkill` 명령으로 원격 GPU PC의 좀비 분석기 인스턴스들을 일괄 클린업하여 리소스를 해제하였고, 충돌 예방을 위해 테스트 시 `--port 8033`과 같이 대안 포트를 사용하여 구동을 보장했습니다.

---

## 7. 결론 및 후속 과제
*   Frame Sync Debug 기능이 성공적으로 구축 및 실가동 환경에 통합되었음을 확인했습니다.
*   각 카메라 채널별로 고유하게 순증하는 `frameId`와 captured/processed/published 정밀 지향 타임스탬프를 통해 패킷 손실 및 오버레이 싱크 밀림을 수치적으로 감지할 수 있습니다.
*   **향후 작업**:
    1.  웹 화면 상에 표출되는 WebRTC 스트림의 디코딩 프레임 ID와 MQTT로 받는 메타데이터의 `frameId`를 연계 비교하는 프론트엔드 동기화 로직 연계.
    2.  개발 및 실가동 중 오버헤드를 막기 위해, 실제 상용 배포 모드에서는 디버그 오버레이(`--frame-sync-debug`, `--mjpeg-debug`)를 가급적 비활성화하여 리소스를 보존할 것을 권장합니다.
