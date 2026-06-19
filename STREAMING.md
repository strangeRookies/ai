# RTSP to Web Streaming

The frontend does not use RTSP URLs directly. Keep RTSP on the AI/GPU PC, then expose browser-safe streams to the dashboard.

## Path Standard: `cameraLoginId`

All camera path identifiers are based on `cameraLoginId` as registered in the backend DB.
Do **not** use bare `cam1`, `cam2` paths — use `cam_01`, `cam_02`, etc.

| Layer | URL pattern | Description |
| --- | --- | --- |
| RTSP publish | `rtsp://<host>:8554/cam_01` | AI worker 및 미디어 서버 간의 원본 송출 주소 |
| WebRTC WHEP | `http://<host>:8889/cam_01/whep` | **프론트엔드 기본 영상 주소 (우선 적용)** |
| HLS | `http://<host>:8888/cam_01/index.m3u8` | **프론트엔드 예비 영상 주소 (WebRTC 실패 시 Fallback)** |
| AI Overlay (MJPEG) | `http://<host>:8010` (동적) | AI 분석용 오버레이 포트 (기본값 8010 + AI worker 실행 인덱스) |
| AI runner input | same RTSP path as publish | AI 분석 모듈이 MediaMTX로부터 가져오는 원본 RTSP 피드 |

> [!IMPORTANT]
> **RTSP/WebRTC 포트 가용성과 스트림 존재 여부**
> * MediaMTX의 WebRTC 포트(`8889`)가 열려 있더라도, 해당 `cameraLoginId` 경로로 **실제 RTSP 스트림 송출(Publisher)이 동작하고 있지 않다면 WHEP 404 Not Found** 에러가 발생합니다. 즉, "포트 LISTEN 상태"와 "해당 스트림 경로의 실존 여부"는 다릅니다.

## Architecture

```text
Camera or video file
  -> MediaMTX RTSP server :8554 (path = cameraLoginId, e.g. cam_01)
       +---> WebRTC WHEP :8889/{cameraLoginId}/whep  <--- (Primary Stream)
       +---> HLS :8888/{cameraLoginId}/index.m3u8     <--- (Fallback Stream)
  -> AI overlay MJPEG :8010+ (Camera/AI worker당 1개 포트 배정)
       +---> frontend video player / canvas
```

> [!NOTE]
> * **AI Overlay 포트 주의사항:** 이 포트는 영상 감지 인원(사람)당 하나가 아니라, **카메라/AI worker당 하나**입니다. 한 화면 내에 여러 감지 대상이 있어도 하나의 오버레이 영상에 모든 바운딩 박스가 함께 출력됩니다.
> * 현재 GPU PC에서는 `cam_04`가 `serve_ai_overlay.py --port 8010`으로 지정되어 구동되며, 다중 카메라 구동 시 `overlay-base-port 8010`에 구동 순서(worker index)에 맞춰 `8010`, `8011` 등으로 자동 순회 할당됩니다.

`serve_ai_overlay.py`는 FFmpeg을 쉘 명령어로 호출하지 않고, 내부적으로 OpenCV를 통해 RTSP 스트림을 직접 디코딩하여 이미지 연산을 거친 뒤 MJPEG 스트림을 직접 클라이언트에 쏩니다. FFmpeg은 샘플 비디오나 웹캠 영상을 RTSP 서버에 실시간 공급(Publish)하는 목적으로 주로 사용됩니다.

## 1. Start RTSP Server

```bash
cd /home/welabs/yolo_training/strange_ai_lstm
chmod +x scripts/*.sh
./scripts/run_rtsp_server.sh
```

This serves any publisher path. Active camera paths are registered in the backend DB as `cameraLoginId`.

## 2. Publish Test Video or Webcam

Recorded video (publish to `cam_01`):

```bash
ffmpeg -re -stream_loop -1 \
  -i /path/to/sample.mp4 \
  -an -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://127.0.0.1:8554/cam_01
```

Linux webcam:

```bash
ls /dev/video*
ffmpeg -f v4l2 -i /dev/video0 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -f rtsp rtsp://localhost:8554/cam_01
```

Windows webcam to GPU PC:

```powershell
ffmpeg -list_devices true -f dshow -i dummy
ffmpeg -f dshow -i video="YOUR WEBCAM NAME" -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://GPU_PC_IP:8554/cam_01
```

## 3. MediaMTX Port & Host Network Note

* MediaMTX는 Docker 컨테이너 `mediamtx`로 구동되지만, **`--network=host`**를 사용해 실행됩니다.
* 이 때문에 로컬 윈도우 PC의 `docker ps` 명령어 상 **PORTS 필드에 아무 포트도 표시되지 않더라도**, 실제 GPU 호스트 OS에서는 아래 포트들을 정상적으로 `LISTEN`하고 서비스 중입니다.
  * `8554` (RTSP)
  * `8888` (HLS 스트림 데이터)
  * `8889` (WebRTC WHEP HTTP)
  * `8189` (WebRTC ICE TCP/UDP)
  * `8000`/`8001` (UDP 보조 채널 포트)
* **주의:** `9997` (MediaMTX 관리 API 포트)은 현재 시스템의 동작과 연관된 근거가 없으므로 임의로 확정해 사용하지 않고, 사용이 필요한 경우 원격 환경을 별도 확인해야 합니다.

## [Legacy] Web Stream Bridge (MJPEG Port 8000)

이전 버전의 시스템에서 카메라별로 포트 8000번 대역을 뚫어 다수의 사람이 한 스트림을 쪼개어 보도록 띄웠던 `run_stream_bridge.sh` 환경은 현재 **레거시(Legacy)** 사양입니다. 
신규 설계에서는 개별 카메라 오버레이 뷰어(`serve_ai_overlay.py`)를 통해 `8010`번 대역에서 직접 카메라별로 1개씩 포트를 할당하는 구조를 취합니다.

* **동작 방식 (과거 레거시):**
  ```bash
  export CAMERA_1_RTSP_URL='rtsp://localhost:8554/cam_01'
  export CAMERA_2_RTSP_URL='rtsp://localhost:8554/cam_02'
  ./scripts/run_stream_bridge.sh
  ```
* **프론트엔드 레거시 설정 (`VITE_STREAM_BASE_URL`):**
  레거시 8000 포트용 환경변수 `VITE_STREAM_BASE_URL` 설명은 최신 WebRTC/HLS 우선 구조에서는 사용되지 않습니다.

## Troubleshooting

1. 만약 `rtsp://localhost:8554` 연결 거부(Connection refused)가 발생한다면, 미디어 서버(MediaMTX) 자체가 꺼져 있는 것입니다.
2. 만약 WHEP POST `/cam_01/whep` 호출 시 **404 Not Found**가 발생한다면, 포트 8889는 열려 있으나 해당 스트림(RTSP)을 밀어주는 **송출기(Publisher - ffmpeg 등) 프로세스가 꺼져 있는 상태**입니다.
3. GPU PC 포트 방화벽 차단 해결 (필요시):
   ```bash
   sudo ufw allow 8554/tcp
   sudo ufw allow 8888/tcp
   sudo ufw allow 8889/tcp
   sudo ufw allow 8189/tcp
   sudo ufw allow 8010:8013/tcp
   ```

Do not put RTSP usernames or passwords in frontend code.

