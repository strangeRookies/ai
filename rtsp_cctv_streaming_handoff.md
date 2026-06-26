# RTSP CCTV Live Streaming Handoff

작성일: 2026-05-29

## 목표

CCTV 관제 화면에서 실제 카메라 영상을 실시간으로 보여주기 위한 RTSP 기반 스트리밍 구조를 구성했다.

현재 구현 목표는 다음과 같다.

- 로컬 PC 웹캠을 GPU PC의 RTSP 서버로 송출
- GPU PC에서 RTSP 스트림을 읽어 브라우저가 볼 수 있는 MJPEG 스트림으로 변환
- 프론트엔드 대시보드에서 실시간 CCTV 카드로 표시
- 이후 AI Worker가 추론 결과를 백엔드로 보내고, 백엔드는 이벤트/상태를 관리하는 방향으로 확장

## 전체 구조

```text
로컬 PC 웹캠
  -> ffmpeg
  -> SSH tunnel localhost:18554
  -> GPU PC MediaMTX RTSP server localhost:8554/cam1
  -> strange_ai/serve_mjpeg.py localhost:8000
  -> SSH tunnel localhost:18000
  -> strange_front localhost:5173
```

프론트 포트와 영상 포트는 역할이 다르다.

```text
localhost:5173  = 프론트엔드 Vite 개발 서버
localhost:18000 = GPU PC의 MJPEG 브릿지를 로컬에서 보는 SSH 터널 포트
localhost:18554 = 로컬 웹캠을 GPU PC RTSP 서버로 보내는 SSH 터널 포트
```

## 주요 코드 변경

### strange_ai

추가/수정된 파일:

- `serve_mjpeg.py`
  - RTSP URL을 OpenCV로 읽는다.
  - `/stream/<camera_id>`로 MJPEG 스트림을 제공한다.
  - `/cameras`, `/health` 상태 API를 제공한다.
- `stream/mediamtx.yml`
  - MediaMTX RTSP 서버 설정
  - `cam1`, `cam2`, `cam3`, `cam4` path 사용
- `scripts/run_rtsp_server.sh`
  - MediaMTX 또는 Docker 기반 RTSP 서버 실행
- `scripts/run_stream_bridge.sh`
  - `serve_mjpeg.py` 브릿지 실행
- `scripts/publish_sample_video.sh`
  - 영상 파일을 RTSP로 송출
- `scripts/publish_webcam_linux.sh`
  - Linux 웹캠을 RTSP로 송출
- `STREAMING.md`
  - RTSP 서버, 브릿지, 테스트 송출 절차 문서화
- `.env.example`
  - MJPEG/RTSP 관련 환경변수 추가

### strange_front

구현된 방향:

- 프론트에서는 RTSP URL을 직접 쓰지 않는다.
- `VITE_STREAM_BASE_URL`로 MJPEG 브릿지 주소를 받는다.
- CCTV 카드의 `<img>`가 `/stream/camera-1` 같은 브라우저 안전 URL을 읽는다.

로컬 프론트 `.env` 예시:

```env
VITE_STREAM_BASE_URL=http://localhost:18000
```

프론트 접속 주소:

```text
http://localhost:5173
```

## 왜 SSH 터널을 썼는가

로컬 PC와 GPU PC가 둘 다 `192.168.0.x` 대역처럼 보였지만 서로 ping이 되지 않았다.

확인된 상태:

- GPU PC IP: `192.168.0.66`
- 로컬 PC IP: `192.168.0.151`
- 양방향 ping 실패
- GPU PC는 `enp130s0` 유선랜 사용
- Wi-Fi 인터페이스는 보이지 않음

즉 로컬 PC와 GPU PC가 같은 대역처럼 보여도 실제로는 유선/무선 격리, 게스트망, AP isolation, 공유기 설정 등의 이유로 직접 통신이 불가능했다.

하지만 SSH 접속은 가능했기 때문에 SSH 터널로 우회했다.

## 실행 순서

아래 명령은 실제 개발 중 성공한 구조 기준이다.

### 1. GPU PC에서 RTSP 서버 실행

GPU PC 터미널 1:

```bash
cd /home/welabs/yolo_training/strange_ai
./scripts/run_rtsp_server.sh
```

정상 로그 예시:

```text
MediaMTX v1.18.2
listener opened on :8554
```

### 2. GPU PC에서 MJPEG 브릿지 실행

GPU PC 터미널 2:

```bash
cd /home/welabs/yolo_training/strange_ai
source .venv/bin/activate
./scripts/run_stream_bridge.sh
```

이미 실행 중이면 새로 실행할 때 다음 오류가 날 수 있다.

```text
OSError: [Errno 98] Address already in use
```

이 경우 기존 브릿지 확인:

```bash
ps aux | grep serve_mjpeg.py | grep -v grep
curl http://localhost:8000/cameras
```

멈춘 프로세스가 포트만 잡고 있으면 종료:

```bash
kill <PID>
```

필요하면 강제 종료:

```bash
kill -9 <PID>
```

### 3. Windows 로컬 PC에서 RTSP 입력 터널 열기

Windows 터미널 1:

```cmd
ssh -N -L 18554:localhost:8554 welabs@원래_접속하던_서버주소
```

주의:

- `192.168.0.66`이 아니라 실제로 SSH 접속할 때 쓰던 주소를 사용한다.
- 이 창은 아무 출력 없이 멈춰 있는 상태가 정상이다.
- 창을 닫으면 터널도 종료된다.

### 4. Windows 로컬 PC에서 MJPEG 출력 터널 열기

Windows 터미널 2:

```cmd
ssh -N -L 18000:localhost:8000 welabs@원래_접속하던_서버주소
```

이 터널은 로컬 브라우저/프론트에서 GPU PC의 `serve_mjpeg.py`를 보기 위한 것이다.

확인:

```cmd
curl http://localhost:18000/cameras
```

JSON이 나오면 터널이 정상이다.

### 5. Windows 로컬 PC에서 웹캠을 RTSP로 송출

먼저 웹캠 이름 확인:

```cmd
ffmpeg -list_devices true -f dshow -i dummy
```

웹캠 옵션 확인:

```cmd
ffmpeg -f dshow -list_options true -i video="USB HD Webcam"
```

확인된 지원 옵션:

```text
vcodec=mjpeg 1280x720 fps=30
vcodec=mjpeg 640x480 fps=30
vcodec=mjpeg 320x240 fps=30
pixel_format=yuyv422 640x480 fps=30
```

실제 송출 명령:

```cmd
ffmpeg -rtbufsize 512M -f dshow -vcodec mjpeg -framerate 30 -video_size 320x240 -i video="USB HD Webcam" -vf fps=10 -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -pix_fmt yuv420p -g 10 -bf 0 -an -rtsp_transport tcp -f rtsp rtsp://localhost:18554/cam1
```

MediaMTX 정상 로그:

```text
stream is available and online, 1 track (H264)
is publishing to path 'cam1'
```

### 6. 브라우저에서 직접 스트림 확인

Windows 브라우저:

```text
http://localhost:18000/stream/camera-1
```

여기서 웹캠이 보이면 브릿지와 터널은 정상이다.

### 7. 프론트에서 확인

`strange_front/.env`:

```env
VITE_STREAM_BASE_URL=http://localhost:18000
```

Vite는 `.env` 변경 후 재시작해야 한다.

```cmd
npm run dev
```

브라우저:

```text
http://localhost:5173
```

## 다른 PC 카메라 추가

다른 PC의 카메라를 추가하려면 해당 PC에서 8554 터널만 열면 된다.

```cmd
ssh -N -L 18554:localhost:8554 welabs@원래_접속하던_서버주소
```

그 PC 웹캠을 `cam2`로 송출:

```cmd
ffmpeg -rtbufsize 512M -f dshow -vcodec mjpeg -framerate 30 -video_size 320x240 -i video="그 PC 웹캠 이름" -vf fps=10 -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -pix_fmt yuv420p -g 10 -bf 0 -an -rtsp_transport tcp -f rtsp rtsp://localhost:18554/cam2
```

그 PC에서도 화면을 보고 싶으면 8000 터널도 추가로 연다.

```cmd
ssh -N -L 18000:localhost:8000 welabs@원래_접속하던_서버주소
```

정리:

```text
카메라만 보내는 PC: 8554 터널 필요
화면도 보는 PC: 8000 터널도 필요
프론트도 띄우는 PC: .env에 VITE_STREAM_BASE_URL=http://localhost:18000 필요
```

## 문제 해결 기록

### `DESCRIBE failed: 404 Not Found`

의미:

```text
RTSP 서버는 켜져 있지만 해당 path(cam1 등)에 아직 영상이 publish되지 않음
```

해결:

- ffmpeg 송출이 시작되어야 한다.
- MediaMTX 로그에 `is publishing to path 'cam1'`가 떠야 한다.

### `Address already in use`

의미:

```text
serve_mjpeg.py가 이미 8000 포트를 사용 중
```

확인:

```bash
ps aux | grep serve_mjpeg.py | grep -v grep
```

종료:

```bash
kill <PID>
kill -9 <PID>
```

### `real-time buffer too full`

의미:

```text
웹캠 입력 속도보다 ffmpeg 처리/전송 속도가 느림
```

대응:

- 해상도 낮춤: `320x240`
- 출력 FPS 낮춤: `-vf fps=10`
- 버퍼 증가: `-rtbufsize 512M`
- `-preset ultrafast`, `-tune zerolatency` 사용

### `http://localhost:18000/stream/camera-1`이 안 열림

확인 순서:

GPU PC:

```bash
curl http://localhost:8000/cameras
```

Windows:

```cmd
curl http://localhost:18000/cameras
```

판단:

```text
GPU PC localhost:8000 안 됨
  -> 브릿지 문제

GPU PC는 되는데 Windows localhost:18000 안 됨
  -> 8000 SSH 터널 문제

Windows localhost:18000/cameras는 되는데 stream만 안 됨
  -> cam1 publish 또는 브릿지 camera 상태 확인
```

## 모델/AI 추론 방향성

CCTV 관제 시스템 관점에서는 추론을 백엔드 API 서버에 직접 넣기보다 AI Worker로 분리하는 방향이 더 적합하다.

추천 구조:

```text
Camera/RTSP
  -> strange_ai AI Worker
  -> event JSON POST
  -> strange_back Backend
  -> REST/WebSocket
  -> strange_front Dashboard
```

역할 분리:

```text
Frontend
- 실시간 CCTV 화면
- 위험 이벤트 표시
- 관리자 확인/처리

Backend
- 카메라 목록 관리
- 이벤트 저장
- 알림 상태 관리
- 프론트 API 제공
- AI 추론 결과 수신/검증

AI Worker
- RTSP/Webcam 스트림 읽기
- 프레임 디코딩 및 버퍼링
- 모델 추론
- fall/faint/normal/crowd/fire/smoke 이벤트 판단
- backend로 JSON 이벤트 전송
```

영상과 이벤트 경로는 분리한다.

```text
실시간 영상:
Camera/RTSP -> AI Worker MJPEG/WebRTC -> Frontend

이벤트/알림:
AI Worker -> Backend -> Frontend
```

현재 단계 추천:

```text
1. 모델은 일단 strange_ai에 둔다.
2. 백엔드는 추론하지 말고 이벤트 수신 API를 먼저 만든다.
3. AI Worker가 추론 결과를 Backend로 POST한다.
4. Frontend는 영상은 AI stream, 이벤트는 Backend API/WebSocket에서 받는다.
5. CPU 추론이 충분히 빠르면 나중에 AI Worker를 GPU PC가 아니라 백엔드 서버 옆에 배포한다.
```

즉 지금 결론은 다음과 같다.

```text
CCTV 관제 목표라면 AI Worker 분리 구조가 맞다.
백엔드는 관제 상태/이벤트/저장을 책임진다.
AI Worker는 영상 처리와 추론을 책임진다.
프론트는 영상 스트림과 이벤트 상태를 조합해서 보여준다.
```
