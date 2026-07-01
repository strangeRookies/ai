# Camera Process Lifecycle & Cleanup Guide

이 문서는 실시간 CCTV 카메라 연동 시 생성되는 Python 및 FFmpeg 프로세스의 라이프사이클 관리 기준, 기대 프로세스 수, 그리고 수동 정리 가이드라인을 제공합니다.

---

## 1. 중복 프로세스 및 GPU 메모리 누수 원인 분석

이전 시스템에서 4개의 카메라만 작동시켰음에도 과도한 프로세스와 GPU 메모리 누수가 발생한 원인은 다음과 같습니다:

1. **포트 충돌 없는 백그라운드 중복 기동**:
   `serve_ai_overlay.py`는 기본적으로 MJPEG 스트리밍을 디버깅 용도로 노출하지 않을 때(Metadata-only mode, 기본값) HTTP 서버 소켓을 바인딩하지 않습니다. 이로 인해 동일한 카메라 ID로 프로세스를 중복 실행해도 포트 충돌(Address already in use) 에러가 발생하지 않고 여러 개의 프로세스가 동시에 기동되어 동일한 GPU 상에서 YOLO Pose 및 LSTM 행동 인퍼런스를 조용히 중복 연동하게 되었습니다.
2. **이전 세션의 FFmpeg 고립(Orphaned) 프로세스**:
   이전 실행 도중 parent 프로세스가 비정상 종료(SIGKILL 등) 되었을 때, 자식 FFmpeg 프로세스가 좀비 프로세스 또는 고립 프로세스로 유지되어 GPU 가속(nvenc) 또는 CPU를 계속 점유하게 되었습니다.
3. **가상환경/런타임 재기동 시 자동 정리 누락**:
   새로운 런타임 기동 시 기존에 돌고 있던 동일 포트/동일 스트림 주소의 FFmpeg 퍼블리셔가 정리되지 않은 채 새로운 퍼블리셔가 추가 기동하여 중복 스트리밍을 진행했습니다.

---

## 2. 4개 카메라 실행 시 기대 프로세스 수

4개의 카메라를 정상 기동했을 때의 정상 프로세스 구성 요건은 다음과 같습니다:

* **총 Python 프로세스 수**: **6개**
  * `run_registered_cameras.py` (마스터 러너): **1개**
  * `start_simulated_rtsp_from_folder.py` (시뮬레이터 퍼블리셔): **1개**
  * `serve_ai_overlay.py` (카메라별 분석 워커): **4개**
* **총 FFmpeg 프로세스 수**: **4개** (카메라별 퍼블리싱용 ffmpeg 4개)
* **총 GPU active 프로세스 수**: **4개**
  * 실질적인 GPU 인퍼런스를 도맡는 4개의 `serve_ai_overlay.py` 프로세스가 GPU 메모리(VRAM)를 각각 약 1개분씩 안정적으로 점유합니다.
  * (참고: `--ffmpeg-mode nvenc`로 기동 시 FFmpeg 가속 프로세스 4개가 추가로 GPU에 보일 수 있습니다.)

---

## 3. 프로세스 관리 및 자동화 구현 사항

1. **중앙 파일 레지스트리 도입 (`runs/camera_worker_registry.json`)**:
   모든 카메라 워커 및 FFmpeg 퍼블리셔 기동 시 프로세스 ID(PID), 기동 시간(startedAt), 입력 RTSP 주소(rtspUrl) 및 출력 타겟 주소(outputPath)를 등록합니다.
2. **psutil 기반의 중복 기동 원천 차단**:
   새로운 워커나 퍼블리셔 기동 시 레지스트리 및 OS 실제 활성 프로세스 커맨드를 추적합니다. 동일 카메라 ID 또는 동일 RTSP 출력 목적지로 작동 중인 이전 프로세스가 검출될 경우, 즉시 `terminate` -> `wait(timeout)` -> `kill` 단계적 시퀀스로 정리한 뒤 신규 프로세스를 안전하게 띄웁니다.
3. **Graceful Shutdown**:
   마스터 러너 종료 및 `SIGINT`/`SIGTERM` 수신 시 등록된 모든 자식 프로세스를 역순으로 확실하게 회수 및 Unregister 처리합니다.

---

## 4. 수동 정리 및 진단 가이드

시스템 오류나 비정상 강제 종료 등으로 인해 프로세스가 남아있을 우려가 있을 때 사용할 수 있는 수동 진단 및 정리 커맨드입니다.

### 4.1. 프로세스 기동 상태 진단
```bash
# 1. 런타임 관련 Python 및 FFmpeg 프로세스 상세 확인
ps -eo pid,ppid,cmd | grep -E "python|ffmpeg|run_registered|serve_ai|demo_streamer|start_simulated" | grep -v grep

# 2. FFmpeg 프로세스만 빠르게 확인
pgrep -af ffmpeg

# 3. GPU 점유 상태 확인
nvidia-smi
```

### 4.2. 수동 강제 정리 명령어 (일괄 Kill)
```bash
# 모든 시뮬레이션 ffmpeg 퍼블리셔 및 AI 워커 강제 종료
pkill -f 'serve_ai_overlay.py'
pkill -f 'start_simulated_rtsp_from_folder.py'
pkill -f 'run_registered_cameras.py'
pkill -f 'demo_streamer.py'
pkill -9 -f 'ffmpeg'

# 레지스트리 캐시 파일 초기화
rm -f runs/camera_worker_registry.json
```
---

## 2026-07-01 ffmpeg mode baseline

Default operational mode is now `--ffmpeg-mode auto` or `FFMPEG_MODE=auto`.
`auto` checks NVENC availability first, uses NVENC only when available, and falls back toward `copy` / CPU encoding when failures repeat.

Use `--ffmpeg-mode copy` as a temporary safe mode while diagnosing NVENC instability.
Use `--ffmpeg-mode nvenc` only for explicit NVENC testing, not as the normal launcher default.
