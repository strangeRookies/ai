# Windows .bat & GPU Server AI Process Lifecycle Guide

이 문서는 프로젝트 내 Windows `.bat` 스크립트의 역할, 중복 실행 방지 기능, AI/RTSP/ffmpeg/python 프로세스의 안전한 수동/자동 수명 주기 관리 및 상태 점검 방법을 안내합니다.

---

## 1. 배치 파일 목록 및 역할

| 배치 파일 명 | 위치 | 역할 | 관리 상태 |
| :--- | :--- | :--- | :--- |
| **`AI_실행_딸깍.bat`** | 루트 (`/`) | GPU 서버 최신 코드 동기화(develop 브랜치), SSH 터널 개설, MediaMTX/시뮬레이터/AI 러너 기동을 지원하는 공식 통합 런처 | **Active (공식)** |
| **`status_ai_processes.bat`** | 루트 (`/`) | 로컬/원격 포트 점검, 동작 중인 로컬/원격 python/ffmpeg 프로세스 조회, 액티브 워커 레지스트리 및 GPU 점유 현황 통합 진단 스크립트 | **Active (신규)** |
| **`cleanup_ai_processes.bat`** | 루트 (`/`) | 로컬 SSH 터널 제거, GPU 서버 상의 AI 워커/FFmpeg/RTSP 퍼블리셔/MediaMTX 프로세스 점진적 안전 종료 스크립트 (`--dry-run` 지원) | **Active (신규)** |
| `AI_실행_딸깍.bat` | `strange_ai/` | 하브 하드코딩된 구버전 실행기 | **Deprecated (사용 금지)** |
| `gradlew.bat` | `strange_back/` | 백엔드 빌드/실행용 Gradle 래퍼 | **Active (백엔드 전용)** |

> [!WARNING]
> 반드시 루트의 `AI_실행_딸깍.bat`를 사용하십시오. `strange_ai/` 내부의 파일은 중복 실행 방지 및 최신 안정화 터미널 코드가 누락된 예전 버전이므로 사용하지 마십시오.

---

## 2. 안전한 기동 및 종료 흐름

```mermaid
graph TD
    A[기동 전: status_ai_processes.bat 로 상태 확인] --> B{포트 충돌/중복 실행?}
    B -- 예 --> C[cleanup_ai_processes.bat 실행]
    B -- 아니오 --> D[AI_실행_딸깍.bat 실행]
    D --> E["1. [1/5] 사전 상태 점검"]
    E --> F["2. [2/5] GPU 서버 코드 동기화"]
    F --> G["3. [3/5] SSH 터널 창 실행"]
    G --> H["4. [4/5] 원격 AI 서비스 실행"]
    H --> I["5. [5/5] 상태 점검"]
```

### 2.1. 중복 실행 및 포트 충돌 방지 로직 (로컬 Windows)
`AI_실행_딸깍.bat` 기동 시, 로컬 포트 `8888, 8889, 8189, 18080`이 이미 활성화되어 있는지 점검합니다.
활성화되어 있다면 기존 터널이 열려 있다는 경고와 함께 **자동 정리 여부(Y/N)**를 묻습니다. `Y` 입력 시 자동으로 `cleanup_ai_processes.bat`를 호출하여 이전 좀비 터널을 말끔히 청소한 뒤 안전하게 세션을 다시 기동합니다.

### 2.2. 역방향 백엔드 통신 확인 및 유연한 에러 처리
비밀번호 입력 딜레이로 포트 포워딩 연결이 늦게 완료될 경우, 스크립트가 강제 종료되지 않도록 **Retry / Skip / Exit** 메뉴를 제공합니다. 사용자는 터널 창에 비밀번호 입력을 마친 후 `[1] Retry`를 눌러 다시 통신 유효성을 확인할 수 있습니다.

---

## 3. 카메라 4개 구동 시 정상 프로세스 현황

4개 카메라 등록 기준, 시스템에서 기동되어야 하는 프로세스의 정상 범위는 다음과 같습니다.

### 3.1. 원격 GPU 서버 프로세스 (Linux)
* **Python 프로세스 (총 6개)**:
  * `start_simulated_rtsp_from_folder.py` (시뮬레이터 퍼블리셔) - **1개**
  * `run_registered_cameras.py` (카메라 관리 러너) - **1개**
  * `serve_ai_overlay.py` (카메라별 분석 워커) - **4개**
* **FFmpeg 프로세스 (총 4개)**:
  * 시뮬레이터에서 4개 RTSP 채널을 MediaMTX로 송출하는 ffmpeg 인코더 - **4개**
* **Docker 프로세스 (총 1개)**:
  * `mediamtx` 컨테이너 - **1개**

### 3.2. 로컬 Windows 프로세스
* **SSH 프로세스 (총 2개)**:
  * 백그라운드용 또는 터널 윈도우 창용 `ssh.exe` - **2개**
* **포트 포워딩 포트**:
  * `8888` (HLS), `8889` (WebRTC), `8189` (MediaMTX UI), `18080` (백엔드 리버스 프록시)

---

## 4. 수동 점검 및 트러블슈팅 커맨드

### 4.1. 로컬 Windows 검증 명령어
* **실행 중인 프로세스 조회**:
  ```cmd
  tasklist | findstr /i "ssh.exe java.exe node.exe"
  ```
* **점유 중인 포트 조회**:
  ```cmd
  netstat -ano | findstr /R "LISTENING.*:8888 LISTENING.*:8889 LISTENING.*:18080"
  ```

### 4.2. 원격 GPU 서버 검증 명령어
* **프로젝트 관련 AI 프로세스 조회**:
  ```bash
  ps -eo pid,ppid,cmd | grep -E "python|ffmpeg|run_registered|serve_ai|start_simulated|mediamtx" | grep -v grep
  ```
* **GPU 메모리 점유 및 Compute 프로세스 조회**:
  ```bash
  nvidia-smi
  ```
* **포트 리스너 현황 조회**:
  ```bash
  ss -tlnp | grep -E "8554|8888|8889|8189|18080"
  ```

---

## 5. 종료 스크립트 사용법

모든 AI 프로세스와 SSH 터널을 종료하고 리소스를 반환하려면 아래 스크립트를 사용합니다.

### 5.1. 드라이런 (종료 대상 프로세스 목록 미리보기)
실제 프로세스를 죽이지 않고 어떤 프로세스가 정리 대상인지 안전하게 조회합니다.
```cmd
cleanup_ai_processes.bat --dry-run
```

### 5.2. 실제 프로세스 정리 실행
종료 순서에 따라 원격 AI 워커 -> 원격 시뮬레이터 -> 원격 MediaMTX -> 로컬 터널 순으로 점진적이고 안전하게(SIGTERM -> 대기 -> SIGKILL) 리소스를 종료합니다.
```cmd
cleanup_ai_processes.bat
```
