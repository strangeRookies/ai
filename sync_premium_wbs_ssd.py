import json
import os
import sys
import requests
from requests.auth import HTTPBasicAuth

# Windows 콘솔 유니코드 인코딩 설정 (출력 에러 방지)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# 1. Jira 크리덴셜 및 설정 로드
CONFIG_PATH = r"C:\Users\user\.gemini\antigravity\mcp_config.json"
if not os.path.exists(CONFIG_PATH):
    print(f"[Error] Config file not found at {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

jira_conf = config["mcpServers"]["jira-community"]["env"]
JIRA_URL = jira_conf["JIRA_URL"].rstrip("/")
JIRA_EMAIL = jira_conf["JIRA_EMAIL"]
JIRA_API_TOKEN = jira_conf["JIRA_API_TOKEN"]
auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

PROJECT_KEY = "SSD"

# 2. 기존 백로그 정리 (삭제)
print("[1/4] Cleaning up previous project keys...")
for i in range(1, 150):
    issue_key = f"{PROJECT_KEY}-{i}"
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}"
    try:
        res = requests.delete(url, auth=auth, headers=headers)
    except Exception:
        pass

# 3. 4대 에픽 및 32개 균형 태스크 구조 정의
wbs_data = [
    # M1: 기반 환경 구축
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_infra",
        "code": "M1-INF-01",
        "task": "EMQX MQTT 브로커 Docker 클러스터 배포 및 토픽/보안(ACL) 규칙 기본 설계",
        "deliverable": "emqx.conf 및 docker-compose.infra.yml 파일",
        "method": "인프라",
        "start": "2026-05-26",
        "end": "2026-05-29",
        "epic_idx": 1 # 0: AI, 1: Infra, 2: Back, 3: Front
    },
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_infra",
        "code": "M1-INF-02",
        "task": "로컬 Edge 단말 및 개발 서버 간의 포트포워딩, DDNS 및 VPN 내부망 가상 네트워크 구동",
        "deliverable": "네트워크 가상 망 구성 설계서",
        "method": "네트워크",
        "start": "2026-05-30",
        "end": "2026-06-01",
        "epic_idx": 1
    },
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_ai",
        "code": "M1-AI-01",
        "task": "Jetson Orin 하드웨어 가속기(TensorRT) 연동 및 YOLOv8-Pose 최적화 환경 구축",
        "deliverable": "TensorRT Engine 빌드 및 배포 스크립트",
        "method": "AI개발",
        "start": "2026-05-26",
        "end": "2026-06-01",
        "epic_idx": 0
    },
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_ai",
        "code": "M1-AI-02",
        "task": "RTSP 카메라 스트림 수신 및 OpenCV/GStreamer 하드웨어 디코딩 프레임 캡처 모듈 구현",
        "deliverable": "스트리밍 프레임 캡처 소스코드",
        "method": "AI개발",
        "start": "2026-06-02",
        "end": "2026-06-08",
        "epic_idx": 0
    },
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_back",
        "code": "M1-BAC-01",
        "task": "Spring Boot 3.3 기반 아키텍처 초기화, 멀티 모듈 및 JPA/QueryDSL 기본 인프라 구조 설정",
        "deliverable": "build.gradle 및 Core 도메인 소스코드",
        "method": "백엔드",
        "start": "2026-05-26",
        "end": "2026-06-01",
        "epic_idx": 2
    },
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_back",
        "code": "M1-BAC-02",
        "task": "Spring Security 및 JWT 기반 역할별(점주, 보안요원, 유관기관) 회원가입/인증 프로세스 설계",
        "deliverable": "Security Config 및 사용자 인증 API",
        "method": "보안",
        "start": "2026-06-02",
        "end": "2026-06-08",
        "epic_idx": 2
    },
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_front",
        "code": "M1-FRO-01",
        "task": "Next.js 14 / Vite 기반 UI 보일러플레이트 세팅 및 글로벌 디자인 시스템(글래스모피즘, 다크모드) 토큰 정의",
        "deliverable": "tailwind.config.js 및 index.css 테마 파일",
        "method": "프론트",
        "start": "2026-05-26",
        "end": "2026-06-01",
        "epic_idx": 3
    },
    {
        "milestone": "M1: 기반 환경 구축",
        "track": "strange_front",
        "code": "M1-FRO-02",
        "task": "WebSocket STOMP 프로토콜 연동 및 기본 클라이언트 커넥션 헬스체크 구현",
        "deliverable": "WebSocket Context Provider 컴포넌트",
        "method": "통신",
        "start": "2026-06-02",
        "end": "2026-06-08",
        "epic_idx": 3
    },

    # M2: 핵심 기능 개발
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_ai",
        "code": "M2-AI-01",
        "task": "ByteTrack 기반 다중 객체 추적(MOT) 및 실시간 ID 일시 소실 보정을 위한 Identity Stitching 알고리즘 개발",
        "deliverable": "ByteTrack 커스텀 래퍼 클래스 및 Stitching 소스코드",
        "method": "AI개발",
        "start": "2026-06-09",
        "end": "2026-06-19",
        "epic_idx": 0
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_ai",
        "code": "M2-AI-02",
        "task": "LSTM/ST-GCN 기반 3대 위협 행동(낙상, 실신, 폭행) 기하 수학 엔진 및 실시간 분류 파이프라인 개발",
        "deliverable": "행동 분류 모델(ST-GCN) 파일 및 수학 엔진 모듈",
        "method": "AI모델",
        "start": "2026-06-20",
        "end": "2026-06-30",
        "epic_idx": 0
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_infra",
        "code": "M2-INF-01",
        "task": "PostgreSQL 관계형 DB 및 Redis 분산 캐시 DB 도커 컨테이너화 및 커넥션 풀 최적화",
        "deliverable": "docker-compose.db.yml 및 DB 스키마 DDL",
        "method": "데이터베이스",
        "start": "2026-06-09",
        "end": "2026-06-19",
        "epic_idx": 1
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_infra",
        "code": "M2-INF-02",
        "task": "Github Actions 기반의 4개 마이크로 레포지토리 빌드 자동화 및 Edge 단말 CD 파이프라인 구축",
        "deliverable": "github-actions.yml 워크플로우 명세서",
        "method": "CI/CD",
        "start": "2026-06-20",
        "end": "2026-06-30",
        "epic_idx": 1
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_back",
        "code": "M2-BAC-01",
        "task": "Spring WebFlux 리액티브 MQTT 리스너 구축 및 비동기 수신 이벤트 파이프라인 최적화",
        "deliverable": "MQTT Inbound Adapter 및 리스너 클래스",
        "method": "백엔드",
        "start": "2026-06-09",
        "end": "2026-06-15",
        "epic_idx": 2
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_back",
        "code": "M2-BAC-02",
        "task": "Redis 분산 캐시 기반 실시간 이벤트 디바운싱(TTL 30초 내 중복 알림 필터링) 및 에러 격리 처리 개발",
        "deliverable": "Debouncer Service 및 Redis 캐싱 로직",
        "method": "백엔드",
        "start": "2026-06-16",
        "end": "2026-06-22",
        "epic_idx": 2
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_back",
        "code": "M2-BAC-03",
        "task": "Firebase Cloud Messaging(FCM) HTTP v1 비동기 알림 전송 API 및 디바이스 토큰 관리 서비스 구현",
        "deliverable": "FCM Sender Service 및 토큰 관리 API",
        "method": "알림",
        "start": "2026-06-23",
        "end": "2026-06-30",
        "epic_idx": 2
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_front",
        "code": "M2-FRO-01",
        "task": "실시간 N분할 관제 캔버스 레이아웃 구현 및 웹소켓 데이터 스트림 오버레이 렌더링",
        "deliverable": "Canvas Grid 관제 컴포넌트",
        "method": "프론트",
        "start": "2026-06-09",
        "end": "2026-06-19",
        "epic_idx": 3
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_front",
        "code": "M2-FRO-02",
        "task": "위협 감지 시각적 토스트(알림), 사이렌 경보 사운드 플레이어 및 실시간 사이드바 경보 이력 컴포넌트 개발",
        "deliverable": "Alert Toast & Sidebar 컴포넌트",
        "method": "프론트",
        "start": "2026-06-20",
        "end": "2026-06-25",
        "epic_idx": 3
    },
    {
        "milestone": "M2: 핵심 기능 개발",
        "track": "strange_front",
        "code": "M2-FRO-03",
        "task": "모바일 웹뷰 대응 모바일 푸시 알림 터치 시 진입하는 랜딩/대처 가이드 상세 UI 페이지 구축",
        "deliverable": "Mobile Detail View 및 대처 가이드 UI",
        "method": "프론트",
        "start": "2026-06-26",
        "end": "2026-06-30",
        "epic_idx": 3
    },

    # M3: 기능 통합 검토
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_ai",
        "code": "M3-AI-01",
        "task": "NumPy 슬라이딩 윈도우 FIFO 메모리 큐 구현 및 10초 임시 프레임 스택 아카이빙 기능 연동",
        "deliverable": "Memory Buffer Module 소스코드",
        "method": "AI개발",
        "start": "2026-07-01",
        "end": "2026-07-07",
        "epic_idx": 0
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_ai",
        "code": "M3-AI-02",
        "task": "Edge 단말 추론 리소스 점검 및 프레임 드랍(FPS Jitter) 보정 필터(Kalman Filter) 고도화",
        "deliverable": "엣지 PC 추론 성능 평가 리포트",
        "method": "최적화",
        "start": "2026-07-08",
        "end": "2026-07-14",
        "epic_idx": 0
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_infra",
        "code": "M3-INF-01",
        "task": "Prometheus + Grafana 통합 인프라 메트릭(CPU, GPU, RAM, MQTT 커넥션) 대시보드 구축",
        "deliverable": "grafana_dashboard.json 및 데이터 연동 설정",
        "method": "모니터링",
        "start": "2026-07-01",
        "end": "2026-07-07",
        "epic_idx": 1
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_infra",
        "code": "M3-INF-02",
        "task": "Loki / Promtail 로그 수집 인프라 구축을 통한 4대 컴포넌트 실시간 로그 통합 모니터링 세팅",
        "deliverable": "docker-compose.monitoring.yml 명세서",
        "method": "로그수집",
        "start": "2026-07-08",
        "end": "2026-07-14",
        "epic_idx": 1
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_back",
        "code": "M3-BAC-01",
        "task": "AWS SDK 연동을 통한 FFmpeg 10초 증거 비디오 MP4 트랜스코딩 및 AWS S3 업로드, 90일 만료 수명주기 설정",
        "deliverable": "S3 Archiving Service 및 Lifecycle 설정 파일",
        "method": "백엔드",
        "start": "2026-07-01",
        "end": "2026-07-07",
        "epic_idx": 2
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_back",
        "code": "M3-BAC-02",
        "task": "날짜, 위협 유형, 카메라 ID 등 조건별 다차원 아카이브 이력 조회 및 비디오 다운로드 REST API 구현",
        "deliverable": "Archive Search Controller API 소스코드",
        "method": "API개발",
        "start": "2026-07-08",
        "end": "2026-07-14",
        "epic_idx": 2
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_front",
        "code": "M3-FRO-01",
        "task": "위험 이벤트 발생 위치 실시간 표기를 위한 2D CAD 매장 평면도 SVG 매핑 및 동적 애니메이션 핀 렌더링",
        "deliverable": "SVG Floorplan 동적 지도 컴포넌트",
        "method": "프론트",
        "start": "2026-07-01",
        "end": "2026-07-06",
        "epic_idx": 3
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_front",
        "code": "M3-FRO-02",
        "task": "Daum 주소 OpenAPI 및 행정안전부/유관기관 출동 시스템 모킹 가상 강제 에스컬레이션(112/119 Webhook) 호출 인터페이스",
        "deliverable": "Escalation API 연동 소스코드 및 버튼 컨트롤러",
        "method": "연동",
        "start": "2026-07-07",
        "end": "2026-07-10",
        "epic_idx": 3
    },
    {
        "milestone": "M3: 기능 통합 검토",
        "track": "strange_front",
        "code": "M3-FRO-03",
        "task": "아카이브 클라우드 동영상 플레이어 및 타임라인 탐색 슬라이더를 포함한 조회 대시보드 개발",
        "deliverable": "Video Player Dashboard 조회 컴포넌트",
        "method": "프론트",
        "start": "2026-07-11",
        "end": "2026-07-14",
        "epic_idx": 3
    },

    # M4: 1차 프로토타입 릴리즈
    {
        "milestone": "M4: 1차 프로토타입 릴리즈",
        "track": "strange_ai",
        "code": "M4-AI-01",
        "task": "Edge PC 환경에서 실감형 3대 시나리오(낙상, 실신, 폭행) 시뮬레이션 기반 추론 정확도 및 FPS 성능 최종 리포트 작성",
        "deliverable": "AI 모델 최종 검증 및 성능 평증 리포트",
        "method": "보고서",
        "start": "2026-07-15",
        "end": "2026-07-21",
        "epic_idx": 0
    },
    {
        "milestone": "M4: 1차 프로토타입 릴리즈",
        "track": "strange_infra",
        "code": "M4-INF-01",
        "task": "Edge AI - 백엔드 - 프론트엔드 - DB - MQTT를 포함한 전체 Docker Compose 네트워크 통합 배포 및 TLS 보안 검증",
        "deliverable": "production.yml 배포 명세서 및 SSL 인증 설정",
        "method": "배포",
        "start": "2026-07-15",
        "end": "2026-07-18",
        "epic_idx": 1
    },
    {
        "milestone": "M4: 1차 프로토타입 릴리즈",
        "track": "strange_infra",
        "code": "M4-INF-02",
        "task": "통합 E2E 테스트 자동화 스크립트 실행 및 전체 시스템의 DoD(완료 정의) 최종 확인 모니터링",
        "deliverable": "E2E 통합 테스트 검증 보고서",
        "method": "테스트",
        "start": "2026-07-19",
        "end": "2026-07-21",
        "epic_idx": 1
    },
    {
        "milestone": "M4: 1차 프로토타입 릴리즈",
        "track": "strange_back",
        "code": "M4-BAC-01",
        "task": "대량 이벤트 유입 시나리오 대비 백엔드 스트레스 테스트(JMeter) 진행 및 부하 분산(Rate Limiter) 설계 검증",
        "deliverable": "JMeter 부하 테스트 결과 및 처리성능 보고서",
        "method": "테스트",
        "start": "2026-07-15",
        "end": "2026-07-21",
        "epic_idx": 2
    },
    {
        "milestone": "M4: 1차 프로토타입 릴리즈",
        "track": "strange_front",
        "code": "M4-FRO-01",
        "task": "최종 사용자 시나리오(모바일 알림 -> 대시보드 동기화 -> 2D 맵 핀 확인 -> 119 출동 에스컬레이션)의 최종 사용성 평가 검증",
        "deliverable": "최종 UI/UX 정밀도 및 사용성 검토 보고서",
        "method": "테스트",
        "start": "2026-07-15",
        "end": "2026-07-21",
        "epic_idx": 3
    }
]

# 4. Jira 에픽 생성 정의 (4대 트랙별)
epics_to_create = [
    {
        "summary": "[AI][M2] 엣지 AI 비전 분석 파이프라인 구축",
        "description": "Jetson Orin 하드웨어 가속기(TensorRT) 기반의 실시간 다채널 비디오 디코딩, YOLO Pose 기반 관절 특징점 추출, LSTM/ST-GCN 시계열 행동 분석 및 Identity Stitching을 포함한 코어 비전 AI 분석 인프라를 마련합니다.",
        "track": "strange_ai"
    },
    {
        "summary": "[인프라][M1] 클라우드 및 엣지 인프라 환경 구축",
        "track": "strange_infra",
        "description": "EMQX MQTT 브로커 Docker 클러스터 배포, PostgreSQL 및 Redis 분산 캐시 DB 구축, Github Actions 자동 빌드 및 배포(CI/CD) 파이프라인, Prometheus/Grafana 및 Loki/Promtail 연계 모니터링 및 전체 통합 Docker Compose 환경을 수립합니다."
    },
    {
        "summary": "[Back][M2] 핵심 이벤트 처리 및 이원화 라이프사이클 백엔드 구축",
        "track": "strange_back",
        "description": "Spring Boot 3.3 기반 아키텍처 초기화, Spring Security/JWT 회원 및 보안 아키텍처, WebFlux 리액티브 MQTT 리스너, Redis 분산락/디바운싱, Google FCM HTTP v1 비동기 Push 알림 전송 API, AWS S3 다중 스토리지 동영상 MP4 아카이빙 및 조회 조건 API를 완수합니다."
    },
    {
        "summary": "[Front][M3] 실시간 통합 관제 및 아카이빙 대시보드 연동",
        "track": "strange_front",
        "description": "React/Next.js 14 기반 HUD 디자인 시스템 구축, WebSocket STOMP 연동 헬스체크, N분할 실시간 비디오 렌더링(Canvas), 위협 경보 Toast 알림, 모바일 웹뷰 대응 푸시 랜딩 및 2D SVG 매장 평면도 맵 동적 핀 맵, OpenAPI 외부 유관기관 에스컬레이션 112/119 호출 및 아카이브 타임라인 플레이어를 완료합니다."
    }
]

created_epic_keys = []

print("[2/4] Creating 4 Core Epics in Jira...")
for i, epic in enumerate(epics_to_create):
    url = f"{JIRA_URL}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": PROJECT_KEY},
            "summary": epic["summary"],
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": epic["description"]}]
                    }
                ]
            },
            "issuetype": {"name": "Epic"}  # fallback check
        }
    }
    # JWM template might require using standard Issue Types like Epic, or standard Task. We try standard Epic.
    # In some JWM instances, "Epic" is named "Epic" or "Task". We first try Epic.
    res = requests.post(url, auth=auth, headers=headers, json=payload)
    if res.status_code == 201:
        epic_key = res.json()["key"]
        print(f" -> Created Epic {epic_key}: {epic['summary']}")
        created_epic_keys.append(epic_key)
    else:
        # Fallback to Task if Epic is not supported as an issue type
        print(f" -> Failed to create Epic '{epic['summary']}' (Status {res.status_code}). Retrying as Task...")
        payload["fields"]["issuetype"] = {"name": "Task"}
        res2 = requests.post(url, auth=auth, headers=headers, json=payload)
        if res2.status_code == 201:
            epic_key = res2.json()["key"]
            print(f" -> Created pseudo-Epic Task {epic_key}: {epic['summary']}")
            created_epic_keys.append(epic_key)
        else:
            print(f" -> Retry Failed! {res2.status_code}: {res2.text}")
            created_epic_keys.append(None)

# 5. 32개 태스크 생성 및 에픽에 바인딩
print("\n[3/4] Creating 32 Balanced WBS Tasks in Jira...")
jira_keys = {}

for task in wbs_data:
    epic_idx = task["epic_idx"]
    epic_key = created_epic_keys[epic_idx] if epic_idx < len(created_epic_keys) else None
    
    # 트랙 접두사 부여
    track_labels = {
        "strange_ai": "[AI]",
        "strange_infra": "[인프라]",
        "strange_back": "[Back]",
        "strange_front": "[Front]"
    }
    prefix = track_labels.get(task["track"], "")
    milestone_tag = f"[{task['milestone'].split(':')[0]}]"
    summary = f"{prefix}{milestone_tag} {task['task']}"
    
    desc_text = (
        f"세부 업무: {task['task']}\n"
        f"트랙: {task['track']}\n"
        f"마일스톤 단계: {task['milestone']}\n"
        f"수행 기간: {task['start']} ~ {task['end']}\n"
        f"기대 산출물: {task['deliverable']}\n"
        f"담당 분야/방법: {task['method']}"
    )
    
    payload = {
        "fields": {
            "project": {"key": PROJECT_KEY},
            "summary": summary[:255], # Jira summary limit safety
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": desc_text}]
                    }
                ]
            },
            "issuetype": {"name": "Task"}
        }
    }
    
    if epic_key:
        payload["fields"]["parent"] = {"key": epic_key}
        
    url = f"{JIRA_URL}/rest/api/3/issue"
    res = requests.post(url, auth=auth, headers=headers, json=payload)
    if res.status_code == 201:
        task_key = res.json()["key"]
        print(f" -> Created Task {task_key}: {summary}")
        jira_keys[task["code"]] = task_key
    else:
        print(f" -> Failed to create Task '{summary}' (Status {res.status_code}): {res.text}")
        jira_keys[task["code"]] = "N/A"

# 6. WBS CSV 및 HTML 재생성
print("\n[4/4] Rendering new balanced project_wbs.csv & project_wbs.html...")

# 실행 스크립트 위치 기준으로 저장 경로 동적 탐색
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "project_wbs.csv")
html_path = os.path.join(current_dir, "project_wbs.html")
try:
    with open(csv_path, "w", encoding="utf-8-sig") as csv_file:

        csv_file.write("마일스톤,트랙 / 저장소,WBS 코드,세부 업무 (Task),산출물 (Deliverables),수행 방법 (Method),시작일,종료일,진행률,Jira 티켓 연동\n")
        
        last_m = ""
        last_t = ""
        for task in wbs_data:
            m_val = task["milestone"] if task["milestone"] != last_m else ""
            t_val = task["track"] if task["track"] != last_t else ""
            
            # CSV 특수문자 이스케이프
            q_m = f'"{m_val}"'
            q_t = f'"{t_val}"'
            q_c = f'"{task["code"]}"'
            q_tk = f'"{task["task"]}"'
            q_d = f'"{task["deliverable"]}"'
            q_mt = f'"{task["method"]}"'
            q_s = f'"{task["start"]}"'
            q_e = f'"{task["end"]}"'
            q_p = '"0%"'
            q_j = f'"{jira_keys.get(task["code"], "N/A")}"'
            
            csv_file.write(f"{q_m},{q_t},{q_c},{q_tk},{q_d},{q_mt},{q_s},{q_e},{q_p},{q_j}\n")
            
            last_m = task["milestone"]
            # 리셋 처리 없이 이전 로우와 묶음으로 처리하도록 유지
    print(f" -> Successfully wrote balanced CSV to {csv_path}")
except Exception as e:
    print(f" -> Failed to write CSV: {str(e)}")

# HTML 생성
html_path = os.path.join(current_dir, "project_wbs.html")


html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 비전 보안 시스템 WBS 고도화 대시보드</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-cyan: #06b6d4;
            --accent-purple: #a855f7;
            --track-ai: #ef4444;
            --track-infra: #3b82f6;
            --track-back: #10b981;
            --track-front: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 10% 20%, rgba(6, 182, 212, 0.05) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(168, 85, 247, 0.05) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            font-family: 'Outfit', 'Inter', 'Noto Sans KR', sans-serif;
            min-height: 100vh;
            padding: 2.5rem 1.5rem;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }

        .logo-section h1 {
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em;
        }

        .logo-section p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-top: 0.25rem;
        }

        .action-bar {
            display: flex;
            gap: 0.75rem;
        }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }

        .btn-primary {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(168, 85, 247, 0.2));
            border-color: rgba(6, 182, 212, 0.4);
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, rgba(6, 182, 212, 0.3), rgba(168, 85, 247, 0.3));
            border-color: rgba(6, 182, 212, 0.6);
            box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
        }

        .table-container {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.875rem;
        }

        th {
            background: rgba(255, 255, 255, 0.02);
            padding: 1rem 1.25rem;
            color: var(--text-primary);
            font-weight: 700;
            border-bottom: 2px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        td {
            padding: 1rem 1.25rem;
            border-bottom: 1px solid var(--border-color);
            vertical-align: middle;
            color: #d1d5db;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.015);
        }

        .milestone-cell {
            font-weight: 700;
            color: var(--text-primary);
            font-size: 0.9rem;
            background: rgba(255, 255, 255, 0.01);
            border-right: 1px solid var(--border-color);
            white-space: nowrap;
        }

        .track-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .track-ai {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .track-infra {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .track-back {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .track-front {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .code-cell {
            font-family: monospace;
            font-weight: 600;
            color: var(--accent-cyan);
        }

        .task-cell {
            font-weight: 500;
            color: var(--text-primary);
            max-width: 320px;
        }

        .deliverable-cell {
            color: var(--text-secondary);
            font-size: 0.825rem;
            max-width: 250px;
        }

        .method-badge {
            background: rgba(255, 255, 255, 0.06);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
            color: var(--text-primary);
            font-weight: 500;
        }

        .date-cell {
            font-family: monospace;
            font-size: 0.8rem;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
            display: inline-block;
            margin-right: 0.5rem;
            vertical-align: middle;
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
        }

        .jira-link {
            color: var(--accent-cyan);
            text-decoration: none;
            font-weight: 600;
            border-bottom: 1px dashed var(--accent-cyan);
            transition: all 0.1s;
        }

        .jira-link:hover {
            color: #22d3ee;
            border-bottom-style: solid;
        }

        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: rgba(17, 24, 39, 0.9);
            color: #fff;
            border: 1px solid var(--accent-cyan);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            backdrop-filter: blur(8px);
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }

        .toast.show {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-section">
                <h1>이상행 보안 시스템 WBS 고도화 대시보드</h1>
                <p>AI · 인프라 · Backend · Frontend 4대 트랙의 동등배분 및 100% Jira 동기화 리액티브 로드맵</p>
            </div>
            <div class="action-bar">
                <button class="btn btn-primary" onclick="copyTableToClipboard()">
                    📋 WBS 복사하기
                </button>
            </div>
        </header>

        <div class="table-container">
            <table id="wbs-table">
                <thead>
                    <tr>
                        <th>마일스톤</th>
                        <th>트랙</th>
                        <th>WBS 코드</th>
                        <th>세부 업무 (Task)</th>
                        <th>산출물 (Deliverables)</th>
                        <th>구분</th>
                        <th>시작일</th>
                        <th>종료일</th>
                        <th>진행률</th>
                        <th>Jira Key</th>
                    </tr>
                </thead>
                <tbody>
"""

# HTML 행 렌더링
last_milestone = ""
for task in wbs_data:
    m_display = ""
    # 마일스톤 행 합치기를 위해 첫 출력 시만 표시
    if task["milestone"] != last_milestone:
        m_display = task["milestone"]
        last_milestone = task["milestone"]
    
    # 트랙별 배색 및 이름
    track_clean_names = {
        "strange_ai": "strange_ai (AI)",
        "strange_infra": "strange_infra (인프라)",
        "strange_back": "strange_back (Back)",
        "strange_front": "strange_front (Front)"
    }
    t_name = track_clean_names.get(task["track"], task["track"])
    t_class = task["track"].replace("strange_", "track-")
    
    j_key = jira_keys.get(task["code"], "N/A")
    j_link = f'<a href="{JIRA_URL}/browse/{j_key}" target="_blank" class="jira-link">{j_key}</a>' if j_key != "N/A" else "N/A"
    
    html_content += f"""                    <tr>
                        <td class="milestone-cell">{m_display}</td>
                        <td><span class="track-badge {t_class}">{t_name}</span></td>
                        <td class="code-cell">{task["code"]}</td>
                        <td class="task-cell">{task["task"]}</td>
                        <td class="deliverable-cell">{task["deliverable"]}</td>
                        <td><span class="method-badge">{task["method"]}</span></td>
                        <td class="date-cell">{task["start"]}</td>
                        <td class="date-cell">{task["end"]}</td>
                        <td style="white-space: nowrap;">
                            <div class="progress-bar"><div class="progress-fill" style="width: 0%;"></div></div>
                            <span style="font-size:0.75rem; color:var(--text-secondary);">0%</span>
                        </td>
                        <td>{j_link}</td>
                    </tr>
"""

html_content += """                </tbody>
            </table>
        </div>
    </div>

    <div id="toast-msg" class="toast">WBS 테이블이 클립보드에 복사되었습니다! Word/Sheets 에 그대로 붙여넣을 수 있습니다.</div>

    <script>
        function copyTableToClipboard() {
            const table = document.getElementById("wbs-table");
            const range = document.createRange();
            range.selectNode(table);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            
            try {
                const successful = document.execCommand('copy');
                if(successful) {
                    showToast();
                }
            } catch (err) {
                console.error("클립보드 복사 실패: ", err);
            }
            window.getSelection().removeAllRanges();
        }

        function showToast() {
            const toast = document.getElementById("toast-msg");
            toast.classList.add("show");
            setTimeout(() => {
                toast.classList.remove("show");
            }, 3000);
        }
    </script>
</body>
</html>
"""

try:
    with open(html_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
    print(f" -> Successfully wrote balanced HTML dashboard to {html_path}")
except Exception as e:
    print(f" -> Failed to write HTML: {str(e)}")

print("\n[OK] Premium WBS Synchronization Pipeline completed successfully!")
sys.exit(0)
