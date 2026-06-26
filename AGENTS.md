# AGENTS.md - 멀티 에이전트 운영 지침

이 문서는 본 프로젝트에서 활동하는 모든 개발 에이전트(AI, Backend, Frontend, Infra/Docs, Integration)의 **역할 분담, 코드 권한 범위, 브랜치 규칙 및 안전 운영 수칙**을 명시합니다. 

모든 에이전트는 작업을 시작하기 전 이 문서를 읽고 지침을 완벽하게 따라야 합니다.

---

## 1. 에이전트 역할 및 수정 권한 범위

본 프로젝트는 Nested Git Repository 구조로 이루어져 있으며, 각 에이전트는 자신에게 할당된 리포지토리 폴더와 피처 브랜치에서만 작업을 수행해야 합니다.

| 에이전트 역할 | 담당 리포지토리 폴더 | 담당 피처 브랜치 | 수정 가능 범위 | 수정 절대 금지 범위 |
| :--- | :--- | :--- | :--- | :--- |
| **AI Agent** | `strange_ai/` | `codex/ai-worker-flow-improvements` | `strange_ai/` 내부 전체 및 루트의 AI 연동 스크립트 | 타 파트 폴더 (`strange_back/`, `strange_front/`, `strange_infra/`), `docs/` |
| **Backend Agent** | `strange_back/` | `codex/incident-acknowledge-recording` | `strange_back/` 내부 전체 | `strange_ai/`, `strange_front/`, `strange_infra/`, `docs/`, 루트 파일 |
| **Frontend Agent**| `strange_front/`| `codex/live-camera-streams` | `strange_front/` 내부 전체 | `strange_ai/`, `strange_back/`, `strange_infra/`, `docs/`, 루트 파일 |
| **Infra/Docs Agent**| `strange_infra/` | `codex/docker-frontend-backend-optimization` | `strange_infra/` 내부 전체 및 `docs/` 폴더 | `strange_ai/`, `strange_back/`, `strange_front/`, 루트 파일 (공통 계약 예외) |
| **Integration Agent**| 전체 워크스페이스 | `staging` (최종 병합용) | 워크스페이스 전체 (단, 병합 및 충돌 해결 목적에 한함) | 독자적인 기능 개발 및 개별 에이전트 전용 피처 브랜치 수정 |

---

## 2. 공통 준수 및 읽기 전용 파일

모든 에이전트는 개발을 시작하기 전 다음 파일을 **반드시 정독(Read-only)**해야 하며, 해당 파일들의 스펙에 부합하게 개발해야 합니다.
1. **`PROJECT_CONTRACT.md`**: 통신 프로토콜, DTO 필드, RTSP/HLS 경로 계약이 수록되어 있습니다.
2. **`README.md`**: 전반적인 로컬 구동 절차가 명시되어 있습니다.
3. **`AGENTS.md`** (본 문서): 본인의 작업 한계선을 파악하기 위해 필독해야 합니다.

---

## 3. 브랜치 전략 및 병합(Merge) 수칙

### 3.1. 피처 브랜치 독립성 유지
- 각 에이전트는 담당 브랜치 이외의 다른 브랜치로 임의로 `checkout`하거나 코드를 수정하지 않습니다.
- 개별 피처 브랜치(`codex/...`) 간에 **직접적인 merge는 엄격히 금지**됩니다. (예: Backend 에이전트가 AI 브랜치를 자신의 브랜치로 merge하는 행위 불가)

### 3.2. 최종 병합은 Integration 브랜치에서만 수행
- 모든 피처 병합 및 충돌 해결은 오직 **`staging` (Integration 에이전트의 담당 브랜치)**에서만 진행됩니다.
- 개별 에이전트는 피처 개발 완료 후, Integration 에이전트에게 병합을 요청(PR 생성 등)하고 대기해야 합니다.

---

## 4. 안전 운영 및 금지 수칙

1. **Destructive Command 금지**:
   - `git reset --hard`, `git clean -fd`, `git push --force`와 같은 이력 파괴성 명령어를 절대 실행하지 마십시오.
   - 대량의 파일 삭제나 코드 베이스 포맷터의 오남용을 금지합니다.
2. **비밀 정보 보호**:
   - `.env`, secrets, credentials, API key 등이 포함된 설정 파일을 직접 화면에 출력하거나, 원격에 push하거나, 코드에 하드코딩하지 마십시오.
3. **폴더 침범 금지**:
   - 에이전트 본인의 수정 가능 범위 이외의 영역을 절대 임의로 수정하여 커밋하지 마십시오. 공통 계약 변경이 필요할 경우, `PROJECT_CONTRACT.md` 변경 프로세스를 먼저 밟으십시오.


## /make-slide
Claude Code: when the user types `/make-slide`, read `.claude/skills/make-slide/SKILL.md` and follow the presentation creation workflow.

Codex: select `make-slide` from `/skills` or invoke it as `$make-slide`. The Codex wrapper is `.agents/skills/make-slide/SKILL.md`.

Browse themes at https://make-slide.vercel.app.
