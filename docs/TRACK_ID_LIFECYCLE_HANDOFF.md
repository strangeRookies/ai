# Track ID lifecycle 진단 핸드오프 (이어서 볼 때)

**상태:** 2026-03~04 세션에서 lifecycle 로그 연동·원인 요약까지 완료.  
**다음:** conf 0.30 재기동을 **한 줄 명령으로 정상 기동** 확인 후, 새 로그만 요약 비교.

---

## 1. 무엇을 했나

- `[track-lifecycle]` 로그: `serve_ai_overlay` / `run_rtsp_inference` 연동
- env: `TRACK_ID_LOG=true` (또는 `TRACKING_DEBUG=true`)
- **로그 위치 (중요):** `ai_runner.log` 가 아님  
  → `runs/registered_cameras/<cameraLoginId>-overlay.log`
- 요약 스크립트(로컬): `scripts/summarize_track_lifecycle.py`  
  (서버에 없으면 인라인 python 요약 사용)

---

## 2. 요약 데이터로 확정된 패턴

| 지표 | 관찰값 (전체 로그, conf 튜닝 전·혼합 가능) |
|------|---------------------------------------------|
| new_track | ~341, 전부 `no_match` |
| lost | ~316, 전부 `max_missing_seconds` (~4.0s) |
| new ≈ lost | **ID churn** (만들고 4초 뒤 지움) |
| new conf | p50≈0.20, **~64% &lt; 0.25** |
| soft/sole match | 거의 없음 (sole 5회) |
| 카메라 | cam_02~05 비슷 → 공통 threshold 이슈 |

**사이클:** 약한 det → `new_track` → 4초 미매칭 → `lost` → 재검출 시 다른 ID.

**아직 올리지 말 것:** grace 4→8초 (고스트만 더 오래 남음).  
**먼저 할 것:** `DETECTOR_CONF` / `TRACK_THRESH` 로 저conf new 줄이기.

---

## 3. 권장 env (다음 실험)

```text
TRACK_ID_LOG=true
DETECTOR_CONF=0.30
TRACK_THRESH=0.30
```

사람 미검출이 심하면 0.25로 한 단계만 내림.  
저conf new가 준 뒤에만 grace(`TRACK_MAX_MISSING_SECONDS` / `TRACKING_GRACE_PERIOD_SECONDS`) 6초 검토.

---

## 4. SSH 재기동 (한 줄, `\` 쓰지 말 것)

경로: `/home/welabs/yolo_training/strange_ai_lstm`  
유저 예: `welabs@...`

```bash
cd /home/welabs/yolo_training/strange_ai_lstm && source .venv/bin/activate && pkill -f "scripts/run_registered_cameras.py" 2>/dev/null || true; pkill -f "scripts/serve_ai_overlay.py" 2>/dev/null || true; sleep 2; nohup env TRACK_ID_LOG=true DETECTOR_CONF=0.30 TRACK_THRESH=0.30 python scripts/run_registered_cameras.py --backend-base-url http://127.0.0.1:18080 --rtsp-base-url rtsp://127.0.0.1:8554 --video-pool /home/welabs/yolo_training/ai_fall_experiments/data/raw/indoor_chromakey/videos --overlay-base-port 8010 --overlay-report-enabled --detector-mode real --yolo-model yolo26n-pose.pt --publisher mqtt --mqtt-host 15.165.248.37 --mqtt-port 1883 --mqtt-topic safety/events --skip-simulated-ffmpeg > ai_runner.log 2>&1 </dev/null & sleep 3; pgrep -af "run_registered_cameras|serve_ai_overlay"; tr '\0' '\n' < /proc/$(pgrep -n -f serve_ai_overlay.py)/environ | grep -E 'DETECTOR_CONF|TRACK_THRESH|TRACK_ID_LOG'; tail -n 40 ai_runner.log
```

**주의 (이미 한 번 발생):** 줄바꿈/`\` 복붙 시 명령 중간에서 `> ai_runner.log &` 가 끼어 **인자가 잘림**. 반드시 **한 줄 통째로**.

성공 확인:

- `serve_ai_overlay` 프로세스 존재
- environ에 `DETECTOR_CONF=0.30` 등
- `ai_runner.log` 에 traceback 없음

---

## 5. 새 로그만 비교할 때

재기동 전 로그 치우기(선택):

```bash
mkdir -p runs/registered_cameras/archive
mv runs/registered_cameras/*-overlay.log runs/registered_cameras/archive/ 2>/dev/null || true
```

2~3분 후 요약:

```bash
cd /home/welabs/yolo_training/strange_ai_lstm && python - <<'PY'
import json,re
from collections import Counter
from pathlib import Path
rx=re.compile(r"\[track-lifecycle\]\s*(\{.*\})\s*$")
ev_c,rs_c=Counter(),Counter(); confs=[]
n=0
for p in sorted(Path("runs/registered_cameras").glob("*-overlay.log")):
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        if "[track-lifecycle]" not in line: continue
        m=rx.search(line)
        if not m: continue
        try: rec=json.loads(m.group(1))
        except: continue
        n+=1
        for e in rec.get("events") or []:
            k=f"{e.get('event')}:{e.get('reason') or '-'}"
            ev_c[e.get("event")]+=1; rs_c[k]+=1
            if e.get("event")=="new_track" and e.get("confidence") is not None:
                confs.append(float(e["confidence"]))
print("records",n,"by event",dict(ev_c))
for k,v in rs_c.most_common(8): print(k,v)
if confs:
    confs.sort()
    print(f"new conf n={len(confs)} min={confs[0]:.3f} p50={confs[len(confs)//2]:.3f} max={confs[-1]:.3f} <0.25={sum(c<0.25 for c in confs)} <0.30={sum(c<0.30 for c in confs)}")
else:
    print("no new_track conf yet")
PY
```

**채팅에는 요약 출력만 붙이면 됨** (원본 lifecycle 전체 불필요).

### 성공 시 기대

- `new_track` / `lost` 크게 감소
- `conf<0.25` ≈ 0, p50 ≥ 0.30
- 화면에서 실인원 검출은 유지

---

## 6. 코드 위치 (strange_ai)

| 역할 | 경로 |
|------|------|
| lifecycle 이벤트 생성 | `tracking/simple_tracker.py` |
| 로그 출력 | `ai/inference/tracking_debug.py` → `log_track_lifecycle_events` |
| overlay 연동 | `scripts/serve_ai_overlay.py` |
| 배치 연동 | `scripts/run_rtsp_inference.py` |
| 카메라별 로그 파일 | `ai/registered_camera_workers.py` → `runs/registered_cameras/*-overlay.log` |
| ByteTrack active set diff | `ai/postprocess/supervision_postprocessor.py` |

브랜치: `codex/ai-worker-flow-improvements` (서버: `~/yolo_training/strange_ai_lstm`)

---

## 7. 다음 세션 체크리스트

1. [ ] 한 줄 재기동 성공 + env 0.30 확인  
2. [ ] 로그 archive 후 2~3분 수집  
3. [ ] 요약 붙여넣기 → new/lost·conf 비교  
4. [ ] 미검출이면 thr 0.25 / churn 남으면 thr·dedup 추가 검토  
5. [ ] 실인원 lost만 남으면 grace 6s 실험  

이 파일만 열어도 이어서 가능.
