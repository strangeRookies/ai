import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV가 설치되어 있지 않습니다. 'pip install opencv-python'을 실행하세요.", file=sys.stderr)
    sys.exit(1)

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    print("[WARNING] Ultralytics가 설치되어 있지 않습니다. 모델 추론 진단은 건너뜁니다. 'pip install ultralytics'를 실행하세요.")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Pose Extractor 정밀 진단 툴")
    parser.add_argument("--metadata-csv", default="../ai_fall_experiments/data/metadata/metadata.csv", help="검증할 메타데이터 CSV 파일 경로")
    parser.add_argument("--models", default="yolo11n-pose.pt,yolo26n-pose.pt,yolov8s-pose.pt", help="진단할 YOLO Pose 모델명 목록 (쉼표 구분)")
    parser.add_argument("--device", default="auto", help="추론 장비 (auto, cpu, 0 등)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 입력 이미지 크기")
    parser.add_argument("--samples", type=int, default=3, help="진단할 샘플 로우 수")
    return parser.parse_args()


def resolve_path_locally(path_str, base_dirs):
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    if p.is_absolute():
        # 절대 경로가 틀린 경우, 파일 이름만 추출하여 상대 경로 검색 시도
        p = Path(p.name)
    for base in base_dirs:
        candidate = base / p
        if candidate.exists():
            return candidate.resolve()
        # 데이터 상대 경로 매칭 시도 (data/raw/... 등)
        for sub in ["", "ai_fall_experiments", "strange_ai"]:
            candidate = base / sub / p
            if candidate.exists():
                return candidate.resolve()
    return None


def test_video_reading(video_path):
    result = {
        "exists": False,
        "is_opened": False,
        "frames_decoded": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "error": None
    }
    
    if not video_path:
        result["error"] = "경로가 지정되지 않았습니다."
        return result
        
    p = Path(video_path)
    result["exists"] = p.exists()
    if not result["exists"]:
        result["error"] = "파일이 디스크에 존재하지 않습니다."
        return result
        
    try:
        cap = cv2.VideoCapture(str(p))
        result["is_opened"] = cap.isOpened()
        if not result["is_opened"]:
            result["error"] = "OpenCV VideoCapture가 비디오 파일을 열지 못했습니다. (코덱 또는 포맷 호환성 문제)"
            cap.release()
            return result
            
        result["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 최대 10프레임 읽기 테스트
        for _ in range(10):
            ok, frame = cap.read()
            if not ok:
                break
            result["frames_decoded"] += 1
            
        cap.release()
        if result["frames_decoded"] == 0:
            result["error"] = "비디오가 열렸으나 프레임 디코딩에 실패했습니다. (코덱 미지원)"
    except Exception as e:
        result["error"] = f"비디오 읽기 중 예외 발생: {str(e)}"
        
    return result


def diagnose_yolo_inference(model_name, frame, device, imgsz):
    if YOLO is None:
        return {"error": "Ultralytics 미설치"}
        
    try:
        # GPU 장비 자동 할당
        if device == "auto":
            device_val = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device_val = device
            
        model = YOLO(model_name)
        
        # 신뢰도 임계값별 스윕 테스트
        thresholds = [0.25, 0.15, 0.10, 0.05]
        sweep_results = {}
        
        for conf in thresholds:
            results = model.predict(frame, device=device_val, imgsz=imgsz, conf=conf, verbose=False)
            boxes_found = 0
            max_conf = 0.0
            kp_conf_avg = 0.0
            has_kp = False
            
            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    boxes_found += len(boxes)
                    confs = boxes.conf.detach().cpu().tolist()
                    max_conf = max(max_conf, max(confs))
                    
                keypoints = getattr(res, "keypoints", None)
                if keypoints is not None and keypoints.conf is not None:
                    has_kp = True
                    kps_conf = keypoints.conf.detach().cpu()
                    if kps_conf.numel() > 0:
                        kp_conf_avg = float(kps_conf.mean().item())
            
            sweep_results[str(conf)] = {
                "detections": boxes_found,
                "max_box_confidence": round(max_conf, 4),
                "has_keypoints": has_kp,
                "avg_keypoint_confidence": round(kp_conf_avg, 4) if has_kp else 0.0
            }
            
        return {
            "model_loaded": True,
            "device_used": device_val,
            "sweep": sweep_results
        }
    except Exception as e:
        return {"error": f"모델 추론 실패 ({model_name}): {str(e)}"}


def main():
    args = parse_args()
    print("=====================================================================")
    print("                 YOLO Pose Extractor 정밀 진단 시스템                ")
    print("=====================================================================")
    
    metadata_csv = Path(args.metadata_csv)
    print(f"[*] 메타데이터 경로: {metadata_csv.resolve()}")
    
    if not metadata_csv.exists():
        print(f"[ERROR] 메타데이터 CSV 파일이 존재하지 않습니다: {metadata_csv}")
        print(">> 해결 방법: 'python ../ai_fall_experiments/scripts/prepare_clips.py'를 먼저 실행하여 데이터셋 클립을 추출하거나 올바른 --metadata-csv 경로를 제공하세요.")
        return
        
    base_dirs = [
        Path.cwd(),
        metadata_csv.parent,
        metadata_csv.parent.parent,
        metadata_csv.parent.parent.parent,
        Path(r"c:\Users\user\Documents\최종 쉴더스"),
        Path(r"c:\Users\user\Documents\최종 쉴더스 프로젝트")
    ]
    
    # CSV 파일 파싱
    rows = []
    try:
        with metadata_csv.open("r", encoding="utf-8-sig", newline="") as fp:
            for row in csv.DictReader(fp):
                rows.append(dict(row))
    except Exception as e:
        print(f"[ERROR] 메타데이터 CSV 읽기 실패: {e}")
        return
        
    print(f"[*] 총 데이터 행(Clips) 수: {len(rows)}")
    if len(rows) == 0:
        print("[WARNING] CSV 파일이 비어 있습니다.")
        return
        
    # 샘플 로우 검증
    samples = rows[:args.samples]
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    
    for i, row in enumerate(samples, start=1):
        print("\n---------------------------------------------------------------------")
        print(f"샘플 {i}: 클립 ID = {row.get('clip_id', 'N/A')} (도메인: {row.get('domain', 'N/A')}, 라벨: {row.get('label_name', 'N/A')})")
        print("---------------------------------------------------------------------")
        
        # 1. 경로 진단
        raw_video_path = row.get("source_video") or row.get("video_path") or ""
        clip_path = row.get("clip_path") or ""
        
        resolved_raw = resolve_path_locally(raw_video_path, base_dirs)
        resolved_clip = resolve_path_locally(clip_path, base_dirs)
        
        print(f" [1] 원본 비디오 경로 해석:")
        print(f"   - 표기 경로: {raw_video_path}")
        print(f"   - 해상 경로: {resolved_raw if resolved_raw else '❌ 찾을 수 없음'}")
        
        print(f" [2] 전처리 클립 경로 해석:")
        print(f"   - 표기 경로: {clip_path}")
        print(f"   - 해상 경로: {resolved_clip if resolved_clip else '❌ 찾을 수 없음'}")
        
        # 2. 비디오 읽기 및 코덱 진단
        print(f" [3] 비디오 프레임 디코딩 테스트 (OpenCV):")
        
        # 전처리 클립 테스트
        clip_diag = test_video_reading(resolved_clip)
        print(f"   * 전처리 클립 디코딩 결과:")
        if clip_diag["exists"]:
            print(f"     - 파일 존재 여부: YES")
            print(f"     - OpenCV Open 성공: {clip_diag['is_opened']}")
            print(f"     - 성공적 디코딩 프레임 수 (최대 10): {clip_diag['frames_decoded']}/10")
            print(f"     - 비디오 해상도: {clip_diag['width']}x{clip_diag['height']}")
            print(f"     - FPS: {clip_diag['fps']}")
            if clip_diag["error"]:
                print(f"     - ❌ 에러: {clip_diag['error']}")
        else:
            print(f"     - ❌ 파일이 디스크에 없음 (동기화 확인 필요)")
            
        # 원본 비디오 테스트
        raw_diag = test_video_reading(resolved_raw)
        print(f"   * 원본 비디오 디코딩 결과:")
        if raw_diag["exists"]:
            print(f"     - 파일 존재 여부: YES")
            print(f"     - OpenCV Open 성공: {raw_diag['is_opened']}")
            print(f"     - 성공적 디코딩 프레임 수 (최대 10): {raw_diag['frames_decoded']}/10")
            print(f"     - 비디오 해상도: {raw_diag['width']}x{raw_diag['height']}")
            print(f"     - FPS: {raw_diag['fps']}")
            if raw_diag["error"]:
                print(f"     - ❌ 에러: {raw_diag['error']}")
        else:
            print(f"     - ❌ 파일이 디스크에 없음")
            
        # 3. YOLO 추론 및 임계값 진단
        if YOLO is not None and resolved_clip and clip_diag["frames_decoded"] > 0:
            print(f" [4] YOLO Pose 모델 추론 및 임계값(Threshold) 스윕 검사:")
            # 첫 프레임 로드
            cap = cv2.VideoCapture(str(resolved_clip))
            _, frame = cap.read()
            cap.release()
            
            for model_name in model_list:
                print(f"   * 모델: {model_name}")
                diag_inf = diagnose_yolo_inference(model_name, frame, args.device, args.imgsz)
                if "error" in diag_inf:
                    print(f"     - ❌ 에러: {diag_inf['error']}")
                else:
                    print(f"     - 추론 장치: {diag_inf['device_used']}")
                    for conf_val, stats in diag_inf["sweep"].items():
                        status_str = "✅ 감지 성공" if stats["detections"] > 0 else "❌ 감지 실패"
                        kp_status = "유효" if stats["has_keypoints"] else "없음"
                        print(f"     - conf={conf_val} 임계값: {status_str} | 감지 객체 수: {stats['detections']} (최대 bbox 신뢰도: {stats['max_box_confidence']}) | 키포인트: {kp_status} (평균 신뢰도: {stats['avg_keypoint_confidence']})")
        elif YOLO is not None:
            print(f" [4] ❌ 전처리 클립이 정상적으로 디코딩되지 않아 YOLO 추론 테스트를 스킵합니다.")
            
    print("\n=====================================================================")
    print("                           진단 종합 리포트                          ")
    print("=====================================================================")
    print("1. [경로 해결책]:")
    print("   - 만약 모든 비디오가 '찾을 수 없음'으로 뜬다면, GPU PC 환경에서 데이터셋(clips 및 raw 비디오)이")
    print("     'ai_fall_experiments/data/processed/clips' 및 'ai_fall_experiments/data/raw' 디렉토리에 동기화되었는지 확인해 주세요.")
    print("2. [프레임 디코딩(OpenCV) 해결책]:")
    print("   - 만약 파일은 존재하지만 '디코딩 프레임 수 = 0' 또는 'OpenCV Open 실패'가 발생할 경우,")
    print("     Windows 환경용 코덱 팩이 누락되었거나 'mp4v' 코덱이 설치되지 않은 상태입니다.")
    print("     (Windows 미디어 기능 팩 설치 또는 'conda install -c conda-forge ffmpeg' 등으로 해결 가능)")
    print("3. [신뢰도 임계값 해결책]:")
    print("   - 만약 conf=0.25에서 감지수가 0이지만 conf=0.15 또는 conf=0.10에서 감지 성공률이 올라간다면,")
    print("     'compare_lstm_extractors.py' 내 YOLO Pose Detector 호출 시 신뢰도 임계값을 0.15 이하로 낮추어 주입해야 합니다.")
    print("4. [클래스 필터링 해결책]:")
    print("   - 추론 성공 시 '키포인트: 없음'이 발생하는 경우, Pose 전용 모델(예: -pose.pt)이 아닌")
    print("     일반 객체 감지용 모델(예: yolov8n.pt)이 사용되고 있지는 않은지 확인이 필요합니다.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
