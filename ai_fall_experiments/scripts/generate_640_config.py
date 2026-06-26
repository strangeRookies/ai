import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser(description="Generate 640px config from base config.")
    parser.add_argument("--config", default="configs/fall_lstm.yaml", help="Path to base config file")
    parser.add_argument("--output", default="configs/fall_lstm_640.yaml", help="Path to output 640 config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)

    if not config_path.exists():
        print(f"[ERROR] Original config not found: {config_path}")
        return

    print(f"[*] Reading base config: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    # 1. 메타데이터 및 클립 경로 수정 (_640 추가)
    if "paths" in config:
        paths = config["paths"]
        if "metadata_csv" in paths:
            # e.g., data/metadata/metadata.csv -> data/metadata/metadata_640.csv
            p = Path(paths["metadata_csv"])
            paths["metadata_csv"] = str(p.parent / f"{p.stem}_640{p.suffix}")
            print(f"[*] Updated metadata_csv: {paths['metadata_csv']}")
        if "clips_dir" in paths:
            # e.g., data/processed/clips -> data/processed/clips_640
            p = Path(paths["clips_dir"])
            paths["clips_dir"] = str(p.parent / f"{p.name}_640")
            print(f"[*] Updated clips_dir: {paths['clips_dir']}")

    # 2. 도메인별 입력 비디오 및 어노테이션 폴더를 640 경로로 수정
    if "domains" in config:
        for domain, domain_cfg in config["domains"].items():
            if "video_dir" in domain_cfg:
                # e.g., data/raw/indoor_background/videos -> data/raw_640/indoor_background/videos
                video_path = domain_cfg["video_dir"]
                if "data/raw" in video_path:
                    domain_cfg["video_dir"] = video_path.replace("data/raw", "data/raw_640")
                    print(f"[*] Updated {domain} video_dir: {domain_cfg['video_dir']}")
            if "annotation_dir" in domain_cfg:
                # e.g., data/raw/indoor_background/annotations -> data/raw_640/indoor_background/annotations
                annot_path = domain_cfg["annotation_dir"]
                if "data/raw" in annot_path:
                    domain_cfg["annotation_dir"] = annot_path.replace("data/raw", "data/raw_640")
                    print(f"[*] Updated {domain} annotation_dir: {domain_cfg['annotation_dir']}")

    # 3. 새로운 640 설정 파일 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, default_flow_style=False, allow_unicode=True)

    print(f"[SUCCESS] Generated 640px config at: {output_path}")

if __name__ == "__main__":
    main()
