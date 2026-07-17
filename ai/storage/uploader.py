import os
import sys
from pathlib import Path


def resolve_s3_bucket_name():
    """Return the canonical bucket setting, accepting the deployed legacy alias."""
    return os.environ.get("AWS_S3_BUCKET_NAME") or os.environ.get("S3_BUCKET_NAME")


def upload_clip(path, metadata=None):
    path_obj = Path(path)
    filename = path_obj.name

    # 1. 환경 변수 로드
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket_name = resolve_s3_bucket_name()
    region_name = os.environ.get("AWS_REGION", "ap-northeast-2")  # 기본 서울 리전

    # 2. 필수 환경 변수 부재 시 로컬 폴백 작동
    if not (aws_access_key and aws_secret_key and bucket_name):
        print(
            f"[clip-uploader][fallback] AWS S3 credentials missing. Retaining file locally at: {path}",
            file=sys.stderr,
        )
        return {"uploaded": False, "path": str(path), "metadata": metadata or {}}

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        # 3. S3 클라이언트 생성 및 파일 업로드
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )

        s3_key = f"clips/{filename}"

        # S3 업로드 실행 (Content-Type을 video/mp4로 지정하여 브라우저에서 바로 스트리밍되도록 설정)
        s3.upload_file(
            str(path),
            bucket_name,
            s3_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        # 4. Public S3 URL 생성
        s3_url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{s3_key}"

        print(f"[clip-uploader] S3 upload successful: url={s3_url}", file=sys.stderr)
        return {
            "uploaded": True,
            "url": s3_url,
            "path": str(path),
            "s3_key": s3_key,
            "metadata": metadata or {},
        }

    except ImportError:
        print(
            "[clip-uploader][fallback] boto3 is not installed. Retaining file locally.",
            file=sys.stderr,
        )
        return {"uploaded": False, "path": str(path), "metadata": metadata or {}}
    except (NoCredentialsError, ClientError) as exc:
        print(
            f"[clip-uploader][fallback] S3 upload failed; retaining file locally. Error: {exc}",
            file=sys.stderr,
        )
        return {"uploaded": False, "path": str(path), "metadata": metadata or {}}
