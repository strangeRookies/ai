import sys


def upload_clip(path, metadata=None):
    print(f"[clip-uploader] local clip saved; upload not configured: path={path}", file=sys.stderr)
    return {"uploaded": False, "path": str(path), "metadata": metadata or {}}
