import json
import os
import psutil
from datetime import datetime
from pathlib import Path

REGISTRY_FILE = Path(__file__).resolve().parents[1] / "runs" / "camera_worker_registry.json"


def check_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def check_process_signature_matches(pid: int, keywords: list[str]) -> bool:
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
        if cmdline:
            # Check if any keyword matches any argument in the command line
            cmd_str = " ".join(cmdline).lower()
            return any(kw.lower() in cmd_str for kw in keywords)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return False


def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        with REGISTRY_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = REGISTRY_FILE.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    tmp_file.replace(REGISTRY_FILE)


def register_worker(camera_login_id: str, pid: int, rtsp_url: str, output_path: str = "") -> None:
    registry = load_registry()
    existing = registry.get(camera_login_id)
    if existing:
        old_pid = existing.get("pid")
        if old_pid and check_pid_alive(old_pid) and check_process_signature_matches(old_pid, ["serve_ai_overlay.py", "python"]):
            raise RuntimeError(f"Worker for camera {camera_login_id} is already running with PID {old_pid}")
    
    registry[camera_login_id] = {
        "type": "worker",
        "pid": pid,
        "started_at": datetime.now().isoformat(),
        "rtsp_url": rtsp_url,
        "output_path": output_path
    }
    save_registry(registry)
    print(f"[worker-registry] Registered worker camera={camera_login_id} (pid={pid}, rtsp={rtsp_url})", flush=True)


def unregister_worker(camera_login_id: str) -> None:
    registry = load_registry()
    if camera_login_id in registry:
        pid = registry[camera_login_id].get("pid")
        del registry[camera_login_id]
        save_registry(registry)
        print(f"[worker-registry] Unregistered worker camera={camera_login_id} (pid={pid})", flush=True)


def force_kill_existing_worker(camera_login_id: str) -> None:
    registry = load_registry()
    existing = registry.get(camera_login_id)
    if existing:
        pid = existing.get("pid")
        if pid:
            try:
                proc = psutil.Process(pid)
                print(f"[worker-registry] Force killing registered worker camera={camera_login_id} (pid={pid})", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except psutil.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    # Scavenge / search system processes via psutil to kill duplicates
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = proc.info['name']
            if name and 'python' in name.lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('serve_ai_overlay.py' in arg for arg in cmdline) and any(camera_login_id in arg for arg in cmdline):
                    pid = proc.info['pid']
                    if pid != os.getpid():
                        print(f"[worker-registry] Scavenged and killing duplicate python worker camera={camera_login_id} (pid={pid})", flush=True)
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            try:
                                proc.wait(timeout=3)
                            except psutil.TimeoutExpired:
                                pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if camera_login_id in registry:
        del registry[camera_login_id]
        save_registry(registry)


def register_publisher(output_path: str, pid: int, rtsp_url: str, camera_login_id: str) -> None:
    registry = load_registry()
    # Check if publisher with same output path is already active
    for key, val in list(registry.items()):
        if val.get("type") == "publisher" and val.get("output_path") == output_path:
            old_pid = val.get("pid")
            if old_pid and check_pid_alive(old_pid) and check_process_signature_matches(old_pid, ["ffmpeg"]):
                raise RuntimeError(f"Publisher for RTSP destination {output_path} is already running with PID {old_pid}")

    pub_key = f"publisher:{camera_login_id}"
    registry[pub_key] = {
        "type": "publisher",
        "pid": pid,
        "started_at": datetime.now().isoformat(),
        "rtsp_url": rtsp_url,
        "output_path": output_path
    }
    save_registry(registry)
    print(f"[worker-registry] Registered publisher output={output_path} (pid={pid}, camera={camera_login_id})", flush=True)


def unregister_publisher_by_path(output_path: str) -> None:
    registry = load_registry()
    for key, val in list(registry.items()):
        if val.get("type") == "publisher" and val.get("output_path") == output_path:
            del registry[key]
            save_registry(registry)
            print(f"[worker-registry] Unregistered publisher output={output_path} (pid={val.get('pid')})", flush=True)
            break


def force_kill_existing_publisher(output_path: str) -> None:
    registry = load_registry()
    for key, val in list(registry.items()):
        if val.get("type") == "publisher" and val.get("output_path") == output_path:
            pid = val.get("pid")
            if pid:
                try:
                    proc = psutil.Process(pid)
                    print(f"[worker-registry] Force killing existing publisher output={output_path} (pid={pid})", flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            del registry[key]
            save_registry(registry)

    # Scavenge system-wide ffmpeg processes targeting the same output_path
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = proc.info['name']
            if name and 'ffmpeg' in name.lower():
                cmdline = proc.info['cmdline']
                if cmdline and any(output_path in arg for arg in cmdline):
                    pid = proc.info['pid']
                    print(f"[worker-registry] Scavenged and killing duplicate ffmpeg publisher output={output_path} (pid={pid})", flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


def get_active_worker_count() -> int:
    registry = load_registry()
    count = 0
    for val in registry.values():
        if val.get("type") == "worker":
            pid = val.get("pid")
            if pid and check_pid_alive(pid):
                count += 1
    return count


def get_active_publisher_count() -> int:
    registry = load_registry()
    count = 0
    for val in registry.values():
        if val.get("type") == "publisher":
            pid = val.get("pid")
            if pid and check_pid_alive(pid):
                count += 1
    return count
