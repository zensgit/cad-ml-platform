from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import httpx


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary not found: {name}")
    return path


def _wait_http(url: str, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code < 500:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.1)
    raise RuntimeError(f"Timeout waiting for {url} (last_err={last_err})")


def _wait_redis_ping(redis_cli: str, port: int, *, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                [redis_cli, "-p", str(port), "PING"],
                stderr=subprocess.STDOUT,
                timeout=1.0,
            ).decode("utf-8", errors="replace")
            if out.strip() == "PONG":
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("Timeout waiting for redis-server PING")


@dataclass(frozen=True)
class _ProcSpec:
    name: str
    args: list[str]
    cwd: Path
    env: Dict[str, str]
    log_path: Path


@contextlib.contextmanager
def _managed_proc(spec: _ProcSpec) -> Iterator[subprocess.Popen[bytes]]:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(spec.log_path, "wb")
    proc = subprocess.Popen(
        spec.args,
        cwd=str(spec.cwd),
        env=spec.env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=False,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        log_f.close()


@contextlib.contextmanager
def _managed_minio_container(
    *,
    s3_port: int,
    console_port: int,
    env: Dict[str, str],
    log_path: Path,
) -> Iterator[Dict[str, Any]]:
    image = env.get("DEDUP2D_E2E_MINIO_IMAGE", "minio/minio:latest").strip() or "minio/minio:latest"
    root_user = env.get("MINIO_ROOT_USER", "minioadmin")
    root_password = env.get("MINIO_ROOT_PASSWORD", "minioadmin")
    name = f"dedup2d-minio-e2e-{int(time.time())}"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "wb")
    try:
        cid = (
            subprocess.check_output(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    name,
                    "-p",
                    f"{s3_port}:9000",
                    "-p",
                    f"{console_port}:9001",
                    "-e",
                    f"MINIO_ROOT_USER={root_user}",
                    "-e",
                    f"MINIO_ROOT_PASSWORD={root_password}",
                    image,
                    "server",
                    "/data",
                    "--console-address",
                    ":9001",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8", errors="replace")
            .strip()
        )
    except subprocess.CalledProcessError as e:
        log_f.write(e.output or b"")
        raise

    try:
        yield {
            "container_id": cid,
            "container_name": name,
            "root_user": root_user,
            "root_password": root_password,
            "s3_port": s3_port,
            "console_port": console_port,
        }
    finally:
        try:
            logs = subprocess.check_output(["docker", "logs", name], stderr=subprocess.STDOUT)
            log_f.write(logs)
        except Exception:
            pass
        try:
            subprocess.check_call(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        log_f.close()


def _write_fake_apps(tmp_dir: Path, *, callback_out: Path) -> None:
    (tmp_dir / "vision_app.py").write_text(
        """\
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok", "service": "fake-vision"}

@app.post("/api/v2/search")
async def search(
    file: UploadFile = File(...),
    mode: str = Form("balanced"),
    max_results: int = Form(50),
    compute_diff: bool = Form(True),
    enable_ml: bool = Form(False),
    enable_geometric: bool = Form(False),
):
    _ = await file.read()
    item = {
        "drawing_id": "D-001",
        "file_hash": "hash_dup_001",
        "file_name": "duplicate.png",
        "similarity": 0.99,
        "confidence": 0.9,
        "match_level": 2,
        "verdict": "duplicate",
        "levels": {"l1": {"phash": 0.99}, "l2": {"embedding": 0.98}},
    }
    return {
        "success": True,
        "total_matches": 1,
        "duplicates": [item],
        "similar": [],
        "final_level": 2,
        "timing": {"total_ms": 1.0, "l1_ms": 0.2, "l2_ms": 0.8, "l3_ms": 0.0, "l4_ms": 0.0},
        "level_stats": {},
        "warnings": [],
        "error": None,
    }
""",
        encoding="utf-8",
    )

    (tmp_dir / "callback_app.py").write_text(
        f"""\
import json
from pathlib import Path

from fastapi import FastAPI, Request

app = FastAPI()
OUT = Path({json.dumps(str(callback_out))})

@app.post("/hook")
async def hook(request: Request):
    body = await request.body()
    headers = dict(request.headers)
    OUT.write_text(json.dumps({{"headers": headers, "body": body.decode("utf-8", errors="replace")}}))
    return {{"ok": True}}
""",
        encoding="utf-8",
    )

    (tmp_dir / "api_app.py").write_text(
        """\
from fastapi import FastAPI

from src.api.v1 import dedup

app = FastAPI()
app.include_router(dedup.router, prefix="/api/v1/dedup")
""",
        encoding="utf-8",
    )


def _verify_signature(*, secret: str, job_id: str, body_text: str, signature_header: str) -> None:
    parts = {}
    for kv in signature_header.split(","):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        parts[k.strip()] = v.strip()
    ts = int(parts["t"])
    sig = parts["v1"]
    msg = f"{ts}.{job_id}.".encode("utf-8") + body_text.encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise AssertionError("signature mismatch")


def _list_minio_objects(*, endpoint: str, bucket: str, access_key: str, secret_key: str, prefix: str) -> list[str]:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = resp.get("Contents") or []
    return [str(obj.get("Key") or "") for obj in contents if obj.get("Key")]


def _create_minio_bucket(*, endpoint: str, bucket: str, access_key: str, secret_key: str) -> None:
    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    try:
        client.create_bucket(Bucket=bucket)
    except Exception:
        # Best-effort: bucket might already exist.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dedup2D webhook E2E smoke test (Redis + ARQ + MinIO S3 backend)"
    )
    parser.add_argument("--keep-dir", action="store_true", help="Do not delete temp dir on success")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--job-timeout", type=float, default=60.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    redis_server = _require_bin("redis-server")
    redis_cli = _require_bin("redis-cli")
    arq_bin = str((Path(sys.executable).parent / "arq"))
    if not Path(arq_bin).exists():
        arq_bin = _require_bin("arq")

    redis_port = _free_port()
    vision_port = _free_port()
    callback_port = _free_port()
    api_port = _free_port()
    minio_s3_port = _free_port()
    minio_console_port = _free_port()

    key_prefix = f"dedup2d_e2e_{int(time.time())}"
    queue_name = f"{key_prefix}:queue"

    tmp_dir = Path(tempfile.mkdtemp(prefix="dedup2d_webhook_e2e_minio_"))
    logs_dir = tmp_dir / "logs"
    callback_out = tmp_dir / "callback_received.json"
    _write_fake_apps(tmp_dir, callback_out=callback_out)

    bucket = f"dedup2d-e2e-{int(time.time())}"
    s3_prefix = f"uploads/{key_prefix}"
    endpoint = f"http://127.0.0.1:{minio_s3_port}"
    access_key = "minioadmin"
    secret_key = "minioadmin"

    common_env = dict(os.environ)
    common_env.update(
        {
            "DEDUP2D_ASYNC_BACKEND": "redis",
            "DEDUP2D_REDIS_URL": f"redis://127.0.0.1:{redis_port}/0",
            "DEDUP2D_REDIS_KEY_PREFIX": key_prefix,
            "DEDUP2D_ARQ_QUEUE_NAME": queue_name,
            "DEDUP2D_ASYNC_TTL_SECONDS": "3600",
            "DEDUP2D_ASYNC_MAX_JOBS": "50",
            "DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS": "60",
            # Use S3 backend
            "DEDUP2D_FILE_STORAGE": "s3",
            "DEDUP2D_S3_BUCKET": bucket,
            "DEDUP2D_S3_PREFIX": s3_prefix,
            "DEDUP2D_S3_ENDPOINT": endpoint,
            "DEDUP2D_S3_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": access_key,
            "AWS_SECRET_ACCESS_KEY": secret_key,
            "DEDUP2D_FILE_STORAGE_CLEANUP_ON_FINISH": "1",
            # Allow local HTTP callback in dev smoke test
            "DEDUP2D_CALLBACK_ALLOW_HTTP": "1",
            "DEDUP2D_CALLBACK_BLOCK_PRIVATE_NETWORKS": "0",
            "DEDUP2D_CALLBACK_MAX_ATTEMPTS": "1",
            "DEDUP2D_CALLBACK_TIMEOUT_SECONDS": "2",
            "DEDUP2D_CALLBACK_BACKOFF_BASE_SECONDS": "0",
            "DEDUP2D_CALLBACK_BACKOFF_MAX_SECONDS": "0",
            "DEDUP2D_CALLBACK_HMAC_SECRET": "e2e_secret",
            # Wire ML platform -> fake vision
            "DEDUPCAD_VISION_URL": f"http://127.0.0.1:{vision_port}",
            "DEDUPCAD_VISION_TIMEOUT_SECONDS": "10",
        }
    )
    env_with_tmp = dict(common_env)
    env_with_tmp["PYTHONPATH"] = str(tmp_dir) + os.pathsep + str(repo_root)

    specs = [
        _ProcSpec(
            name="redis",
            args=[redis_server, "--bind", "127.0.0.1", "--port", str(redis_port), "--save", "", "--appendonly", "no"],
            cwd=tmp_dir,
            env=common_env,
            log_path=logs_dir / "redis.log",
        ),
        _ProcSpec(
            name="vision",
            args=[
                python,
                "-m",
                "uvicorn",
                "vision_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(vision_port),
                "--log-level",
                "warning",
            ],
            cwd=tmp_dir,
            env=env_with_tmp,
            log_path=logs_dir / "vision.log",
        ),
        _ProcSpec(
            name="callback",
            args=[
                python,
                "-m",
                "uvicorn",
                "callback_app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(callback_port),
                "--log-level",
                "warning",
            ],
            cwd=tmp_dir,
            env=env_with_tmp,
            log_path=logs_dir / "callback.log",
        ),
        _ProcSpec(
            name="api",
            args=[python, "-m", "uvicorn", "api_app:app", "--host", "127.0.0.1", "--port", str(api_port), "--log-level", "warning"],
            cwd=repo_root,
            env=env_with_tmp,
            log_path=logs_dir / "api.log",
        ),
        _ProcSpec(
            name="worker",
            args=[arq_bin, "src.core.dedupcad_2d_worker.WorkerSettings"],
            cwd=repo_root,
            env=common_env,
            log_path=logs_dir / "worker.log",
        ),
    ]

    print("tmp_dir:", tmp_dir)
    print("redis:", f"redis://127.0.0.1:{redis_port}/0")
    print("api:", f"http://127.0.0.1:{api_port}")
    print("vision:", f"http://127.0.0.1:{vision_port}")
    print("callback:", f"http://127.0.0.1:{callback_port}/hook")
    print("minio_s3:", endpoint)
    print("bucket:", bucket)
    print("prefix:", s3_prefix)
    print("key_prefix:", key_prefix)
    print("queue_name:", queue_name)

    success = False
    try:
        with _managed_minio_container(
            s3_port=minio_s3_port,
            console_port=minio_console_port,
            env=common_env,
            log_path=logs_dir / "minio.log",
        ):
            _wait_http(f"{endpoint}/minio/health/ready", timeout_seconds=args.startup_timeout)
            _create_minio_bucket(endpoint=endpoint, bucket=bucket, access_key=access_key, secret_key=secret_key)

            with _managed_proc(specs[0]):
                _wait_redis_ping(redis_cli, redis_port, timeout_seconds=args.startup_timeout)
                with _managed_proc(specs[1]):
                    _wait_http(f"http://127.0.0.1:{vision_port}/health", timeout_seconds=args.startup_timeout)
                    with _managed_proc(specs[2]):
                        _wait_http(f"http://127.0.0.1:{callback_port}/docs", timeout_seconds=args.startup_timeout)
                        with _managed_proc(specs[3]):
                            _wait_http(f"http://127.0.0.1:{api_port}/docs", timeout_seconds=args.startup_timeout)
                            with _managed_proc(specs[4]):
                                submit_url = f"http://127.0.0.1:{api_port}/api/v1/dedup/2d/search"
                                headers = {"X-API-Key": "tenant_abc123"}
                                params = {
                                    "async": "true",
                                    "mode": "balanced",
                                    "max_results": "10",
                                    "callback_url": f"http://127.0.0.1:{callback_port}/hook",
                                }
                                files = {"file": ("drawing.png", b"fake_png_bytes", "image/png")}
                                with httpx.Client(timeout=10.0) as client:
                                    r = client.post(submit_url, params=params, files=files, headers=headers)
                                    r.raise_for_status()
                                    submit = r.json()

                                job_id = str(submit["job_id"])
                                poll_url = f"http://127.0.0.1:{api_port}" + str(submit["poll_url"])
                                deadline = time.time() + args.job_timeout
                                job: Dict[str, Any] = {}
                                with httpx.Client(timeout=5.0) as client:
                                    while time.time() < deadline:
                                        resp = client.get(poll_url, headers=headers)
                                        resp.raise_for_status()
                                        job = resp.json()
                                        if job.get("status") in {"completed", "failed", "canceled"}:
                                            break
                                        time.sleep(0.25)

                                if job.get("status") != "completed":
                                    raise AssertionError(f"unexpected job status: {job.get('status')}")

                                deadline = time.time() + 10.0
                                while time.time() < deadline and not callback_out.exists():
                                    time.sleep(0.1)
                                if not callback_out.exists():
                                    raise AssertionError("callback not received")

                                cb_data = json.loads(callback_out.read_text(encoding="utf-8"))
                                body_text = str(cb_data.get("body") or "")
                                headers_in = cb_data.get("headers") or {}
                                sig = headers_in.get("x-dedup-signature") or headers_in.get("X-Dedup-Signature")
                                if not sig:
                                    raise AssertionError("missing X-Dedup-Signature")
                                _verify_signature(
                                    secret="e2e_secret",
                                    job_id=job_id,
                                    body_text=body_text,
                                    signature_header=str(sig),
                                )

                                payload = json.loads(body_text)
                                if payload.get("job_id") != job_id:
                                    raise AssertionError("callback job_id mismatch")

                                # Verify MinIO objects cleaned up (cleanup_on_finish enabled)
                                deadline = time.time() + 10.0
                                while time.time() < deadline:
                                    keys = _list_minio_objects(
                                        endpoint=endpoint,
                                        bucket=bucket,
                                        access_key=access_key,
                                        secret_key=secret_key,
                                        prefix=s3_prefix,
                                    )
                                    if not keys:
                                        break
                                    time.sleep(0.25)
                                if keys:
                                    raise AssertionError(f"unexpected leftover S3 objects: {keys[:5]}")

                                print("OK: job completed + callback received + signature verified + minio cleaned")
                                success = True
    finally:
        if not args.keep_dir and success:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

