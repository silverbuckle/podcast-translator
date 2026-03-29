# server.py - FastAPI Web サーバー
#
# パイプラインをブラウザから起動・監視するための軽量サーバー。
# ジョブは ThreadPoolExecutor(max_workers=1) で逐次実行。

import io
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import run

app = FastAPI()

OUTPUT_DIR = Path(__file__).parent.parent / "output"
STATIC_DIR = Path(__file__).parent / "static"

# 出力ファイルを /files/ で配信
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# --- ジョブ管理 ---

executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}


class JobRequest(BaseModel):
    url: str
    start: str | None = None
    end: str | None = None
    mode: str = "full"


class _TeeWriter:
    """stdout を元の出力先とバッファの両方に書き込む。"""

    def __init__(self, original: io.TextIOBase, buffer: io.StringIO):
        self.original = original
        self.buffer = buffer

    def write(self, s: str) -> int:
        self.original.write(s)
        self.buffer.write(s)
        return len(s)

    def flush(self):
        self.original.flush()


def _run_job(job_id: str):
    job = jobs[job_id]
    job["status"] = "running"
    job["started_at"] = time.time()
    log_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = _TeeWriter(old_stdout, log_buffer)
    try:
        result = run(
            job["url"],
            start=job.get("start"),
            end=job.get("end"),
            mode=job.get("mode", "full"),
        )
        job["status"] = "completed"
        job["result"] = str(result)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        sys.stdout = old_stdout
        job["logs"] = log_buffer.getvalue()
        job["finished_at"] = time.time()


# --- API エンドポイント ---

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(req: JobRequest):
    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "id": job_id,
        "url": req.url,
        "start": req.start,
        "end": req.end,
        "mode": req.mode,
        "status": "queued",
        "created_at": time.time(),
    }
    executor.submit(_run_job, job_id)
    return {"job_id": job_id}


@app.get("/api/jobs")
async def list_jobs():
    return sorted(jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "not found"}, 404

    resp = {**job}
    if job["status"] == "running" and "started_at" in job:
        resp["elapsed"] = round(time.time() - job["started_at"])
    # ログの末尾から現在のステップを抽出
    logs = job.get("logs", "")
    if logs:
        resp["logs_tail"] = "\n".join(logs.strip().split("\n")[-10:])
    return resp


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
