"""
Training Agent — 部署在算法服务器（GPU机）上的轻量 HTTP 服务。

职责：
  1. 接收来自存储服务器 Spring Boot 的"启动训练"请求
  2. 将请求参数转成命令行并以 subprocess 启动训练脚本
  3. 提供"停止训练"和"查询状态"接口
  4. 训练脚本自身负责回调存储服务器的进度接口（与现在完全一致）

部署：
  pip install fastapi uvicorn
  python training_agent.py            # 默认监听 0.0.0.0:5000
  python training_agent.py --port 5000
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from threading import Thread
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Training Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# taskId -> subprocess.Popen
_running: Dict[str, subprocess.Popen] = {}


# ──────────────────────────────────────────────
# 请求体模型（与 Spring Boot buildPythonCommand 中的参数对应）
# ──────────────────────────────────────────────

class TrainRequest(BaseModel):
    taskId: str
    scriptPath: str               # 训练脚本绝对路径，由 Spring Boot 指定
    pythonPath: str = "python3"   # Python 解释器路径
    apiBaseUrl: str               # Spring Boot 的地址，供脚本回调进度
    args: Dict[str, str]          # 其余所有命令行参数 key→value
    # 无 value 的开关参数（如 --fp16）放在 flags 列表里
    flags: list[str] = []


# ──────────────────────────────────────────────
# 接口
# ──────────────────────────────────────────────

@app.post("/train")
def start_training(req: TrainRequest):
    """启动一个训练任务（异步，立即返回）"""
    task_id = req.taskId

    if task_id in _running and _running[task_id].poll() is None:
        raise HTTPException(status_code=409, detail=f"Task {task_id} is already running")

    # 构建命令行
    cmd = [req.pythonPath, req.scriptPath]
    cmd += ["--task_id", task_id]
    cmd += ["--api_base_url", req.apiBaseUrl]

    for key, value in req.args.items():
        cmd += [f"--{key}", str(value)]

    for flag in req.flags:
        cmd.append(f"--{flag}")

    logger.info("Starting training: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _running[task_id] = proc

    # 后台线程消费输出，避免管道阻塞
    def _drain(p: subprocess.Popen, tid: str):
        for line in p.stdout:
            logger.info("[%s] %s", tid, line.rstrip())
        exit_code = p.wait()
        logger.info("Training finished: taskId=%s, exit=%d", tid, exit_code)
        _running.pop(tid, None)

    Thread(target=_drain, args=(proc, task_id), daemon=True).start()

    return {"taskId": task_id, "pid": proc.pid, "status": "STARTED"}


@app.post("/stop/{task_id}")
def stop_training(task_id: str):
    """停止训练任务"""
    proc = _running.get(task_id)
    if proc is None or proc.poll() is not None:
        raise HTTPException(status_code=404, detail=f"No running task: {task_id}")

    # 先发 SIGTERM，再等 5 秒，最后 SIGKILL
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    _running.pop(task_id, None)
    logger.info("Stopped training: taskId=%s", task_id)
    return {"taskId": task_id, "status": "STOPPED"}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """查询任务是否仍在运行"""
    proc = _running.get(task_id)
    running = proc is not None and proc.poll() is None
    return {
        "taskId": task_id,
        "running": running,
        "pid": proc.pid if running else None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "runningTasks": len(_running)}


# ──────────────────────────────────────────────
# 启动入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    logger.info("Training Agent starting on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)
