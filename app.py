from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "data" / "module_d_output" / "manifest.json"
PERCEPTION_DIR = BASE_DIR / "results" / "perception_output"
REFINEMENT_DIR = BASE_DIR / "results" / "refinement_output"

app = FastAPI()
tasks: dict[str, dict[str, object]] = {}
tasks_lock = Lock()

# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

def run_command(
    command: list[str],
    execution_logs: list[dict[str, object]],
    *,
    stage: str,
    sample_id: str | None = None,
) -> None:
    """Execute one pipeline command and always preserve its output."""
    completed = subprocess.run(
        command,
        cwd=BASE_DIR,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    execution_logs.append(
        {
            "stage": stage,
            "sample_id": sample_id,
            "command": subprocess.list2cmdline(command),
            "return_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

def update_task(task_id: str, **changes: object) -> None:
    with tasks_lock:
        tasks[task_id].update(changes)


def run_pipeline_task(
    task_id: str,
    target_sample_id: str | None = None,
) -> None:
    execution_logs: list[dict[str, object]] = []
    python = sys.executable

    try:
        update_task(
            task_id,
            status="running",
            progress=0,
            current_step="Preparing manifest...",
        )
        run_command(
            [
                python,
                "tools/prepare_module_d_manifest.py",
                "--input-dir",
                "data/module_d_output",
                "--output",
                "data/module_d_output/manifest.json",
            ],
            execution_logs,
            stage="prepare_manifest",
        )
        update_task(task_id, logs=deepcopy(execution_logs))

        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Manifest does not contain a non-empty samples list")

        batch_samples: list[dict[str, str]] = []
        sample_ids: set[str] = set()
        for sample in samples:
            if not isinstance(sample, dict) or not sample.get("sample_id"):
                raise ValueError("Every manifest sample must contain sample_id")
            sample_id = str(sample["sample_id"])
            if Path(sample_id).name != sample_id:
                raise ValueError(f"Unsafe sample_id in manifest: {sample_id}")
            if target_sample_id is not None and sample_id != target_sample_id:
                continue
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")

            expected_trajectory = (
                MANIFEST_PATH.parent
                / sample_id
                / f"{sample_id}_trajectory.json"
            ).resolve()
            trajectory_value = sample.get("trajectory")
            trajectory_path = (
                (MANIFEST_PATH.parent / str(trajectory_value)).resolve()
                if trajectory_value
                else expected_trajectory
            )
            if not trajectory_path.is_relative_to(BASE_DIR):
                raise ValueError(f"Unsafe trajectory path for sample: {sample_id}")
            if trajectory_path != expected_trajectory:
                raise ValueError(
                    f"Unexpected trajectory path for {sample_id}: "
                    f"expected {expected_trajectory}, got {trajectory_path}"
                )
            if not trajectory_path.is_file():
                raise FileNotFoundError(
                    f"Trajectory not found for {sample_id}: {trajectory_path}"
                )

            sample_ids.add(sample_id)
            batch_samples.append(
                {
                    "sample_id": sample_id,
                    "trajectory": trajectory_path.relative_to(BASE_DIR).as_posix(),
                }
            )

        if not batch_samples:
            if target_sample_id is not None:
                raise ValueError(
                    f"Sample not found in manifest: {target_sample_id}"
                )
            raise ValueError("Manifest contains no processable samples")

        execution_logs.append(
            {
                "stage": "parse_manifest",
                "sample_id": None,
                "command": None,
                "return_code": 0,
                "stdout": (
                    f"Loaded {len(batch_samples)} sample(s): "
                    + ", ".join(item["sample_id"] for item in batch_samples)
                ),
                "stderr": "",
            }
        )
        update_task(
            task_id,
            sample_count=len(batch_samples),
            logs=deepcopy(execution_logs),
        )

        PERCEPTION_DIR.mkdir(parents=True, exist_ok=True)
        total_steps = 1 + len(batch_samples) * 3
        completed_steps = 1

        for item in batch_samples:
            sample_id = item["sample_id"]
            perception_output = (
                f"results/perception_output/{sample_id}.json"
            )
            samples_output = (
                f"results/c_prepare_sample/{sample_id}_samples.jsonl"
            )
            samples_pretty_output = (
                f"results/c_prepare_sample/{sample_id}_samples.pretty.json"
            )
            refined_output = (
                f"results/refinement_output/{sample_id}_refined.jsonl"
            )
            refined_pretty_output = (
                f"results/refinement_output/{sample_id}_refined.pretty.json"
            )

            update_task(
                task_id,
                progress=round((completed_steps / total_steps) * 100),
                current_step=f"Running perception for {sample_id}...",
            )
            run_command(
                [
                    python,
                    "run_video_task.py",
                    "--manifest",
                    "data/module_d_output/manifest.json",
                    "--sample-id",
                    sample_id,
                    "--view-mode",
                    "fused",
                    "--model",
                    "qwen3-vl-flash",
                    "--rewriter",
                    "llm",
                    "--rewrite-model",
                    "qwen3-vl-flash",
                    "--output",
                    perception_output,
                ],
                execution_logs,
                stage="perception",
                sample_id=sample_id,
            )
            completed_steps += 1
            update_task(
                task_id,
                progress=round((completed_steps / total_steps) * 100),
                current_step=f"Preparing sample {sample_id}...",
                logs=deepcopy(execution_logs),
            )

            run_command(
                [
                    python,
                    "-m",
                    "src.module_c.prepare_samples",
                    "--perception-file",
                    perception_output,
                    "--motion-path",
                    item["trajectory"],
                    "--motion-fps",
                    "30",
                    "--motion-tracks",
                    "Root,Hand_R,Hand_L",
                    "--output",
                    samples_output,
                    "--pretty-output",
                    samples_pretty_output,
                    "--sample-level",
                    "video",
                ],
                execution_logs,
                stage="prepare_samples",
                sample_id=sample_id,
            )
            completed_steps += 1
            update_task(
                task_id,
                progress=round((completed_steps / total_steps) * 100),
                current_step=f"Refining sample {sample_id}...",
                logs=deepcopy(execution_logs),
            )

            run_command(
                [
                    python,
                    "-m",
                    "src.module_c.run_refinement",
                    "--config",
                    "configs/module_c_default.yaml",
                    "--input",
                    samples_output,
                    "--output",
                    refined_output,
                    "--pretty-output",
                    refined_pretty_output,
                ],
                execution_logs,
                stage="refinement",
                sample_id=sample_id,
            )
            completed_steps += 1
            update_task(
                task_id,
                progress=round((completed_steps / total_steps) * 100),
                current_step=f"Completed {sample_id}",
                logs=deepcopy(execution_logs),
            )

        update_task(
            task_id,
            status="completed",
            progress=100,
            current_step=f"Completed {len(batch_samples)} samples",
            logs=deepcopy(execution_logs),
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr or error.stdout or str(error)
        update_task(
            task_id,
            status="failed",
            current_step="Pipeline failed",
            error=detail,
            logs=deepcopy(execution_logs),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        update_task(
            task_id,
            status="failed",
            current_step="Pipeline failed",
            error=str(error),
            logs=deepcopy(execution_logs),
        )
    except Exception as error:
        update_task(
            task_id,
            status="failed",
            current_step="Pipeline failed",
            error=f"{type(error).__name__}: {error}",
            logs=deepcopy(execution_logs),
        )


@app.post("/run-pipeline")
def run_pipeline(
    background_tasks: BackgroundTasks,
    target_sample_id: str | None = None,
):
    if target_sample_id is not None:
        target_sample_id = target_sample_id.strip()
        if not target_sample_id:
            raise HTTPException(
                status_code=400,
                detail="target_sample_id must not be empty",
            )
        if Path(target_sample_id).name != target_sample_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid target_sample_id",
            )

    task_id = uuid4().hex
    with tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "current_step": "Waiting to start...",
            "sample_count": 0,
            "target_sample_id": target_sample_id,
            "logs": [],
            "error": None,
        }
    background_tasks.add_task(
        run_pipeline_task,
        task_id,
        target_sample_id,
    )
    return {
        "success": True,
        "task_id": task_id,
        "status": "queued",
        "target_sample_id": target_sample_id,
    }


@app.get("/get-status/{task_id}")
def get_status(task_id: str):
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return deepcopy(task)

@app.get("/get-results")
def get_results():
    result_files = sorted(REFINEMENT_DIR.glob("*_refined.pretty.json"))
    if not result_files:
        return {"success": False, "message": "No results found yet"}

    results: list[dict[str, object]] = []
    try:
        for result_file in result_files:
            with result_file.open("r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                raise ValueError(
                    f"Refinement output must be a list: {result_file}"
                )
            if not all(isinstance(record, dict) for record in records):
                raise ValueError(
                    f"Invalid refinement record in: {result_file}"
                )
            results.extend(records)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return {
        "success": True,
        "results": results,
        "result_files": [path.name for path in result_files],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)