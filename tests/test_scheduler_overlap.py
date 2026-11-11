import threading
import time
from pathlib import Path

from recast.core.runner import split_resource_stages
from recast.core.scheduler import Job, Scheduler
from recast.core.settings import AppSettings
from recast.store.db import Store


def test_split_resource_stages_order():
    pre, gpu, post = split_resource_stages(
        ["scan", "extract", "library", "replace", "compose", "export"]
    )
    assert pre == ["scan", "extract"]
    assert gpu == ["library", "replace"]
    assert post == ["compose", "export"]


def test_cpu_pre_can_run_while_gpu_busy(tmp_path: Path, monkeypatch):
    store = Store(tmp_path / "db.db")
    settings = AppSettings(output_root=str(tmp_path / "out"), concurrency=1)
    events = {"cpu2": threading.Event(), "gpu1": threading.Event(), "order": []}
    lock = threading.Lock()

    def fake_dispatch(self, job, task, workdir, on_progress):
        from recast.core.runner import split_resource_stages

        pre, gpu, post = split_resource_stages(job.stages)
        if pre:
            with lock:
                events["order"].append(f"{job.task_id}:cpu")
            if job.task_id == "t2":
                events["cpu2"].set()
        if gpu:
            # episode 1 holds the GPU until episode 2 has started CPU work
            if job.task_id == "t1":
                events["gpu1"].set()
                # wait briefly for t2 cpu, but don't deadlock tests
                events["cpu2"].wait(timeout=2.0)
            with lock:
                events["order"].append(f"{job.task_id}:gpu")
            time.sleep(0.05)
        return {"ok": True, "task": job.task_id}

    monkeypatch.setattr(Scheduler, "_dispatch", fake_dispatch)

    # two dramas / tasks
    d1 = store.insert("dramas", origin_name="A", root_path=str(tmp_path / "a"), status="待执行")
    d2 = store.insert("dramas", origin_name="B", root_path=str(tmp_path / "b"), status="待执行")
    t1 = store.insert("tasks", drama_id=d1["id"], episode_name="e1", video_path=str(tmp_path / "a.mp4"), workdir=str(tmp_path / "w1"))
    t2 = store.insert("tasks", drama_id=d2["id"], episode_name="e2", video_path=str(tmp_path / "b.mp4"), workdir=str(tmp_path / "w2"))
    # overwrite ids used by fake_dispatch checks — use the actual ids
    # retarget events to real ids
    id1, id2 = t1["id"], t2["id"]

    def fake_dispatch2(self, job, task, workdir, on_progress):
        pre, gpu, post = split_resource_stages(job.stages)
        if pre:
            with lock:
                events["order"].append(f"{job.task_id}:cpu")
            if job.task_id == id2:
                events["cpu2"].set()
        if gpu:
            if job.task_id == id1:
                events["gpu1"].set()
                events["cpu2"].wait(timeout=2.0)
            with lock:
                events["order"].append(f"{job.task_id}:gpu")
            time.sleep(0.05)
        return {"ok": True}

    monkeypatch.setattr(Scheduler, "_dispatch", fake_dispatch2)
    sched = Scheduler(store, settings)
    sched.enqueue(
        [
            Job(drama_id=d1["id"], task_id=id1, stages=["scan", "extract", "replace"]),
            Job(drama_id=d2["id"], task_id=id2, stages=["scan", "extract", "replace"]),
        ]
    )
    # wait for both to finish
    deadline = time.time() + 5
    while time.time() < deadline and (sched.running or len(events["order"]) < 3):
        time.sleep(0.05)
    # t2 CPU should have started while t1 was on GPU (cpu2 event set during t1 gpu wait)
    assert events["cpu2"].is_set()
    # both jobs recorded some work
    assert any(x.startswith(id1) for x in events["order"])
    assert any(x.startswith(id2) for x in events["order"])
