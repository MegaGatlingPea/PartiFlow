"""Persistent, profile-compatible Ray workers for complete samples."""

from __future__ import annotations

from collections import Counter, deque
import importlib
import math
import os
import re
import sys
import time


GPU_FRACTION = 1 / 16


def cpu_budget():
    limits = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            limits.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        match = re.fullmatch(r"\s*(\d+)(?:\(x\d+\))?(?:,.*)?\s*", os.environ.get(name, ""))
        if match and int(match[1]) > 0:
            limits.append(int(match[1]))
    return max(1, min(limits))


def worker_count(settings, jobs, budget=None):
    threads = settings.get("threads", 1)
    workers = settings.get("workers")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if workers is not None and (type(workers) is not int or workers < 1):
        raise ValueError("workers must be a positive integer or null")
    budget = cpu_budget() if budget is None else budget
    cap = budget // threads
    if cap < 1:
        raise ValueError("postprocess threads exceeds the allocated CPU budget")
    return min(jobs, cap, workers if workers is not None else cap)


def _thread_environment(threads):
    return {name: str(threads) for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    )}


def _profile(item):
    profile = getattr(item, "sdk_profile", None)
    if profile is None:
        profile = getattr(item, "runtime_profile", "standard_beta")
    if not isinstance(profile, str) or not profile:
        raise ValueError("postprocess item requires a nonempty SDK profile")
    return profile, bool(getattr(item, "needs_gpu", False))


def _filesystem_error(error):
    if isinstance(error, OSError):
        return error
    cause = getattr(error, "cause", None)
    return cause if isinstance(cause, OSError) else None


def dispatch_items(items, settings, *, on_result=None, ray_module=None):
    """Return ordered results while compatible idle actors take the next sample.

    Only metadata crosses the Ray boundary. Geometry recovery is delegated to
    the runner, which owns publication and the small per-sample records.
    """
    items = list(items)
    if not items:
        return [], []
    from tqdm import tqdm
    from .runner import PostprocessWorker, recover_actor_failure

    timeout = settings.get("timeout", 1800)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    threads = settings.get("threads", 1)
    budget = cpu_budget()
    count = worker_count(settings, len(items), budget)
    waiting = {}
    for index, item in enumerate(items):
        waiting.setdefault(_profile(item), deque()).append((index, item))
    ray = importlib.import_module("ray") if ray_module is None else ray_module
    owned_runtime = not ray.is_initialized()
    actors, ready, active = {}, {}, {}
    results = [None] * len(items)
    audit = []
    pending_termination = set()
    progress = tqdm(total=len(items), desc="Postprocess", unit="sample", file=sys.stderr)

    def await_terminal(references):
        # A killed actor must stop writing before the caller can roll back files.
        pending = set(references)
        pending_termination.update(pending)
        while pending:
            finished, _ = ray.wait(list(pending), num_returns=len(pending), timeout=min(timeout, 60))
            pending.difference_update(finished)
            pending_termination.difference_update(finished)

    def kill_actor(actor, references=()):
        ray.kill(actor, no_restart=True)
        await_terminal(references)

    def actor_stats(actor, info, reason=None):
        try:
            ref = actor.stats.remote()
            finished, _ = ray.wait([ref], num_returns=1, timeout=timeout)
            stats = ray.get(ref) if finished else {"stats_error": "worker stats timeout"}
        except Exception as exc:
            stats = {"stats_error": f"{type(exc).__name__}: {exc}"}
        entry = {**info, **stats}
        if reason:
            entry["retired_reason"] = reason
        audit.append(entry)

    def retire(actor, reason="profile complete"):
        info = actors[actor]
        actor_stats(actor, info, reason)
        kill_actor(actor)
        actors.pop(actor)

    def publish(index, result):
        if not isinstance(result, dict):
            raise TypeError("postprocess worker must return a result mapping")
        results[index] = result
        if on_result is not None:
            on_result(index, result)
        progress.update(1)

    try:
        gpu_items = sum(len(queue) for key, queue in waiting.items() if key[1])
        if owned_runtime:
            kwargs = dict(address="local", num_cpus=count * threads,
                          include_dashboard=False, log_to_driver=False)
            if not gpu_items:
                kwargs["num_gpus"] = 0
            ray.init(**kwargs)
        resources = ray.cluster_resources() if owned_runtime else ray.available_resources()
        budget = min(budget, math.floor(resources.get("CPU", 0)))
        count = worker_count(settings, len(items), budget)
        gpu_slots = math.floor(resources.get("GPU", 0) / GPU_FRACTION)
        if gpu_items and gpu_slots < 1:
            raise ValueError("Ray has insufficient GPU resources for postprocess")
        count = min(count, len(items) - gpu_items + min(gpu_items, gpu_slots))
        remote_types = {}

        def available_key():
            starters = Counter(info["key"] for actor, _ in ready.values()
                               if (info := actors.get(actor)) is not None)
            gpu_actors = sum(info["key"][1] for info in actors.values())
            candidates = []
            for key, queue in waiting.items():
                if len(queue) <= starters[key] or (key[1] and gpu_actors >= gpu_slots):
                    continue
                candidates.append((queue[starters[key]][0], key))
            return min(candidates)[1] if candidates else None

        def start_actor(key):
            if key[1] not in remote_types:
                remote_types[key[1]] = ray.remote(
                    num_cpus=threads, num_gpus=GPU_FRACTION if key[1] else 0,
                    max_restarts=0, max_task_retries=0,
                    runtime_env={"env_vars": _thread_environment(threads)},
                )(PostprocessWorker)
            config = {**settings, "sdk_profile": key[0], "needs_gpu": key[1]}
            actor = remote_types[key[1]].remote(config)
            actors[actor] = {"key": key, "sdk_profile": key[0],
                             "needs_gpu": key[1], "submitted": 0}
            ready[actor.ready.remote()] = (actor, time.monotonic())

        def refill():
            while len(actors) < count:
                key = available_key()
                if key is None:
                    break
                start_actor(key)

        def submit_or_retire(actor):
            queue = waiting[actors[actor]["key"]]
            if not queue:
                retire(actor)
                refill()
                return
            index, item = queue.popleft()
            ref = actor.process.remote(item)
            actors[actor]["submitted"] += 1
            active[ref] = (actor, index, item, time.monotonic())

        def lost_sample(ref, actor, index, item, reason):
            info = actors[actor]
            kill_actor(actor, [ref])
            actors.pop(actor)
            audit.append({**info, "retired_reason": reason})
            effective_settings = {**settings, "sdk_profile": info["sdk_profile"],
                                  "needs_gpu": info["needs_gpu"]}
            result = recover_actor_failure(item, effective_settings, reason)
            publish(index, result)
            refill()

        def consume(ref):
            if ref in ready:
                actor, _ = ready.pop(ref)
                try:
                    info = ray.get(ref)
                except Exception as exc:
                    raise RuntimeError(f"postprocess worker startup failed: {exc}") from exc
                profile = info.get("sdk_profile", info.get("profile")) if isinstance(info, dict) else None
                if (not isinstance(info, dict) or type(info.get("pid")) is not int
                        or info["pid"] <= 0 or profile != actors[actor]["sdk_profile"]):
                    raise RuntimeError("postprocess worker returned invalid startup information")
                actors[actor].update(info)
                submit_or_retire(actor)
                return
            actor, index, item, _ = active.pop(ref)
            try:
                result = ray.get(ref)
            except Exception as exc:
                filesystem_error = _filesystem_error(exc)
                if filesystem_error is not None:
                    raise filesystem_error
                lost_sample(ref, actor, index, item, f"worker failed: {type(exc).__name__}: {exc}")
            else:
                if not isinstance(result, dict):
                    lost_sample(ref, actor, index, item, "worker returned a non-mapping result")
                else:
                    publish(index, result)
                    submit_or_retire(actor)

        refill()
        while ready or active:
            deadlines = [started + timeout for _, started in ready.values()]
            deadlines.extend(value[3] + timeout for value in active.values())
            delay = max(0, min(deadlines) - time.monotonic())
            finished, _ = ray.wait([*ready, *active], num_returns=1, timeout=delay)
            for ref in finished:
                consume(ref)
            while True:
                now = time.monotonic()
                expired_ready = [ref for ref, (_, started) in ready.items() if now - started >= timeout]
                expired_active = [ref for ref, value in active.items() if now - value[3] >= timeout]
                expired = [*expired_ready, *expired_active]
                if not expired:
                    break
                finished, _ = ray.wait(expired, num_returns=len(expired), timeout=0)
                if finished:
                    for ref in finished:
                        consume(ref)
                    continue
                if expired_ready:
                    raise RuntimeError(f"postprocess worker startup timeout after {timeout} seconds")
                ref = expired_active[0]
                actor, index, item, _ = active.pop(ref)
                lost_sample(ref, actor, index, item, f"worker timeout after {timeout} seconds")
        if any(queue for queue in waiting.values()) or any(result is None for result in results):
            raise RuntimeError("postprocess queue ended before all samples completed")
        return results, audit
    finally:
        try:
            references = [*ready, *active, *pending_termination]
            for actor in list(actors):
                ray.kill(actor, no_restart=True)
            await_terminal(references)
            if owned_runtime:
                ray.shutdown()
        finally:
            progress.close()
