"""Local stage diagnostics without synchronizing CUDA or resetting allocator statistics."""

import contextvars
from contextlib import contextmanager
import hashlib
import logging
import os
import threading
import time

import psutil
import torch

import comfy.model_management
import comfy.patcher_extension


def log_memory(label, device):
    if os.environ.get("SELFLIFT_MEMORY_LOG", "0") != "1":
        return
    device = torch.device(device)
    fields = []
    try:
        fields.append(f"process_rss={psutil.Process().memory_info().rss / 2**20:.2f} MiB")
        fields.append(f"system_available={psutil.virtual_memory().available / 2**20:.2f} MiB")
    except (psutil.Error, OSError) as error:
        fields.append(f"host_memory_unavailable={error}")
    if device.type == "cuda" and torch.cuda.is_initialized():
        try:
            free, total = torch.cuda.mem_get_info(device)
            fields.extend([
                f"allocated={torch.cuda.memory_allocated(device) / 2**20:.2f} MiB",
                f"reserved={torch.cuda.memory_reserved(device) / 2**20:.2f} MiB",
                f"process_peak_allocated={torch.cuda.max_memory_allocated(device) / 2**20:.2f} MiB",
                f"device_used={((total - free) / 2**20):.2f} MiB",
                f"device_free={free / 2**20:.2f} MiB",
            ])
        except RuntimeError as error:
            fields.append(f"cuda_memory_unavailable={error}")
    logging.info("[SelfLift memory] %s device=%s; %s", label, device, "; ".join(fields))


# Use Comfy's patcher wrappers rather than replacing global sampler functions.
# Context-local ownership makes copied wrappers inert outside this invocation.
_active_stage = contextvars.ContextVar("selflift_diagnostic_stage", default=None)
_active_sample = contextvars.ContextVar("selflift_diagnostic_sample", default=None)
_scope_lock = threading.RLock()
_scopes = {}


def _patch_token(patches):
    """Process-local identity of the weight-patch stack; never read tensor values."""
    digest = hashlib.sha256()
    for key, entries in sorted(patches.items(), key=lambda item: str(item[0])):
        digest.update(str(key).encode())
        for entry in entries:
            # Comfy stores (patch strength, patch payload, model strength, offset, function).
            metadata = tuple(value if isinstance(value, (int, float, type(None))) else id(value)
                             for value in entry)
            digest.update(str(metadata).encode())
    return digest.hexdigest()[:12]


def _snapshot(model, device, patch_tokens):
    fields = []
    try:
        fields.append(f"rss_mib={psutil.Process().memory_info().rss / 2**20:.1f}")
    except (psutil.Error, OSError) as error:
        fields.append(f"host_memory_unavailable={type(error).__name__}")
    try:
        if model is not None:
            base = getattr(model, "model", None)
            patches = getattr(model, "patches", {})
            patch_key = (id(patches), str(getattr(model, "patches_uuid", None)))
            if patch_key not in patch_tokens:
                patch_tokens[patch_key] = _patch_token(patches)
            fields.extend([f"patcher={id(model):x}", f"base={id(base):x}",
                           f"patch_uuid={getattr(model, 'patches_uuid', None)}",
                           f"applied_uuid={getattr(base, 'current_weight_patches_uuid', None)}",
                           f"patch_keys={len(patches)}", f"patch_token={patch_tokens[patch_key]}"])
            for name in ("is_dynamic", "loaded_size", "loaded_ram_size", "current_loaded_device"):
                method = getattr(model, name, None)
                if callable(method):
                    value = method()
                    unit = "_bytes" if name in ("loaded_size", "loaded_ram_size") else ""
                    fields.append(f"{name}{unit}={value}")
            tracked = [item.model for item in tuple(comfy.model_management.current_loaded_models)]
            fields.append(f"comfy_tracked={any(item is model for item in tracked)}")
            fields.append(f"base_tracked={any(getattr(item, 'model', None) is base for item in tracked)}")
    except Exception as error:
        fields.append(f"model_snapshot_unavailable={type(error).__name__}")
    try:
        if device is not None:
            device = torch.device(device)
            fields.append(f"device={device}")
            if device.type == "cuda" and torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info(device)
                fields.extend([f"cuda_alloc_mib={torch.cuda.memory_allocated(device) / 2**20:.1f}",
                               f"cuda_reserved_mib={torch.cuda.memory_reserved(device) / 2**20:.1f}",
                               f"cuda_free_mib={free / 2**20:.1f}", f"cuda_total_mib={total / 2**20:.1f}"])
    except Exception as error:
        # Observability must not turn an optional backend/memory probe into a failed generation.
        fields.append(f"snapshot_unavailable={type(error).__name__}")
    return " ".join(fields)


class _DiagnosticStage:
    def __init__(self, label, model, device):
        self.label, self.model, self.device = label, model, device
        self.started = self.previous = time.monotonic()
        self.owner = threading.get_ident()
        self.samples = 0
        self.patch_tokens = {}
        self.enabled = os.environ.get("SELFLIFT_DIAGNOSTICS", "1") != "0"
        self.trace = f"{time.monotonic_ns():x}"

    def event(self, event, model=None, **values):
        if not self.enabled:
            return
        now = time.monotonic()
        logging.info("[SelfLift diagnostic] trace=%s stage=%s event=%s mono=%.6f elapsed=%.3fs delta=%.3fs %s %s",
                     self.trace, self.label, event, now, now - self.started, now - self.previous,
                     " ".join(f"{key}={value}" for key, value in values.items()),
                     _snapshot(self.model if model is None else model, self.device, self.patch_tokens))
        self.previous = now


def _current_stage():
    stage = _active_stage.get()
    return stage if stage is not None and stage.owner == threading.get_ident() else None


def _outer_sample(executor, *args, **kwargs):
    stage = _current_stage()
    if stage is None:
        return executor(*args, **kwargs)
    stage.samples += 1
    sample = {"number": stage.samples, "start": time.monotonic()}
    token = _active_sample.set(sample)
    status = "error"
    try:
        result = executor(*args, **kwargs)
        status = "ok"
        return result
    finally:
        stage.event("sample_end", model=executor.class_obj.model_patcher,
                    sample=sample["number"], status=status, wall_s=f"{time.monotonic() - sample['start']:.3f}",
                    **{key: value for key, value in sample.items() if key.endswith("_s")},
                    compile_s="unknown", cuda_sync=False)
        _active_sample.reset(token)


def _prepare_sampling(executor, model, *args, **kwargs):
    stage, sample = _current_stage(), _active_sample.get()
    if stage is None or sample is None:
        return executor(model, *args, **kwargs)
    stage.event("prepare_start", model=model, sample=sample["number"])
    started = time.monotonic()
    status = "error"
    try:
        result = executor(model, *args, **kwargs)
        status = "ok"
        return result
    finally:
        sample["prepare_wall_s"] = f"{time.monotonic() - started:.3f}"
        stage.event("prepare_end", model=model, sample=sample["number"], status=status, wall_s=sample["prepare_wall_s"])


def _sampler_sample(executor, *args, **kwargs):
    stage, sample = _current_stage(), _active_sample.get()
    if stage is None or sample is None:
        return executor(*args, **kwargs)
    sample["sampler_entry_s"] = f"{time.monotonic() - sample['start']:.3f}"
    callback = args[3] if len(args) > 3 else kwargs.get("callback")
    if callback is not None:
        def timed_callback(*callback_args, **callback_kwargs):
            started = time.monotonic()
            first = "first_callback_entry_s" not in sample
            if first:
                sample["first_callback_entry_s"] = f"{started - sample['start']:.3f}"
            try:
                return callback(*callback_args, **callback_kwargs)
            finally:
                if first:
                    sample["first_callback_wall_s"] = f"{time.monotonic() - started:.3f}"
        if len(args) > 3:
            args = (*args[:3], timed_callback, *args[4:])
        else:
            kwargs = {**kwargs, "callback": timed_callback}
    return executor(*args, **kwargs)


def _predict_noise(executor, *args, **kwargs):
    stage, sample = _current_stage(), _active_sample.get()
    if stage is None or sample is None or "first_predict_wall_s" in sample:
        return executor(*args, **kwargs)
    started = time.monotonic()
    sample["first_predict_entry_s"] = f"{started - sample['start']:.3f}"
    try:
        return executor(*args, **kwargs)
    finally:
        sample["first_predict_wall_s"] = f"{time.monotonic() - started:.3f}"
        stage.event("first_predict_end", model=executor.class_obj.model_patcher,
                    sample=sample["number"], wall_s=sample["first_predict_wall_s"])


@contextmanager
def diagnostic_scope(label, model=None, device=None):
    stage = _DiagnosticStage(label, model, device if device is not None else getattr(model, "load_device", None))
    if not stage.enabled:
        yield stage
        return
    # Swap a copied wrapper dictionary atomically, preserving the actual patcher
    # identity (cloning it could itself change Comfy's residency/clone handling).
    attached = model is not None and hasattr(model, "wrappers")
    if attached:
        with _scope_lock:
            key = id(model)
            if key not in _scopes:
                original = model.wrappers
                installed = comfy.patcher_extension.copy_nested_dicts(original)
                for kind, wrapper in (("outer_sample", _outer_sample), ("prepare_sampling", _prepare_sampling),
                                      ("sampler_sample", _sampler_sample), ("predict_noise", _predict_noise)):
                    # Prepend so timing includes existing patcher wrappers (e.g. tiling).
                    existing = installed.get(kind, {})
                    installed[kind] = {"selflift_diagnostics": [wrapper], **existing}
                model.wrappers = installed
                _scopes[key] = [original, installed, 0]
            _scopes[key][2] += 1
    token = _active_stage.set(stage)
    stage.event("start", timing="monotonic_wall_no_cuda_sync", residency="loaded_size_bytes_not_full_residency")
    status = "error"
    try:
        yield stage
        status = "ok"
    finally:
        stage.event("end", samples=stage.samples, status=status)
        _active_stage.reset(token)
        if attached:
            with _scope_lock:
                original, installed, users = _scopes[key]
                _scopes[key][2] = users - 1
                if users == 1:
                    if model.wrappers is installed:
                        model.wrappers = original
                    del _scopes[key]
