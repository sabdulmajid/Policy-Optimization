from __future__ import annotations

import subprocess

import torch


def _parse_nvidia_smi_csv(output: str) -> list[dict[str, object]]:
    gpus: list[dict[str, object]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        index_text, name, total_text, free_text, util_text, driver = [part.strip() for part in line.split(",", maxsplit=5)]
        gpus.append(
            {
                "index": int(index_text),
                "name": name,
                "memory_total_mib": int(total_text),
                "memory_free_mib": int(free_text),
                "utilization_gpu_pct": int(util_text),
                "driver_version": driver,
            }
        )
    return gpus


def query_gpu_inventory() -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        inventory: list[dict[str, object]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            inventory.append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_total_mib": int(props.total_memory // (1024 * 1024)),
                }
            )
        return inventory
    return _parse_nvidia_smi_csv(proc.stdout)


def inspect_gpu_environment(device: torch.device | str) -> dict[str, object]:
    resolved = torch.device(device)
    payload: dict[str, object] = {
        "requested_device": str(resolved),
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": torch.cuda.device_count(),
    }
    if resolved.type != "cuda":
        payload["status"] = "ok"
        payload["gpus"] = query_gpu_inventory()
        return payload
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device '{resolved}', but CUDA is not available.")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if resolved.index is None or resolved.index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device '{resolved}' is out of range for this host.")
    inventory = query_gpu_inventory()
    selected_gpu = next((gpu for gpu in inventory if gpu.get("index") == resolved.index), None)
    if selected_gpu is None:
        props = torch.cuda.get_device_properties(resolved.index)
        selected_gpu = {
            "index": resolved.index,
            "name": props.name,
            "memory_total_mib": int(props.total_memory // (1024 * 1024)),
        }
    payload["status"] = "ok"
    payload["selected_gpu"] = selected_gpu
    payload["gpus"] = inventory
    return payload
