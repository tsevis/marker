"""Persistent marker worker process.

Loads the models once, then converts PDFs on demand — one JSON request per line
on stdin. Keeping the process alive across a batch is the single biggest win for
multi-file conversion: model loading costs several seconds and would otherwise be
paid again for every file.

Protocol
    stdin   {"pdf": "…", "out_dir": "…"}            one request per line
    stdout  {"ready": true, "device": "mps", …}     once, after models load
            {"ok": true, "out": "…"}                per request
            {"ok": false, "error": "…"}             per request
    stderr  logs and tqdm progress bars, parsed by the GUI for live progress

Run it directly to see the tuning this machine would get:
    python marker_worker.py --show-tuning
"""

import json
import os
import sys
from pathlib import Path

# Batch sizes tuned for Apple Silicon unified memory, keyed by the minimum RAM
# tier they apply to. surya's MPS defaults (detector 8, layout 4, recognition 64)
# are sized for a base 8 GB M-series chip; on a machine with a large unified
# memory pool those tiny batches leave the GPU idling between dispatches, so we
# push them toward the CUDA-class values instead.
APPLE_SILICON_TIERS: tuple[tuple[int, dict[str, str]], ...] = (
    (
        64,
        {
            "DETECTOR_BATCH_SIZE": "36",
            "LAYOUT_BATCH_SIZE": "32",
            "OCR_ERROR_BATCH_SIZE": "64",
        },
    ),
    (
        32,
        {
            "DETECTOR_BATCH_SIZE": "24",
            "LAYOUT_BATCH_SIZE": "16",
            "OCR_ERROR_BATCH_SIZE": "32",
        },
    ),
    (
        16,
        {
            "DETECTOR_BATCH_SIZE": "12",
            "LAYOUT_BATCH_SIZE": "8",
            "OCR_ERROR_BATCH_SIZE": "16",
        },
    ),
)


def unified_memory_gb() -> int:
    """Physical RAM in GB (unified memory on Apple Silicon); 0 if unknown."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024**3)
    except (ValueError, OSError, AttributeError):
        return 0


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and os.uname().machine == "arm64"


def apple_silicon_env() -> dict[str, str]:
    """Environment tuning for Apple Silicon, or {} on other platforms.

    Returns only variables the caller has not already set, so an explicit
    override in the environment always wins.
    """
    if not is_apple_silicon():
        return {}

    tuning = {
        # transformers uses .isin, which MPS lacks; without this, model load fails.
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        # A forked tokenizer pool deadlocks and gains nothing here.
        "TOKENIZERS_PARALLELISM": "false",
        # Detection postprocessing is pure CPU and caps itself at 8 workers.
        "DETECTOR_POSTPROCESSING_CPU_WORKERS": str(min(16, os.cpu_count() or 8)),
    }
    ram = unified_memory_gb()
    for minimum, batch_sizes in APPLE_SILICON_TIERS:
        if ram >= minimum:
            tuning.update(batch_sizes)
            break

    return {k: v for k, v in tuning.items() if k not in os.environ}


def apply_tuning() -> dict[str, str]:
    """Apply the tuning to os.environ. Must run before importing marker/surya,
    which read these values into pydantic settings at import time."""
    tuning = apple_silicon_env()
    os.environ.update(tuning)
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    return tuning


def convert_one(models: dict, pdf: str, out_dir: str) -> str:
    """Convert a single PDF with already-loaded models. Returns the markdown path."""
    from marker.config.parser import ConfigParser
    from marker.output import save_output

    config_parser = ConfigParser({"output_dir": out_dir, "output_format": "markdown"})
    converter = config_parser.get_converter_cls()(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(pdf)
    out_folder = config_parser.get_output_folder(pdf)
    base = config_parser.get_base_filename(pdf)
    save_output(rendered, out_folder, base)
    return str(Path(out_folder) / f"{base}.md")


def reclaim_memory() -> None:
    """Drop cached MPS blocks between files so a long batch does not creep."""
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - never let cleanup kill the worker
        pass


def reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> None:
    tuning = apply_tuning()

    from marker.logger import configure_logging, get_logger
    from marker.models import create_model_dict
    from marker.settings import settings

    configure_logging()
    logger = get_logger()

    models = create_model_dict()
    reply({"ready": True, "device": settings.TORCH_DEVICE_MODEL, "tuning": tuning})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            reply({"ok": False, "error": f"Bad request: {e}"})
            continue
        try:
            out = convert_one(models, request["pdf"], request["out_dir"])
            logger.info(f"Saved markdown to {out}")
            reply({"ok": True, "out": out})
        except Exception as e:  # noqa: BLE001 - one bad file must not kill the worker
            logger.error(f"Failed to convert {request.get('pdf')}: {e}")
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            reclaim_memory()


if __name__ == "__main__":
    if "--show-tuning" in sys.argv:
        print(f"Apple Silicon: {is_apple_silicon()}  RAM: {unified_memory_gb()} GB")
        for k, v in (apple_silicon_env() or {"(none)": "platform defaults"}).items():
            print(f"  {k}={v}")
    else:
        main()
