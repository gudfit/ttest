from __future__ import annotations
from typing import Any, Dict, Iterable, Optional


def init_wandb(
    cfg: Any,
    run_name: str,
    tags: Optional[Iterable[str]] = None,
    config_dict: Optional[Dict] = None,
):
    try:
        wb_cfg = getattr(cfg, "logging", None)
        if wb_cfg is None:
            return None
        wb = getattr(wb_cfg, "wandb", None)
        if wb is None or not getattr(wb, "enabled", False):
            return None
        import wandb

        project = getattr(wb, "project", "lldc")
        entity = getattr(wb, "entity", None)
        mode = "online" if getattr(wb, "online", True) else "offline"
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=list(tags or []),
            config=config_dict or {},
            mode=mode,
        )
        return wandb
    except Exception:
        return None


def log_metrics(wb, step: int | None = None, **kwargs):
    if wb is None:
        return
    wb.log(kwargs, step=step)


def log_artifact_file(wb, path: str, name: str, type_: str = "artifact"):
    if wb is None:
        return
    import wandb

    art = wandb.Artifact(name, type=type_)
    art.add_file(path, name=name.split("/")[-1])
    wb.log_artifact(art)
