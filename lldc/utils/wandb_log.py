# lldc/utils/wandb_log.py
from __future__ import annotations


def _cfg_get(cfg, path, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, key, None) if not isinstance(cur, dict) else cur.get(key)
    return default if cur is None else cur


try:
    import wandb as _wandb
except Exception:
    _wandb = None


def start(cfg, run_name: str, tags=None):
    enabled = bool(_cfg_get(cfg, "logging.wandb.enabled", False))
    if not enabled or _wandb is None:
        return None
    project = _cfg_get(cfg, "logging.wandb.project", "lldc")
    entity = _cfg_get(cfg, "logging.wandb.entity", None)
    config = None
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return _wandb.init(
        project=project, entity=entity, name=run_name, tags=tags or [], config=config
    )


def log(metrics: dict, step: int | None = None):
    if _wandb is None:
        return
    if _wandb.run is None:
        return
    _wandb.log(metrics, step=step)


def finish():
    if _wandb is None:
        return
    if _wandb.run is not None:
        _wandb.finish()
