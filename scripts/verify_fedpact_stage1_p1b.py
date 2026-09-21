"""P1-Bのselected proxyをcheckpointから別processで再計算する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from PIL import Image
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config_loader import load_yaml, merged_config  # noqa: E402
from src.fl_utils import load_fl_checkpoint, vector_to_trainable_state  # noqa: E402
from src.peft_setup import attach_dual_lora, lora_params_for_adapter  # noqa: E402
from src.step07.fedpact_layer2 import proxy_match_metrics  # noqa: E402
from src.train_common import device_or_auto  # noqa: E402
from src.vl_model import build_model, unfreeze_backbone  # noqa: E402


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else _ROOT / path).resolve()


def _load_state(model, checkpoint: Path) -> None:
    names, vector, _ = load_fl_checkpoint(checkpoint)
    vector_to_trainable_state(model, names, vector)


@torch.no_grad()
def _predict(model, processor, frames, prompt, max_length) -> list[float]:
    model.eval()
    model.backbone.set_adapter("client")
    logits, _ = model.forward_batch(
        processor, frames, prompt, max_length, output_hidden_states=True
    )
    return logits[0].detach().float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/fedpact_stage1.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = _resolve(args.run)
    trace = json.loads((run_dir / "server" / "layer2_trace.json").read_text(encoding="utf-8"))
    pair = json.loads((run_dir / "server" / "proxy_pair.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = next(
        item
        for item in trace["proxy_candidates"]
        if item["candidate_id"] == trace["selected_candidate_id"]
    )

    cfg = merged_config(load_yaml(args.config))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    scfg = cfg["fedpact_stage1"]
    parent_run = _resolve(trace["parent_p1a_run"])
    m_g_0 = _resolve(str(scfg["initial_global_checkpoint"]))
    m_g_1 = parent_run / "server" / "M_G_1.pt"
    device_name = device_or_auto(True) if args.device == "auto" else args.device
    device = torch.device(device_name)
    model_id = str(cfg.get("model", {}).get("client_id") or cfg["model"]["id"])
    model, processor = build_model(cfg, device, model_id=model_id)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(
        model.backbone, cfg, client="client", surrogate="surrogate"
    )
    model.backbone.set_adapter("client")
    lora_params_for_adapter(model.backbone, "client")
    frames = [Image.open(run_dir / relative).convert("RGB") for relative in pair["x_tilde_{k,r}"]["frame_files"]]
    prompt = str(dcfg["image_prompt"])
    max_length = int(tcfg.get("max_length", 256))
    try:
        _load_state(model, m_g_0)
        before_logits = _predict(model, processor, frames, prompt, max_length)
        _load_state(model, m_g_1)
        after_logits = _predict(model, processor, frames, prompt, max_length)
    finally:
        for frame in frames:
            frame.close()
    metrics = proxy_match_metrics(
        trace["prototype"],
        global_before_logits=before_logits,
        global_after_logits=after_logits,
        lambda_post=float(trace["configuration"]["lambda_post"]),
        lambda_delta=float(trace["configuration"]["lambda_delta"]),
    )
    recorded = selected["metrics"]
    differences = {
        key: abs(float(metrics[key]) - float(recorded[key]))
        for key in ("post_kl", "delta_mse", "proxy_objective")
    }
    logits_match = bool(
        np.allclose(before_logits, selected["global_before_logits"], rtol=1e-5, atol=1e-6)
        and np.allclose(after_logits, selected["global_after_logits"], rtol=1e-5, atol=1e-6)
    )
    metrics_match = all(
        np.isclose(float(metrics[key]), float(recorded[key]), rtol=1e-5, atol=1e-6)
        for key in differences
    )
    minimum_candidate_id = min(
        trace["proxy_candidates"],
        key=lambda item: (
            float(item["metrics"]["proxy_objective"]),
            str(item["seed_id"]),
            str(item["transformation"]["name"]),
            str(item["transformation"].get("value")),
        ),
    )["candidate_id"]
    selection_match = minimum_candidate_id == trace["selected_candidate_id"]
    status = "pass" if logits_match and metrics_match and selection_match else "fail"
    verification = {
        "schema": "fedpact.layer2.recompute.v1",
        "run_id": summary["run_id"],
        "status": status,
        "selected_candidate_id": trace["selected_candidate_id"],
        "checkpoint_recomputed_before_logits": before_logits,
        "checkpoint_recomputed_after_logits": after_logits,
        "recomputed_metrics": metrics,
        "recorded_metrics": recorded,
        "absolute_differences": differences,
        "checks": {
            "checkpoint_logits_match": logits_match,
            "metrics_match": metrics_match,
            "selection_is_minimum_objective": selection_match,
        },
    }
    _save_json(run_dir / "server" / "independent_recompute.json", verification)
    summary["independent_recompute"] = verification
    summary["status"] = status
    _save_json(summary_path, summary)
    _save_json(
        run_dir / "run_status.json",
        {"run_id": summary["run_id"], "gate": "P1-B", "status": status},
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

