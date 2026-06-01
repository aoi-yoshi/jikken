"""
Step 6: Flower クライアント（ローカル学習→LoRA+ヘッドを送信）。
環境変数 THESIS_CLIENT_ID=0/1 でデータ分割を切り替える。
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os

import torch
from torch.amp import autocast
from tqdm import tqdm

from src.config_loader import merged_config
from src.dataset_manifest import load_sample_row, read_manifest_filtered, split_clients, stratified_subset
from src.fl_utils import trainable_state_vector, vector_to_trainable_state
from src.logging_utils import RunLogger
from src.metrics import set_seed
from src.paths import ensure_dirs
from src.peft_setup import attach_dual_lora, lora_params_for_adapter, sync_surrogate_from_client
from src.train_common import device_or_auto, iter_row_chunks, load_samples
from src.vl_model import build_model, logits_loss, unfreeze_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--server", default="127.0.0.1:8080")
    args = ap.parse_args()
    cfg = merged_config()
    cid = int(os.environ.get("THESIS_CLIENT_ID", "0"))
    set_seed(int(cfg["train"]["seed"]) + cid * 1000)

    device = torch.device(device_or_auto(True))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    art = cfg["artifacts"]
    run_dir = Path(art["runs"]) / f"step06_fl_client_{cid}"
    ensure_dirs(run_dir)
    log = RunLogger(run_dir, name="fl_client")
    log.log_meta({"step": 6, "client_id": cid})

    manifest = Path(dcfg["manifest_path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    rows = read_manifest_filtered(manifest, dcfg)
    rows = stratified_subset(rows, int(tcfg.get("max_train_samples", 200)), int(cfg["train"]["seed"]))
    a, b = split_clients(rows, int(cfg["train"]["seed"]))
    train_rows = a if cid == 0 else b

    model, processor = build_model(cfg, device)
    unfreeze_backbone(model)
    model.backbone = attach_dual_lora(model.backbone, cfg, client="client", surrogate="surrogate")
    model.backbone.set_adapter("client")

    client_lora = lora_params_for_adapter(model.backbone, "client")
    params = list(model.classifier.parameters()) + client_lora
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))

    prompt = str(dcfg.get("image_prompt", ""))
    max_length = int(tcfg.get("max_length", 256))
    bs = int(tcfg.get("batch_size", 1))
    local_epochs = int(cfg["fl"].get("local_epochs", int(tcfg.get("num_epochs", 1))))

    import flwr as fl

    class Client(fl.client.NumPyClient):
        def __init__(self) -> None:
            names, _ = trainable_state_vector(model)
            self._names = names

        def get_parameters(self, config):
            names, vec = trainable_state_vector(model)
            self._names = names
            return [vec]

        def set_parameters(self, parameters):
            if parameters is None:
                return
            if isinstance(parameters, list) and len(parameters) == 1:
                vec = parameters[0]
                if self._names:
                    vector_to_trainable_state(model, self._names, vec)
            sync_surrogate_from_client(model.backbone, client="client", surrogate="surrogate")

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.train()
            steps = 0
            loss_sum = 0.0
            for _ in range(local_epochs):
                for chunk in iter_row_chunks(train_rows, bs):
                    samples = load_samples(chunk, dcfg)
                    y = torch.tensor([s.label for s in samples], device=device, dtype=torch.long)
                    frames_batch = [s.frames for s in samples]
                    opt.zero_grad(set_to_none=True)
                    model.backbone.set_adapter("client")
                    with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        logits, _ = model.forward_samples(
                            processor, frames_batch, prompt, max_length, output_hidden_states=True
                        )
                        loss = logits_loss(logits.float(), y)
                    loss.backward()
                    opt.step()
                    loss_sum += float(loss.detach().cpu())
                    steps += 1
            names, vec = trainable_state_vector(model)
            self._names = names
            log.log({"cid": cid, "avg_train_loss": loss_sum / max(1, steps)})
            return [vec], len(train_rows), {"avg_train_loss": loss_sum / max(1, steps)}

        def evaluate(self, parameters, config):
            return float(0.0), len(train_rows), {}

    fl.client.start_numpy_client(server_address=args.server, client=Client())


if __name__ == "__main__":
    main()
