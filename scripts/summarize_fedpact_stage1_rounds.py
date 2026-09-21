from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def short_digest(value: str) -> str:
    return value[:12]


def selected_pair_diagnostics(layer2: dict[str, Any]) -> dict[str, Any]:
    prototypes = {item["prototype_id"]: item for item in layer2["prototypes"]}
    candidates = {item["candidate_id"]: item for item in layer2["proxy_candidates"]}
    rows: list[dict[str, Any]] = []
    for selection in layer2["selections"]:
        prototype = prototypes[selection["prototype_id"]]
        candidate = candidates[selection["candidate_id"]]
        before = softmax(candidate["global_before_logits"])
        after = softmax(candidate["global_after_logits"])
        target_delta_risk = float(prototype["delta_mu_local"][1])
        proxy_delta_risk = float(after[1] - before[1])
        rows.append(
            {
                "prototype_id": prototype["prototype_id"],
                "prototype_class": int(prototype["class_label"]),
                "seed_id": selection["seed_id"],
                "seed_label": int(selection["seed_label"]),
                "seed_label_matches_prototype_class": (
                    int(selection["seed_label"]) == int(prototype["class_label"])
                ),
                "target_delta_risk": target_delta_risk,
                "proxy_delta_risk": proxy_delta_risk,
                "delta_direction_matches": target_delta_risk * proxy_delta_risk >= 0.0,
                "post_kl": float(selection["metrics"]["post_kl"]),
                "delta_mse": float(selection["metrics"]["delta_mse"]),
                "L_match": float(selection["metrics"]["proxy_objective"]),
            }
        )
    return {
        "pairs": rows,
        "seed_label_match_count": sum(row["seed_label_matches_prototype_class"] for row in rows),
        "delta_direction_match_count": sum(row["delta_direction_matches"] for row in rows),
    }


def seed_label_prediction_agreement(layer2: dict[str, Any]) -> dict[str, int]:
    rankings = next(iter(layer2["ranked_seeds_by_prototype"].values()))
    agreement = 0
    for item in rankings:
        predicted = max(range(len(item["updated_probability"])), key=item["updated_probability"].__getitem__)
        agreement += int(predicted == int(item["label"]))
    return {"agreement": agreement, "total": len(rankings)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/fedpact_stage1_20260917",
    )
    args = parser.parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_root = ROOT / "artifacts" / "runs" / "fedpact_stage1"
    run_specs = [
        (1, "p1d_layer1_20260917_01", "p1d_full_20260917_02"),
        (2, "p1d_round2_layer1_20260917_01", "p1d_round2_full_20260917_01"),
        (3, "p1d_round3_layer1_20260917_01", "p1d_round3_full_20260917_02"),
    ]
    rounds: list[dict[str, Any]] = []
    for round_number, layer1_run_id, full_run_id in run_specs:
        layer1_dir = run_root / layer1_run_id
        full_dir = run_root / full_run_id
        layer1 = read_json(layer1_dir / "summary.json")
        full = read_json(full_dir / "summary.json")
        layer2 = read_json(full_dir / "server" / "layer2_trace.json")
        layer3 = read_json(full_dir / "foundation" / "layer3_trace.json")
        diagnostics = selected_pair_diagnostics(layer2)
        seed_agreement = seed_label_prediction_agreement(layer2)
        rounds.append(
            {
                "round": round_number,
                "run_ids": {"layer1": layer1_run_id, "full": full_run_id},
                "status": {"layer1": layer1["status"], "full": full["status"]},
                "global_surrogate": {
                    "before_state_id": layer1["M_G_0"]["state_id"],
                    "after_state_id": layer1["M_G_1"]["state_id"],
                    "before_digests": layer1["M_G_0"]["digests"],
                    "after_digests": layer1["M_G_1"]["digests"],
                    "before_checkpoint_sha256": layer1["configuration"]["initial_checkpoint_file_sha256"],
                    "after_checkpoint_sha256": layer1["M_G_1"]["checkpoint_file_sha256"],
                    "client_lora_update_norms": [
                        float(client["lora_update_norm"]) for client in layer1["clients"]
                    ],
                    "client_head_update_norms": [
                        float(client["head_update_norm"]) for client in layer1["clients"]
                    ],
                    "fedavg_max_abs_diff_offline_recompute": float(
                        layer1["M_G_1"]["max_abs_diff_offline_recompute"]
                    ),
                    "redistribution_digest_match": bool(
                        layer1["pass_conditions"]["redistribution_digest_match"]
                    ),
                },
                "layer2": {
                    "server_seed_count": int(full["configuration"]["server_seed_count"]),
                    "server_seed_label_counts": full["configuration"]["server_seed_label_counts"],
                    "class_unrestricted_retrieval": bool(full["checks"]["class_unrestricted_retrieval"]),
                    "proxy_candidate_count": int(full["configuration"]["proxy_candidate_count"]),
                    "proxy_pair_count": int(full["configuration"]["proxy_pair_count"]),
                    "retrieval_used_updated_global": bool(full["checks"]["retrieval_used_M_G_1"]),
                    "delta_used_round_before_after_global": bool(
                        full["checks"]["proxy_delta_used_same_candidate_M_G_0_M_G_1"]
                    ),
                    "seed_label_vs_updated_global_prediction": seed_agreement,
                    "selected_pair_diagnostics": diagnostics,
                },
                "foundation": {
                    "before_state_id": layer3["state_lineage"]["before"],
                    "after_state_id": layer3["state_lineage"]["after"],
                    "before_digest": full["foundation"]["before_digest"],
                    "after_digest": full["foundation"]["after_digest"],
                    "acceptance_policy": full["foundation"].get("acceptance_policy", "nonincrease"),
                    "accepted_learning_rate": float(full["foundation"]["accepted_learning_rate"]),
                    "weighted_loss_before": float(full["foundation"]["weighted_loss_before"]),
                    "weighted_loss_after": float(full["foundation"]["weighted_loss_after"]),
                    "weighted_loss_nonincrease": bool(
                        full["checks"].get("weighted_loss_nonincrease", False)
                    ),
                    "update_l2_norm_fp32": float(full["foundation"]["update_l2_norm_fp32"]),
                    "classification_head_unchanged": bool(full["checks"]["classification_head_unchanged"]),
                    "non_target_parameter_versions_unchanged": bool(
                        full["checks"]["non_target_parameter_versions_unchanged"]
                    ),
                },
                "privacy": {
                    "raw_data_non_transmission_audit": bool(
                        layer1["pass_conditions"]["raw_data_non_transmission_audit"]
                    ),
                    "client_payload_audits": [
                        client["communication_audit"]["status"] for client in layer1["clients"]
                    ],
                },
                "timings_sec": full["timings_sec"],
            }
        )

    continuity: list[dict[str, Any]] = []
    for previous, current in zip(rounds, rounds[1:]):
        global_digest_match = (
            previous["global_surrogate"]["after_digests"]
            == current["global_surrogate"]["before_digests"]
        )
        global_checkpoint_match = (
            previous["global_surrogate"]["after_checkpoint_sha256"]
            == current["global_surrogate"]["before_checkpoint_sha256"]
        )
        foundation_digest_match = (
            previous["foundation"]["after_digest"] == current["foundation"]["before_digest"]
        )
        continuity.append(
            {
                "transition": f"round {previous['round']} -> {current['round']}",
                "global_surrogate_digest_match": global_digest_match,
                "global_surrogate_checkpoint_sha256_match": global_checkpoint_match,
                "foundation_lora_digest_match": foundation_digest_match,
                "pass": global_digest_match and global_checkpoint_match and foundation_digest_match,
            }
        )

    strict_failure_path = run_root / "p1d_round3_full_20260917_01" / "run_status.json"
    strict_failure = read_json(strict_failure_path)
    overall_pass = (
        all(item["status"] == {"layer1": "pass", "full": "pass"} for item in rounds)
        and all(item["pass"] for item in continuity)
        and all(item["privacy"]["raw_data_non_transmission_audit"] for item in rounds)
    )
    report = {
        "schema": "fedpact.stage1.three_round_summary.v1",
        "status": "pass" if overall_pass else "fail",
        "scope": "Three-round execution and state-lineage verification; not a performance result.",
        "rounds": rounds,
        "continuity": continuity,
        "retained_diagnostic_failure": strict_failure,
        "claim_boundary": {
            "supported": [
                "The two clients started each round from the same redistributed global surrogate state.",
                "Client LoRA and classification-head updates, FedAvg, class-unrestricted retrieval, proxy generation, and foundation LoRA updates executed for three consecutive rounds.",
                "Global-surrogate and foundation-LoRA digests match at both round transitions.",
            ],
            "not_supported": [
                "FedPACT improves predictive performance.",
                "The selected proxies faithfully reproduce every prototype delta.",
                "The method transfers to an unseen or heterogeneous foundation model.",
                "The communication audit constitutes a formal privacy guarantee.",
            ],
        },
    }
    write_json(output_dir / "three_round_state_lineage.json", report)

    markdown = [
        "# FedPACT Stage I：3 round状態継承確認",
        "",
        f"判定：**{'pass' if overall_pass else 'fail'}**（実行・配線・状態継承のみ。性能結果ではない）",
        "",
        "| round | M_G | client LoRA update norm | FedAvg再計算差 | seed / proxy | F | 同じ4 pair上の加重KL |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for item in rounds:
        global_state = item["global_surrogate"]
        foundation = item["foundation"]
        markdown.append(
            "| {round} | {before}→{after} | {norm0:.4f} / {norm1:.4f} | {fedavg:.2e} | "
            "60 / 4 | {f_before}→{f_after} ({policy}) | {loss_before:.6f}→{loss_after:.6f} |".format(
                round=item["round"],
                before=global_state["before_state_id"],
                after=global_state["after_state_id"],
                norm0=global_state["client_lora_update_norms"][0],
                norm1=global_state["client_lora_update_norms"][1],
                fedavg=global_state["fedavg_max_abs_diff_offline_recompute"],
                f_before=foundation["before_state_id"],
                f_after=foundation["after_state_id"],
                policy=foundation["acceptance_policy"],
                loss_before=foundation["weighted_loss_before"],
                loss_after=foundation["weighted_loss_after"],
            )
        )
    markdown.extend(["", "## round間の照合", ""])
    for item in continuity:
        markdown.append(
            f"- {item['transition']}: global digest={item['global_surrogate_digest_match']}, "
            f"checkpoint SHA-256={item['global_surrogate_checkpoint_sha256_match']}, "
            f"foundation digest={item['foundation_lora_digest_match']}"
        )
    markdown.extend(
        [
            "",
            "## 診断上の重要点",
            "",
            "- round 3では、厳しい `nonincrease` モードで試した学習率 10^-6〜10^-11 の全てで同じ4 pair上の加重KLが非増加にならず、失敗runとして保存した。",
            "- 第I段階の必須要件は性能改善ではなく更新実行なので、round 3は `execution` モードで有限・非ゼロのfoundation LoRA更新と対象外パラメータ不変を確認した。KLの増減は診断値として残した。",
            "- seed labelと更新済みglobal surrogateのargmaxは、round 1で60本中43本のみ一致した。seed label不一致には検索挙動だけでなくmodel誤分類も混在する。",
            "- round 1の4 pairではdelta方向が3/4一致し、client-1/class 0は逆向きだった。proxy再現成功とはまだ言えない。",
            "",
            "## 言えること／言えないこと",
            "",
            "3 roundにわたり処理フローと状態継承が接続され、監査可能なログが保存されたことは言える。一方、性能向上、proxy品質、未知データへの転移、形式的なprivacy保証はこの結果からは言えない。",
            "",
        ]
    )
    (output_dir / "three_round_state_lineage.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
