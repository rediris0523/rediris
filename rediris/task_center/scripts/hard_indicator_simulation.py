import argparse
import copy
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
import bittensor as bt


@dataclass(frozen=True)
class NodeWallet:
    wallet_name: str
    hotkey_name: str
    hotkey: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_wallet_hotkey(wallet_name: str, hotkey_name: str) -> str:
    # Note: assumes wallet key files already exist locally under ~/.bittensor/wallets/.
    w = bt.wallet(name=wallet_name, hotkey=hotkey_name)
    return w.hotkey.ss58_address


def metagraph_weights_snapshot(
    subtensor: bt.subtensor,
    netuid: int,
    validator_hotkeys: List[str],
    miner_hotkeys: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int], Dict[str, int]]:
    metagraph = subtensor.metagraph(netuid=netuid)

    validator_uids: Dict[str, int] = {}
    miner_uids: Dict[str, int] = {}

    for vk in validator_hotkeys:
        try:
            validator_uids[vk] = metagraph.hotkeys.index(vk)
        except ValueError:
            raise RuntimeError(f"Validator hotkey not found in metagraph: {vk}")

    for mk in miner_hotkeys:
        try:
            miner_uids[mk] = metagraph.hotkeys.index(mk)
        except ValueError:
            raise RuntimeError(f"Miner hotkey not found in metagraph: {mk}")

    # metagraph.W shape is typically [validator_uid, miner_uid] for a given subnet.
    # We snapshot only the weights for our validators/miners.
    snapshot: Dict[str, Dict[str, float]] = {}
    for vk, v_uid in validator_uids.items():
        row = metagraph.W[v_uid]
        snapshot[vk] = {mk: float(row[m_uid]) for mk, m_uid in miner_uids.items()}

    return snapshot, validator_uids, miner_uids


def http_json(client: httpx.Client, method: str, url: str, *, json_body: Optional[Dict] = None) -> Any:
    resp = client.request(method, url, json=json_body, timeout=90.0)
    resp.raise_for_status()
    if not resp.content:
        return None
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local/devnet-only hard indicator simulation; runtime HOLD unless explicitly authorized.")
    parser.add_argument("--website-admin-url", default="http://localhost:8000", help="website_admin base URL for an explicitly authorized local/devnet run")
    parser.add_argument("--netuid", type=int, required=True, help="Bittensor netuid for an explicitly authorized local/devnet run")
    parser.add_argument("--network", default="local_or_devnet_placeholder", help="bittensor network name")
    parser.add_argument("--hotkey-name", default="default", help="hotkey name shared by all nodes")

    parser.add_argument("--miner-wallet-names", nargs="+", required=True, help="miner wallet names for an explicitly authorized local/devnet run")
    parser.add_argument("--validator-wallet-names", nargs="+", required=True, help="validator wallet names for an explicitly authorized local/devnet run")

    parser.add_argument("--template-name", default="REPLACE_WITH_AUTHORIZED_TEMPLATE", help="website_admin task template name")
    parser.add_argument("--dataset-repo", default="REPLACE_WITH_AUTHORIZED_LOCAL_OR_PUBLIC_DATASET", help="dataset repo id for dataset_spec.repository_id")
    parser.add_argument("--sample-count", type=int, default=50, help="dataset_spec.sample_count for fast simulation")

    # Training overrides (speed > quality for the purpose of producing LoRAs and triggering weights).
    parser.add_argument("--iteration-count", type=int, default=20, help="training_spec.iteration_count")
    parser.add_argument("--batch-size", type=int, default=1, help="training_spec.batch_size")
    parser.add_argument("--lora-rank", type=int, default=8, help="training_spec.lora_rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="training_spec.lora_alpha")

    # Task timeline overrides (durations are in hours).
    parser.add_argument("--announcement-hours", type=float, default=0.001, help="announcement_duration in hours")
    parser.add_argument("--dataset-validation-hours", type=float, default=0.001, help="dataset_validation_duration in hours")
    parser.add_argument("--execution-hours", type=float, default=0.005, help="execution_duration in hours")
    parser.add_argument("--review-hours", type=float, default=0.005, help="review_duration in hours")
    parser.add_argument("--reward-hours", type=float, default=0.02, help="reward_duration in hours")
    parser.add_argument("--default-reward-miners", type=int, default=10, help="task default_reward_miners")

    parser.add_argument("--poll-interval-sec", type=int, default=15, help="polling interval for task/miners")
    parser.add_argument("--max-wait-seconds", type=int, default=7200, help="max wait time for local/devnet evidence")
    parser.add_argument("--weight-delta-threshold", type=float, default=1e-6, help="weight change threshold")
    parser.add_argument("--evidence-out", default="", help="output json path (default: auto-named in cwd)")

    args = parser.parse_args()

    website_admin_base = args.website_admin_url.rstrip("/")
    task_center_miners_public_path_tpl = f"/v1/tasks/public/{{task_id}}/miners"

    miners: List[NodeWallet] = []
    validators: List[NodeWallet] = []
    for mn in args.miner_wallet_names:
        hot = load_wallet_hotkey(mn, args.hotkey_name)
        miners.append(NodeWallet(wallet_name=mn, hotkey_name=args.hotkey_name, hotkey=hot))
    for vn in args.validator_wallet_names:
        hot = load_wallet_hotkey(vn, args.hotkey_name)
        validators.append(NodeWallet(wallet_name=vn, hotkey_name=args.hotkey_name, hotkey=hot))

    miner_hotkeys = [m.hotkey for m in miners]
    validator_hotkeys = [v.hotkey for v in validators]

    # In this repo, `bittensor.network` is used as the subnet selector.
    # We keep this script lightweight and match the project's bittensor usage patterns.
    subtensor = bt.subtensor(network=args.network)

    before_start = utc_now_iso()
    weights_before, validator_uids, miner_uids = metagraph_weights_snapshot(
        subtensor=subtensor,
        netuid=args.netuid,
        validator_hotkeys=validator_hotkeys,
        miner_hotkeys=miner_hotkeys,
    )

    with httpx.Client() as client:
        templates_url = f"{website_admin_base}/v1/task-templates"
        # list endpoint uses query params
        resp = client.get(templates_url, params={"workflow_type": "text_lora_creation", "is_active": True, "page": 1, "page_size": 200}, timeout=90.0)
        resp.raise_for_status()
        templates_body = resp.json()
        templates = templates_body.get("templates", [])

        selected_template = None
        for t in templates:
            if t.get("name") == args.template_name:
                selected_template = t
                break
        if not selected_template:
            raise RuntimeError(f"Template not found: {args.template_name}. Available: {[t.get('name') for t in templates]}")

        workflow_spec = copy.deepcopy(selected_template["workflow_spec"])
        # Override dataset_spec for bookcorpus
        dataset_spec = workflow_spec.get("dataset_spec", {})
        dataset_spec["repository_id"] = args.dataset_repo
        dataset_spec["sample_count"] = args.sample_count
        # Avoid strict QA validation when bookcorpus doesn't have question/answer fields.
        dataset_spec["question_column"] = None
        dataset_spec["answer_column"] = None
        dataset_spec["source"] = dataset_spec.get("source", "huggingface")
        workflow_spec["dataset_spec"] = dataset_spec

        training_spec = workflow_spec.get("training_spec", {})
        training_spec["iteration_count"] = args.iteration_count
        training_spec["batch_size"] = args.batch_size
        training_spec["lora_rank"] = args.lora_rank
        training_spec["lora_alpha"] = args.lora_alpha
        workflow_spec["training_spec"] = training_spec

        task_id = f"hard-indicator-{int(time.time())}"
        publish_url = f"{website_admin_base}/v1/tasks/publish"
        publish_req = {
            "task_id": task_id,
            "workflow_type": "text_lora_creation",
            "workflow_spec": workflow_spec,
            "publish_status": "published",
            "announcement_duration": args.announcement_hours,
            "dataset_validation_duration": args.dataset_validation_hours,
            "execution_duration": args.execution_hours,
            "review_duration": args.review_hours,
            "reward_duration": args.reward_hours,
            "default_reward_miners": args.default_reward_miners,
        }

        publish_resp = client.post(publish_url, json=publish_req, timeout=120.0)
        publish_resp.raise_for_status()
        publish_body = publish_resp.json()

        # Wait until miners have submitted LoRAs (model_url will be filled by real miners).
        submitted_by_miner: Dict[str, Dict[str, Any]] = {}
        poll_start = time.time()
        deadline = poll_start + args.max_wait_seconds
        while time.time() < deadline:
            task_miners_url = f"{website_admin_base}{task_center_miners_public_path_tpl.format(task_id=task_id)}"
            body = http_json(client, "GET", task_miners_url)
            miners_entries = body.get("miners", []) or []

            submitted_by_miner = {
                e.get("hotkey"): e.get("submission")
                for e in miners_entries
                if e.get("submission") is not None
            }

            submitted_hotkeys = [mk for mk in miner_hotkeys if mk in submitted_by_miner]
            if len(submitted_hotkeys) >= len(miner_hotkeys):
                break

            time.sleep(args.poll_interval_sec)

        if len(submitted_by_miner) < len(miner_hotkeys):
            raise RuntimeError(
                f"Timeout waiting miner submissions. got={len(submitted_by_miner)} expected={len(miner_hotkeys)}"
            )

        after_submit_time = utc_now_iso()

        # Wait for local/devnet evidence.
        weights_after: Optional[Dict[str, Dict[str, float]]] = None
        changed_pairs: List[Dict[str, Any]] = []
        while time.time() < deadline:
            try:
                weights_after_snapshot, _, _ = metagraph_weights_snapshot(
                    subtensor=subtensor,
                    netuid=args.netuid,
                    validator_hotkeys=validator_hotkeys,
                    miner_hotkeys=miner_hotkeys,
                )
            except Exception:
                time.sleep(20)
                continue

            changed_pairs.clear()
            for vk in validator_hotkeys:
                for mk in miner_hotkeys:
                    b = weights_before[vk].get(mk, 0.0)
                    a = weights_after_snapshot[vk].get(mk, 0.0)
                    if abs(a - b) > args.weight_delta_threshold:
                        changed_pairs.append({"validator_hotkey": vk, "miner_hotkey": mk, "before": b, "after": a})

            if changed_pairs:
                weights_after = weights_after_snapshot
                break

            time.sleep(20)

        if weights_after is None:
            raise RuntimeError("Timeout waiting for validator local/devnet evidence from chain metagraph.")

    evidence = {
        "task_id": task_id,
        "published_at": publish_body.get("announcement_start") or publish_body.get("message") or "",
        "dataset_repo": args.dataset_repo,
        "dataset_sample_count": args.sample_count,
        "training_overrides": {
            "iteration_count": args.iteration_count,
            "batch_size": args.batch_size,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
        },
        "miners": [{"wallet_name": m.wallet_name, "hotkey": m.hotkey} for m in miners],
        "validators": [{"wallet_name": v.wallet_name, "hotkey": v.hotkey} for v in validators],
        "metagraph": {"netuid": args.netuid, "validator_uids": validator_uids, "miner_uids": miner_uids},
        "weights_before": weights_before,
        "weights_after": weights_after,
        "weight_changed_pairs": changed_pairs,
        "timestamps": {
            "before_snapshot": before_start,
            "after_submit": after_submit_time,
            "generated_at": utc_now_iso(),
        },
        "miner_submissions": submitted_by_miner,  # hotkey -> submission obj (contains model_url/status/timestamps)
    }

    out_path = args.evidence_out.strip()
    if not out_path:
        out_path = f"hard-indicator-evidence-{task_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)

    print(f"[hard-indicator] task_id={task_id}")
    print(f"[hard-indicator] weight_changed_pairs={len(changed_pairs)}")
    print(f"[hard-indicator] evidence saved: {out_path}")


if __name__ == "__main__":
    main()
