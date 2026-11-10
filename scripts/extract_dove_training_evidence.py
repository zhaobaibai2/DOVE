"""Extract the training takeover ratio from the original TensorBoard event file."""

import argparse
import hashlib
from pathlib import Path

from dove_table01 import DATA, write_csv, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-file", type=Path, required=True)
    args = parser.parse_args()
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto import event_pb2

    rates, lengths, human_seen = {}, {}, []
    for raw in RawEventFileLoader(str(args.event_file)).Load():
        event = event_pb2.Event.FromString(raw)
        for value in event.summary.value:
            if value.tag == "episode/takeover_rate":
                rates[event.step] = float(value.simple_value)
            elif value.tag == "episode/length":
                lengths[event.step] = float(value.simple_value)
            elif value.tag == "buffer/human_seen_total":
                human_seen.append({"step": event.step, "human_seen_total": float(value.simple_value)})
    steps = sorted(rates.keys() & lengths.keys())
    if not steps:
        raise ValueError("No paired episode/takeover_rate and episode/length scalars")
    rows = [{"step": step, "takeover_rate": rates[step], "length": lengths[step],
             "estimated_takeover_steps": rates[step] * lengths[step]} for step in steps]
    output = DATA / "evidence"
    output.mkdir(exist_ok=True)
    path = output / "train_episodes.csv"
    write_csv(path, rows)
    if human_seen:
        write_csv(output / "train_human_seen.csv", human_seen)
    sha = hashlib.sha256()
    with args.event_file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    total = sum(row["length"] for row in rows)
    takeover = sum(row["estimated_takeover_steps"] for row in rows)
    audit = {"source_event_file": str(args.event_file.resolve()), "source_sha256": sha.hexdigest(),
             "extracted_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "tags": ["episode/takeover_rate", "episode/length"],
             "n_episodes": len(rows), "logged_episode_steps": total,
             "estimated_takeover_steps": takeover, "train_takeover_percent": 100 * takeover / total,
             "formula": "100 * sum(takeover_rate_i * length_i) / sum(length_i)",
             "last_human_seen": human_seen[-1] if human_seen else None,
             "human_data_steps_4200": "Not recovered; retain as manuscript-reported metadata",
             "online_env_steps_50000": "Configured budget in args.json; final checkpoint is final_policy_46656.pkl"}
    write_json(output / "training_audit.json", audit)
    print(audit)


if __name__ == "__main__":
    main()
