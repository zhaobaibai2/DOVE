#!/usr/bin/env python3
"""Calculate and audit one simulation figure/table from bundled existing inputs."""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import statistics
from pathlib import Path

from common import HERE, binary, compare, digest, markdown_table, number, read_csv, stats, verify_inputs, write_csv, write_json
from plot_curves import plot_curve, takeover_summary, validate_curve


ITEMS = ("figure_01", "figure_02a", "figure_03a", "figure_03b", "table_01", "table_02", "table_03", "table_04")


def export_artwork(inputs, out, filename):
    import pymupdf
    source = inputs / "paper" / filename
    target = out / filename
    shutil.copyfile(source, target)
    with pymupdf.open(source) as doc:
        page = doc[0]
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(target.with_suffix(".png"))
    if digest(source) != digest(target):
        raise ValueError("Archived artwork copy is not byte-identical")
    return dict(status="ARCHIVED_ARTWORK_EXPORT", operation="copy supplied-paper artwork and render PNG preview",
                raw_reconstruction=False, output_sha256=digest(target))


def curve_vector_check(reference, generated, reward):
    import pymupdf
    import numpy as np
    colors = {"SAC": (0.780392, .352941, .286275), "PPO": (.352941, .152941, .32549),
              "DSAC": (.498039, .741176, .482353), "H-DSAC": (.188235, .188235, .596078),
              "DOVE": (0., .392157, 0.), "PVP": (.890196, .713726, 0.)}
    names = ["SAC", "PPO", "DSAC", "H-DSAC", "DOVE"] if reward else ["H-DSAC", "PVP", "DOVE"]

    def paths(path):
        found = {}
        with pymupdf.open(path) as document:
            for drawing in document[0].get_drawings():
                color = drawing.get("color")
                if drawing["type"] != "s" or color is None or len(drawing["items"]) < 10:
                    continue
                key = tuple(round(float(x), 6) for x in color)
                if all(item[0] == "l" for item in drawing["items"]):
                    found[key] = np.asarray([(float(item[-1].x), float(item[-1].y)) for item in drawing["items"]])
        return found

    a, b = paths(reference), paths(generated)
    rows = []
    for name in names:
        key = colors[name]
        if key not in a or key not in b or a[key].shape != b[key].shape:
            raise ValueError(f"Missing vector curve or different point count: {name}")
        error = float(np.max(np.abs(a[key] - b[key])))
        rows.append(dict(method=name, line_segments=len(a[key]), max_error_pt=error, matches=error <= 1e-4))
    if not all(r["matches"] for r in rows):
        raise ValueError(f"Generated curve geometry differs from supplied PDF: {rows}")
    return rows


def figure_curve(inputs, out, reward):
    import numpy as np
    if reward:
        rows = read_csv(inputs / "curves/reward_reference.csv")
        notes = ["Redraws archived final plot values. The complete original reward preprocessing is not recovered.",
                 "The archived DOVE reward inputs include historical modifications; this is not a raw-return recalculation.",
                 "The shaded band is the supplied Std column, not standard error."]
        status = "FINAL_PLOT_VALUES_MATCH"
        stem = "fig_sim_reward_curve_hdsac_compare"
    else:
        rows, processed = takeover_summary(inputs)
        write_csv(out / "processed_runs.csv", processed)
        expected = read_csv(inputs / "curves/takeover_reference.csv")
        lookup = {(r["Method"], number(r["Step"])): r for r in expected}
        if len(lookup) != len(rows):
            raise ValueError("Takeover aggregate row count differs from reference")
        max_error = 0.
        for row in rows:
            reference = lookup[(row["Method"], row["Step"])]
            for key in ["Mean", "Std"]:
                max_error = max(max_error, abs(number(row[key]) - number(reference[key])))
            if int(row["N"]) != int(reference["N"]):
                raise ValueError("Takeover run count differs")
        if max_error > 1e-12:
            raise ValueError(f"Takeover summary differs by {max_error}")
        write_json(out / "aggregate_check.json", dict(max_abs_error=max_error, tolerance=1e-12, rows=len(rows)))
        notes = ["Six archived run inputs; percent-to-fraction conversion for H-DSAC; original clipping and smoothing preserved.",
                 "Two runs per method, linear interpolation to 0:500:50000; sample SD across runs (ddof=1).",
                 "Interpolation extends endpoint values. These are recovered historical plot inputs, not a new experiment."]
        status = "RECOMPUTED_FROM_ARCHIVED_RUNS_MATCH"
        stem = "fig_sim_takeover_hdsac_pvp_dove"
    validate_curve(rows, reward)
    write_csv(out / "plot_data.csv", rows)
    plot_curve(rows, out / stem, reward)
    checks = curve_vector_check(inputs / "paper" / (stem + ".pdf"), out / (stem + ".pdf"), reward)
    write_csv(out / "vector_check.csv", checks)
    return dict(status=status, notes=notes, vector_curve_checks=checks, raw_experiment_rerun=False)


def plain_tex(value):
    value = re.sub(r"\\cite\{[^}]+\}", "", value)
    value = value.replace("\\DOVE{}", "DOVE").replace("\\pm", "±").replace("\\,", "")
    value = re.sub(r"\\(?:textbf|bm|text|mathrm)\{([^{}]*)\}", r"\1", value)
    return value.replace("$", "").replace("{", "").replace("}", "").replace("~", " ").strip()


def main_reference(inputs):
    rows = []
    for line in (inputs / "paper/table_02.tex").read_text().splitlines():
        cells = line.split("&")
        if len(cells) != 7 or cells[0].strip() == "Method":
            continue
        texts = [plain_tex(c) for c in cells]
        row = dict(method=texts[0])
        for key, text in zip(["human_data", "online_steps", "train_cost", "reward", "cost", "success"], texts[1:]):
            matches = re.findall(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)([KM]?)", text)
            vals = [float(v.replace(",", "")) * {"": 1, "K": 1000, "M": 1000000}[unit] for v, unit in matches]
            row[key] = vals[0] if vals else None
            if key in ["train_cost", "reward", "cost", "success"]:
                row[key + "_reported_dispersion"] = vals[1] if "±" in text and len(vals) >= 2 else None
        rows.append(row)
    if len(rows) != 13:
        raise ValueError(f"Expected 13 main-table rows in the supplied paper, got {len(rows)}")
    return rows


def table01(inputs, out):
    source = inputs / "paper/table_01.tex"
    shutil.copyfile(source, out / "table.tex")
    rows = []
    for text in re.split(r"\\\\", source.read_text()):
        cells = text.split("&")
        if len(cells) == 2:
            rows.append(dict(setting_tex=cells[0].strip(), value_tex=cells[1].strip(), evidence="paper_configuration"))
    write_csv(out / "configuration.csv", rows)
    markdown_table(out / "configuration.md", rows)
    return dict(status="CONFIGURATION_EXPORT", notes=["Table I contains prescribed hyperparameters, not measured results.",
                "Exports settings from the supplied paper; does not assert that historical runs used them."])


def table02(inputs, out):
    references = main_reference(inputs)
    write_csv(out / "paper_reference.csv", references)
    shutil.copyfile(inputs / "paper/table_02.tex", out / "paper_reference.tex")
    before = read_csv(inputs / "table_02/original_pre_transform.csv")
    after = read_csv(inputs / "table_02/paper_input_375_scaled.csv")
    if len(before) != 100 or len(after) != 100:
        raise ValueError("Recovered main-table DOVE source must have 100 episodes")
    scale = 375. / max(number(r["reward"]) for r in before)
    transforms = []
    for i, (a, b) in enumerate(zip(before, after)):
        if a["episode_idx"] != b["episode_idx"] or number(a["cost"]) != number(b["cost"]) or binary(a["success"]) != binary(b["success"]):
            raise ValueError("Original and transformed episodes do not align")
        error = number(b["reward"]) - number(a["reward"]) * scale
        transforms.append(dict(row=i, original_reward=number(a["reward"]), scale=scale,
                               archived_plot_reward=number(b["reward"]), actual_minus_scaled=error))
    if max(abs(r["actual_minus_scaled"]) for r in transforms) > 1e-10:
        raise ValueError("Recovered reward transformation does not reproduce the archived input")
    write_csv(out / "reward_transform_check.csv", transforms)
    summaries = []
    for name, rows in [("original_pre_transform", before), ("archived_375_scaled", after)]:
        for metric in ["reward", "cost", "success"]:
            summaries.append(dict(source=name, metric=metric, **stats(r[metric] for r in rows)))
    write_csv(out / "episode_statistics.csv", summaries)
    by = {(r["source"], r["metric"]): r for r in summaries}
    dove = next(r for r in references if r["method"] == "DOVE")
    checks = []
    for metric in ["reward", "cost", "success"]:
        result = by[("archived_375_scaled", metric)]
        checks.append(compare("DOVE", metric, result["mean"], dove[metric], 2, "single checkpoint / 100 episodes"))
        rule = "sd" if metric == "reward" else "se"
        checks.append(compare("DOVE", metric + "_reported_dispersion", result[rule], dove[metric + "_reported_dispersion"], 2,
                              f"historical episode {rule}; not independent-run SE"))
    write_csv(out / "comparison.csv", checks)
    coverage = [dict(method=r["method"], evidence="recovered single-checkpoint episodes" if r["method"] == "DOVE" else "manuscript reference only",
                     independent_training_runs_recovered=0) for r in references]
    write_csv(out / "evidence_coverage.csv", coverage)
    notes = ["All 13 reference rows are extracted from the supplied paper, separately from measurements.",
             "The recovered DOVE source is one checkpoint with 100 episodes; the paper specifies 3 independent runs x 20 episodes.",
             "The historical reward input was scaled by 375/max(original reward). Both original and scaled statistics are retained.",
             "The archived matching dispersion uses episode SD for reward and episode SE for cost/success; these do not verify the paper's independent-run SE.",
             "Baseline raw run records, DOVE training human-step count, and training safety cost are not recovered here."]
    return dict(status="PARTIAL_EVIDENCE", checks=checks, notes=notes)


def table04(inputs, out):
    refs = read_csv(inputs / "paper/table_04_reference.csv")
    summaries, checkpoints, checks = [], [], []
    observed_maps = set()
    for ref in refs:
        mode = ref["mode"]
        files = sorted((inputs / "table_04" / mode).glob("*.csv"))
        if len(files) != 5:
            raise ValueError(f"{mode}: expected five checkpoints")
        all_rows = []
        expected_keys = None
        for path in files:
            rows = read_csv(path)
            keys = [(int(r["map_seed"]), int(r["episode_idx"])) for r in rows]
            observed_maps.update(k[0] for k in keys)
            if len(rows) != 100 or len(set(keys)) != 100:
                raise ValueError(f"{path}: expected 100 unique episode IDs")
            if expected_keys is None:
                expected_keys = set(keys)
            if set(keys) != expected_keys:
                raise ValueError("Inference checkpoints do not share the same map/episode grid")
            for row in rows:
                binary(row["success"])
                binary(row["crash"])
                if not 0 <= number(row["route_completion"]) <= 1.:
                    raise ValueError("Route completion outside [0,1]")
            all_rows.extend(rows)
            checkpoints.append(dict(mode=mode, checkpoint=path.stem, episodes=len(rows),
                                    **{k: stats(r[k] for r in rows)["mean"] for k in ["success", "crash", "route_completion", "reward"]}))
        summary = dict(mode=mode, label=ref["label"], checkpoints=len(files), episodes=len(all_rows))
        for metric in ["success", "crash", "route_completion", "reward"]:
            value = stats(r[metric] for r in all_rows)["mean"]
            summary[metric] = value
            checks.append(compare(mode, metric, value, ref[metric], 2 if metric == "reward" else 3))
        summaries.append(summary)
    if any(c["status"] != "MATCH" for c in checks):
        status = "RESULT_MISMATCH"
    else:
        status = "NUMBERS_MATCH_PROTOCOL_GAP" if len(observed_maps) != 10 else "RECOMPUTED_MATCH"
    write_csv(out / "checkpoint_statistics.csv", checkpoints)
    write_csv(out / "recomputed.csv", summaries)
    write_csv(out / "comparison.csv", checks)
    markdown_table(out / "table.md", summaries)
    lines = ["\\begin{tabular}{lrrrr}", "Inference mode & Success & Collision & Route & Reward \\\\", "\\hline"]
    lines += [f"{r['label']} & {r['success']:.3f} & {r['crash']:.3f} & {r['route_completion']:.3f} & {r['reward']:.2f} \\\\" for r in summaries]
    (out / "table.tex").write_text("\n".join(lines + ["\\end{tabular}"]) + "\n")
    config = json.loads((inputs / "table_04/training_args.json").read_text())
    train_maps = list(range(int(config["start_seed"]), int(config["start_seed"]) + int(config["num_scenarios"])))
    overlap = sorted(observed_maps.intersection(train_maps))
    if overlap:
        status = "NUMBERS_MATCH_PROTOCOL_GAP" if all(c["status"] == "MATCH" for c in checks) else "RESULT_MISMATCH"
    return dict(status=status, numeric_status="RECOMPUTED_MATCH" if all(c["status"] == "MATCH" for c in checks) else "RESULT_MISMATCH",
                observed_map_seeds=sorted(observed_maps), recorded_training_map_seeds=train_maps,
                train_eval_overlap=overlap, checks=checks, notes=["Seven modes; five checkpoints per mode; 100 episodes per checkpoint.",
                "Pool the 500 episodes per mode. This is not an independent-training-run estimate.",
                f"The recovered episode rows use only map seeds {sorted(observed_maps)}; this does not establish a ten-map held-out evaluation.",
                f"The saved training args specify map seeds {train_maps}; train/evaluation overlap is {overlap}.",
                "These files reproduce the reported numbers but do not validate the stated disjoint held-out protocol."])


def normalize_source(source):
    # Historical records contain both workstation-absolute and repository-relative paths.
    token = "/test0508/"
    return source.split(token, 1)[-1] if token in source else source


def table03(inputs, out, actor_counts_dir=None):
    refs = read_csv(inputs / "paper/table_03_reference.csv")
    frozen = read_csv(inputs / "table_03/pair_manifest.csv")
    pair_ids = [(r["source"], int(r["index"])) for r in frozen]
    if len(pair_ids) != 256 or len(set(pair_ids)) != 256:
        raise ValueError("Expected 256 unique frozen intervention pairs")
    pair_id_set = set(pair_ids)
    checkpoint_hashes = {(r["group"], r["checkpoint"]): r["checkpoint_sha256"]
                         for r in read_csv(inputs / "table_03/frozen_checkpoint_hashes.csv")}
    actor_counts_dir = actor_counts_dir or inputs / "table_03/resampled_actor"
    summaries, details, checks, legacy_rows = [], [], [], []
    for ref in refs:
        group = ref["group"]
        expected_n = 50 if group in ("positive_only", "er_without_cpcal") else 46
        value = dict(group=group, checkpoints=expected_n, pairs_per_checkpoint=256, rho_vlb=None,
                     hsr=None, rmr_0p5=None, pairacc_c_mean=None)
        if group == "cpcal_only":
            files = sorted((inputs / "table_03/cpcal_only").glob("*_pairs.csv"))
            if len(files) != expected_n:
                raise ValueError("CPCal-only checkpoint count differs from protocol")
            pooled = []
            for path in files:
                rows = read_csv(path)
                if [(r["source"], int(r["index"])) for r in rows] != pair_ids:
                    raise ValueError(f"Frozen pair IDs differ: {path}")
                for r in rows:
                    binary(r["rho_vlb_indicator"])
                    binary(r["pairacc_c_indicator"])
                    for field in ["hsr_count_of_64", "rmr_0p5_count_of_64"]:
                        if not 0 <= int(r[field]) <= 64 or number(r[field]) != int(r[field]):
                            raise ValueError("Invalid actor sample count")
                pooled.extend(rows)
            value.update(rho_vlb=statistics.fmean(binary(r["rho_vlb_indicator"]) for r in pooled),
                         hsr=sum(int(r["hsr_count_of_64"]) for r in pooled)/(len(pooled)*64),
                         rmr_0p5=sum(int(r["rmr_0p5_count_of_64"]) for r in pooled)/(len(pooled)*64),
                         pairacc_c_mean=statistics.fmean(binary(r["pairacc_c_indicator"]) for r in pooled))
        else:
            files = sorted((inputs / "table_03/vlb" / group).glob("*_mse_pairs.csv"))
            if len(files) != expected_n:
                raise ValueError(f"{group}: VLB checkpoint count differs")
            vlb, policies = [], set()
            for path in files:
                rows = read_csv(path)
                if [(r["source"], int(r["index"])) for r in rows] != pair_ids:
                    raise ValueError(f"VLB pair IDs differ: {path}")
                policies.add(path.name.removesuffix("_mse_pairs.csv"))
                for r in rows:
                    indicator = int(number(r["delta_vlb"]) > 0.)
                    if indicator != binary(r["vlb_ok"]):
                        raise ValueError("VLB indicator does not match strict positive gap")
                    vlb.append(indicator)
                details.append(dict(group=group, checkpoint=path.stem, pairs=len(rows), rho_vlb=statistics.fmean(binary(r["vlb_ok"]) for r in rows)))
            value["rho_vlb"] = statistics.fmean(vlb)
            critic = read_csv(inputs / "table_03/critic_legacy" / (group + ".csv"))
            selected, seen = [], set()
            for r in critic:
                # Archive names are sometimes a relative policy path, sometimes a file stem.
                name = Path(r["policy_name"]).stem
                accuracy = int(number(r["qbar_neg"]) > number(r["qbar_pos"]))
                if accuracy != binary(r["pairacc_c"]):
                    raise ValueError("Stored critic ordering differs from the mean critic scores")
                key = (normalize_source(r["source"]), int(r["source_index"]))
                if name in policies and key in pair_id_set:
                    identity = (name, key)
                    if identity in seen:
                        raise ValueError("Duplicate critic pair/checkpoint observation")
                    seen.add(identity)
                    selected.append(accuracy)
            if len(selected) != expected_n * 256:
                raise ValueError(f"{group}: critic protocol subset has {len(selected)} rows, expected {expected_n*256}")
            value["pairacc_c_mean"] = statistics.fmean(selected)
            sample_dir = actor_counts_dir / group
            if sample_dir.exists():
                metadata = json.loads((sample_dir / "metadata.json").read_text())
                if not metadata["full_protocol"] or metadata["samples_per_pair"] != 64 or metadata["checkpoint_count"] != expected_n:
                    raise ValueError(f"Actor sampling metadata does not satisfy the paper protocol: {group}")
                samples = sorted(sample_dir.glob("*_samples.csv"))
                if {p.name.removesuffix("_samples.csv") for p in samples} != policies:
                    raise ValueError(f"Actor sampling checkpoint set differs from the VLB set: {group}")
                hsr_count = rmr_count = observation_count = 0
                for path in samples:
                    rows = read_csv(path)
                    stem = path.name.removesuffix("_samples.csv")
                    if [(r["source"], int(r["index"])) for r in rows] != pair_ids:
                        raise ValueError("Regenerated actor counts use different intervention pairs")
                    for row in rows:
                        if row["checkpoint_sha256"] != checkpoint_hashes[(group, stem)]:
                            raise ValueError("Regenerated actor counts use a different model checkpoint")
                        for field in ["hsr_count_of_64", "rmr_0p5_count_of_64"]:
                            if number(row[field]) != int(row[field]) or not 0 <= int(row[field]) <= 64:
                                raise ValueError("Invalid regenerated actor count")
                        hsr_count += int(row["hsr_count_of_64"])
                        rmr_count += int(row["rmr_0p5_count_of_64"])
                        observation_count += 64
                value["hsr"] = hsr_count / observation_count
                value["rmr_0p5"] = rmr_count / observation_count
            legacy_rows.append(dict(group=group, legacy_checkpoints=len(set(r["policy_name"] for r in critic)),
                                    legacy_pair_checkpoint_rows=len(critic), legacy_pairs_per_checkpoint=512,
                                    legacy_pairacc_c_mean=statistics.fmean(binary(r["pairacc_c"]) for r in critic),
                                    paper_protocol_pairs_per_checkpoint=256, paper_protocol_checkpoints=expected_n))
        summaries.append(value)
        for metric in ["rho_vlb", "hsr", "rmr_0p5", "pairacc_c_mean"]:
            checks.append(compare(group, metric, value[metric], ref[metric], 3,
                                  "same 256 frozen pairs; expected checkpoint set"))
    write_csv(out / "recomputed_paper_protocol.csv", summaries)
    write_csv(out / "comparison.csv", checks)
    write_csv(out / "legacy_critic_denominator_audit.csv", legacy_rows)
    write_csv(out / "vlb_checkpoint_statistics.csv", details)
    markdown_table(out / "table.md", summaries)
    return dict(status="PARTIAL_EVIDENCE" if any(r["status"] == "MISSING" for r in checks) else (
                "RECOMPUTED_MATCH" if all(r["status"] == "MATCH" for r in checks) else "RESULT_MISMATCH"), checks=checks,
                notes=["Use the supplied paper's 256 pair IDs and 50/50/46/46 saved-checkpoint counts.",
                       "HSR/RMR counts for the three older groups were regenerated by frozen-model inference on 2026-10-30 using 64 samples for each of the same 256 pair/checkpoint observations. No model was trained or changed.",
                       "The legacy critic archive uses 512 pairs and 52/52/48 checkpoints. Its aggregates match several reported cells but violate the current paper's stated denominator.",
                       "The recomputed critic column uses the actual 256-pair/50-or-46-checkpoint subset; legacy aggregates are retained separately.",
                       "Available VLB ordering is recalculated from delta_vlb>0. A discrepancy is reported rather than rounding or selecting checkpoints to force a match."])


def run(item, output, actor_counts_dir=None):
    inputs = HERE / "inputs"
    out = output / item
    out.mkdir(parents=True, exist_ok=True)
    if item == "figure_01":
        report = export_artwork(inputs, out, "framework.pdf")
    elif item == "figure_02a":
        report = export_artwork(inputs, out, "fig_metadrive_hitl_setup.pdf")
    elif item.startswith("figure_03"):
        report = figure_curve(inputs, out, item.endswith("a"))
    elif item == "table_03":
        report = table03(inputs, out, actor_counts_dir)
    else:
        report = {"table_01": table01, "table_02": table02, "table_03": table03, "table_04": table04}[item](inputs, out)
    report.update(item=item, source_manifest_sha256=digest(inputs / "sources.csv"),
                  paper=json.loads((inputs / "paper_identity.json").read_text()), raw_experiment_rerun=False)
    write_json(out / "report.json", report)
    checks = report.get("checks", [])
    counts = dict(collections.Counter(c["status"] for c in checks))
    print(f"{item}: {report['status']} checks={counts} output={out}", flush=True)
    return report


def main():
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("item", choices=[*ITEMS, "all", "figure_03"])
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("DOVE_OUTPUT_DIR", HERE / "outputs")))
    parser.add_argument("--strict", action="store_true", help="exit 2 after writing reports if any table has missing/different evidence")
    parser.add_argument("--actor-counts-dir", type=Path, help="optional replacement HSR/RMR audit directory emitted by resample_actor.py")
    args = parser.parse_args()
    count = verify_inputs()
    print(f"Verified {count} bundled input hashes", flush=True)
    output = args.output_dir.expanduser().resolve()
    selected = ITEMS if args.item == "all" else (("figure_03a", "figure_03b") if args.item == "figure_03" else (args.item,))
    reports = []
    for item in selected:
        reports.append(run(item, output, args.actor_counts_dir.expanduser().resolve() if args.actor_counts_dir else None))
    partial = any(r["status"] in ["PARTIAL_EVIDENCE", "RESULT_MISMATCH", "NUMBERS_MATCH_PROTOCOL_GAP"] for r in reports)
    summary = dict(status="COMPLETED_WITH_EVIDENCE_GAPS" if partial else "COMPLETED", items=[dict(item=r["item"], status=r["status"]) for r in reports])
    write_json(output / (args.item + "_summary.json"), summary)
    if partial:
        print("Evidence gaps/differences are present; see each report.json and comparison.csv. This run is not a claim that all paper results are verified.")
    return 2 if args.strict and partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
