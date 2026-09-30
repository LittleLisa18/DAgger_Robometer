"""Metrics and reports for paired, offline success evaluation (NumPy only)."""

import csv
import json
from pathlib import Path

import numpy as np

from .core import atomic_json
from .evaluation_common import EvaluationStore, score_protocol


def metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid metric inputs")
    predicted = p > threshold
    tp, fp = int((predicted & y).sum()), int((predicted & ~y).sum())
    fn, tn = int((~predicted & y).sum()), int((~predicted & ~y).sum())

    def divide(a, b):
        return a / b if b else None

    precision, recall, specificity = (
        divide(tp, tp + fp),
        divide(tp, tp + fn),
        divide(tn, tn + fp),
    )
    result = {
        "n": len(y),
        "positive": int(y.sum()),
        "negative": int((~y).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "roc_auc": None,
        "pr_auc": None,
        "average_precision": None,
    }
    if y.any() and (~y).any():
        # Group ties at a single threshold: arbitrary tie ordering must not affect AUC.
        order = np.argsort(-p, kind="stable")
        ys, ps = y[order], p[order]
        ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
        tp_curve = np.cumsum(ys)[ends]
        fp_curve = 1 + ends - tp_curve
        recalls = np.r_[0.0, tp_curve / y.sum()]
        fprs = np.r_[0.0, fp_curve / (~y).sum()]
        precisions = np.r_[1.0, tp_curve / (tp_curve + fp_curve)]
        result["roc_auc"] = float(np.trapz(recalls, fprs))
        result["pr_auc"] = float(np.trapz(precisions, recalls))
        result["average_precision"] = float(np.sum(np.diff(recalls) * precisions[1:]))
    return result


def grouped(rows, probability, config):
    def calculate(selected):
        return metrics(
            [r["env_success"] for r in selected],
            [r[probability] for r in selected],
            config.success_threshold,
        )

    tasks = {
        str(t): calculate([r for r in rows if r["task_id"] == t])
        for t in config.task_ids
    }
    macro = {}
    for key in (
        "precision",
        "recall",
        "specificity",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "average_precision",
    ):
        values = [t[key] for t in tasks.values() if t[key] is not None]
        macro[key] = {
            "value": float(np.mean(values)) if values else None,
            "valid_tasks": len(values),
        }
    return {"overall": calculate(rows), "per_task": tasks, "macro": macro}


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(config):
    root = Path(config.output)
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    episodes = list(EvaluationStore(root).episodes(verify=True))
    rows, errors, predictions = [], [], {}
    indexes = {}
    for model in config.score_models:
        path = root / "scores" / model / "index.json"
        if path.exists():
            index = json.loads(path.read_text())
            if index["protocol"] != score_protocol(config):
                raise ValueError("Report scoring protocol mismatch")
            indexes[model] = index
    for directory, meta, trajectory_id in episodes:
        row = {
            k: meta[k]
            for k in (
                "episode_id",
                "task_id",
                "task",
                "initial_state_id",
                "seed",
                "steps",
                "env_success",
                "end_reason",
                "test_only",
            )
        }
        for model in config.score_models:
            row[model + "_probability"] = None
            row[model + "_complete"] = False
            entry = indexes.get(model, {}).get("episodes", {}).get(meta["episode_id"])
            if entry is None:
                errors.append(
                    {
                        "episode_id": meta["episode_id"],
                        "model": model,
                        "error": "No indexed scoring result",
                    }
                )
                continue
            if entry["trajectory_identity"] != trajectory_id:
                raise ValueError("Report trajectory identity mismatch")
            result = json.loads((root / "scores" / model / entry["file"]).read_text())
            if (
                result["identity"]["model"] != indexes[model]["model_identity"]
                or result["identity"]["trajectory"] != trajectory_id
                or result["identity"]["protocol"] != score_protocol(config)
            ):
                raise ValueError("Score file identity mismatch")
            predictions[(model, meta["episode_id"])] = result["predictions"]
            final = result["predictions"].get(str(meta["steps"]))
            if final:
                row[model + "_probability"] = final["success_probability"]
            row[model + "_complete"] = result["complete"]
            errors.extend(
                dict(e, episode_id=meta["episode_id"], model=model)
                for e in result["errors"]
            )
            if not result["complete"]:
                errors.append(
                    {
                        "episode_id": meta["episode_id"],
                        "model": model,
                        "error": "Missing final or diagnostic score",
                    }
                )
        rows.append(row)
    paired = [
        r
        for r in rows
        if all(r[m + "_probability"] is not None for m in config.score_models)
    ]
    for row in paired:
        hashes = [
            predictions[(m, row["episode_id"])][str(row["steps"])]["input_sha256"]
            for m in config.score_models
        ]
        if len(set(hashes)) != 1:
            raise ValueError("Paired models did not receive identical final inputs")
        # All common diagnostic endpoints must also use identical sampled inputs.
        common = set.intersection(
            *(set(predictions[(m, row["episode_id"])]) for m in config.score_models)
        )
        for step in common:
            if (
                len(
                    {
                        predictions[(m, row["episode_id"])][step]["input_sha256"]
                        for m in config.score_models
                    }
                )
                != 1
            ):
                raise ValueError("Paired diagnostic input mismatch")
    expected = len(config.task_ids) * config.episodes
    completed_diagnostics = {
        m: sum(r[m + "_complete"] for r in rows) for m in config.score_models
    }
    summary = {
        "expected": expected,
        "completed_rollouts": len(rows),
        "paired_final": len(paired),
        "complete_diagnostics": completed_diagnostics,
        "complete": len(rows) == expected
        and all(n == expected for n in completed_diagnostics.values()),
        "test_only": config.test_only,
        "auc_definition": "ROC and PR use trapezoidal integration at grouped distinct thresholds; average_precision is reported separately. AUC is undefined without both classes.",
        "models": {},
        "paired": {},
        "diagnostics": {},
    }
    for model in config.score_models:
        key = model + "_probability"
        valid = [r for r in rows if r[key] is not None]
        summary["models"][model] = grouped(valid, key, config)
        summary["paired"][model] = grouped(paired, key, config)
        diagnostic_rows = [r for r in rows if r[model + "_complete"]]
        rules = {"last": [], "mean": [], "median": [], "all_pass": [], "any_pass": []}
        jumps, crosses, transitions = [], 0, 0
        for row in diagnostic_rows:
            values = predictions[(model, row["episode_id"])]
            probs = np.array(
                [values[k]["success_probability"] for k in sorted(values, key=int)]
            )
            jumps.extend(abs(np.diff(probs)).tolist())
            crosses += int(np.count_nonzero(np.diff(probs > config.success_threshold)))
            transitions += max(0, len(probs) - 1)
            for name, value in {
                "last": probs[-1],
                "mean": np.mean(probs),
                "median": np.median(probs),
                "all_pass": float(np.all(probs > config.success_threshold)),
                "any_pass": float(np.any(probs > config.success_threshold)),
            }.items():
                rules[name].append(float(value))
        labels = [r["env_success"] for r in diagnostic_rows]
        summary["diagnostics"][model] = {
            "episodes": len(labels),
            "adjacent_transitions": transitions,
            "threshold_crossings": crosses,
            "absolute_jump_ge_0.5": sum(j >= 0.5 for j in jumps),
            "rules": {
                name: metrics(labels, scores, config.success_threshold)
                for name, scores in rules.items()
            },
        }
    interrupted = []
    for path in (root / "episodes").glob("*/attempt_*/metadata.json"):
        meta = json.loads(path.read_text())
        if meta["end_reason"] == "interrupted":
            interrupted.append(
                {
                    "path": str(path.relative_to(root)),
                    "episode_id": meta["episode_id"],
                    "error": meta["error"],
                }
            )
    pending = [
        str(p.relative_to(root))
        for p in (root / "episodes").glob("*/attempt_*/pending.json")
    ]
    summary["interrupted_attempts"] = interrupted
    summary["pending_attempts"] = pending
    summary["model_disagreements"] = [
        r
        for r in paired
        if (r["original_probability"] > config.success_threshold)
        != (r["finetuned_probability"] > config.success_threshold)
    ]
    atomic_json(output / "metrics.json", summary)
    atomic_json(
        output / "coverage.json",
        {
            "expected": expected,
            "completed": len(rows),
            "paired_final": len(paired),
            "complete_diagnostics": completed_diagnostics,
            "errors": errors,
            "interrupted_attempts": interrupted,
            "pending_attempts": pending,
        },
    )
    fields = [
        "episode_id",
        "task_id",
        "task",
        "initial_state_id",
        "seed",
        "steps",
        "env_success",
        "end_reason",
        "test_only",
        "original_probability",
        "finetuned_probability",
        "original_complete",
        "finetuned_complete",
    ]
    write_csv(output / "episodes.csv", rows, fields)
    task_rows = []
    for model in config.score_models:
        for task, values in summary["models"][model]["per_task"].items():
            task_rows.append(dict(values, model=model, task_id=task))
    write_csv(
        output / "tasks.csv", task_rows, ["model", "task_id"] + list(metrics([], []))
    )
    lines = [
        "# SmolVLA / Robometer %s success evaluation" % config.suite,
        "",
        (
            "TEST-ONLY SMOKE RUN — excluded from formal results."
            if config.test_only
            else "Formal run: fixed natural outcomes, no teacher, no score-driven filtering."
        ),
        "",
        "Completed rollouts: **%d / %d**; paired final scores: **%d**."
        % (len(rows), expected, len(paired)),
        "",
        "| Model | Scored | TP | FP | FN | TN | Precision | Recall | Specificity | F1 | Balanced accuracy | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return "N/A" if value is None else "%.4f" % value

    for model in config.score_models:
        m = summary["models"][model]["overall"]
        lines.append(
            "| %s | %d | %d | %d | %d | %d | %s |"
            % (
                model,
                m["n"],
                m["tp"],
                m["fp"],
                m["fn"],
                m["tn"],
                " | ".join(
                    fmt(m[k])
                    for k in (
                        "precision",
                        "recall",
                        "specificity",
                        "f1",
                        "balanced_accuracy",
                        "roc_auc",
                        "pr_auc",
                    )
                ),
            )
        )
    lines += [
        "",
        "All metrics use p > %.4f. PR-AUC is trapezoidal; average precision is a separate metric in metrics.json."
        % config.success_threshold,
        "Undefined metrics are N/A; no outcome quota or synthetic failures were introduced.",
        "",
        "## Per-task results",
        "",
        "| Task | Natural success / total | Original TP/FP/FN/TN | Finetuned TP/FP/FN/TN |",
        "|---|---:|---|---|",
    ]
    for task in config.task_ids:
        selected = [r for r in rows if r["task_id"] == task]
        cells = [
            "/".join(
                str(summary["models"][m]["per_task"][str(task)][k])
                for k in ("tp", "fp", "fn", "tn")
            )
            for m in config.score_models
        ]
        lines.append(
            "| %s | %d/%d | %s |"
            % (
                task,
                sum(r["env_success"] for r in selected),
                len(selected),
                " | ".join(cells),
            )
        )
    lines += [
        "",
        "## Paired comparison and task macro averages",
        "",
        "| Model | Paired N | Paired precision | Paired recall | Macro balanced accuracy (valid tasks) | Macro ROC-AUC (valid tasks) |",
        "|---|---:|---:|---:|---|---|",
    ]
    for model in config.score_models:
        overall = summary["paired"][model]["overall"]
        macro = summary["paired"][model]["macro"]
        lines.append(
            "| %s | %d | %s | %s | %s (%d) | %s (%d) |"
            % (
                model,
                overall["n"],
                fmt(overall["precision"]),
                fmt(overall["recall"]),
                fmt(macro["balanced_accuracy"]["value"]),
                macro["balanced_accuracy"]["valid_tasks"],
                fmt(macro["roc_auc"]["value"]),
                macro["roc_auc"]["valid_tasks"],
            )
        )
    lines += [
        "",
        "## Terminal-prefix diagnostic rules",
        "",
        "| Model | Rule | N | TP | FP | FN | TN |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in config.score_models:
        for name, m in summary["diagnostics"][model]["rules"].items():
            lines.append(
                "| %s | %s | %d | %d | %d | %d | %d |"
                % (model, name, m["n"], m["tp"], m["fp"], m["fn"], m["tn"])
            )
    lines += [
        "",
        "![Score distributions](scores.png)",
        "",
        "![Terminal-prefix diagnostics](tails.png)",
        "",
        "![Review examples](examples.png)",
        "",
        "## Coverage and interpretation",
        "",
        "Main metrics use each model's valid final predictions. metrics.json also reports both models on the exact common episode set, task macro averages with valid-task counts, and diagnostic-window comparisons.",
        "",
        "Scoring errors are retained in coverage.json. Missing scores are not negative predictions. Interrupted rollout attempts have no success/failure label. Window diagnostics are exploratory, not tuned acceptance rules.",
        "",
        "Images in examples.png are the saved terminal observations used in final sampling. Environment labels are audits; screenshots are not independent proof of task completion.",
        "",
        "Metadata identifies paths and hashes; exact paired input hashes are checked. Results are specific to this student, task/initial-state selection, 20 Hz rollout and checkpoint pair. No claim of held-out generalization or training-data independence is made.",
        "",
        "Completed diagnostic episodes: " + json.dumps(completed_diagnostics),
        "",
        "Protocol complete: **%s**." % summary["complete"],
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    if rows:
        figures(output, rows, episodes, predictions, config)
    return summary


def figures(output, rows, episodes, predictions, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, model in zip(axes, config.score_models):
        for label, color, marker in ((True, "#198875", "o"), (False, "#c84646", "x")):
            selected = [
                r
                for r in rows
                if r["env_success"] == label and r[model + "_probability"] is not None
            ]
            jitter = np.random.default_rng(7).uniform(-0.15, 0.15, len(selected))
            ax.scatter(
                np.array([r["task_id"] for r in selected]) + jitter,
                [r[model + "_probability"] for r in selected],
                color=color,
                marker=marker,
                alpha=0.7,
                label="Env " + str(label),
            )
        ax.axhline(config.success_threshold, color="gray", linestyle="--")
        ax.set(
            title=model,
            xlabel="Task ID",
            ylabel="Final success probability",
            xticks=config.task_ids,
            ylim=(-0.03, 1.04),
        )
        ax.legend()
    fig.savefig(output / "scores.png", dpi=140)
    plt.close(fig)
    priority = sorted(
        rows,
        key=lambda r: (
            not any(
                r[m + "_probability"] is not None
                and (r[m + "_probability"] > config.success_threshold)
                != r["env_success"]
                for m in config.score_models
            ),
            r["task_id"],
            r["episode_id"],
        ),
    )
    selected = []
    for row in priority:
        if row["task_id"] not in [r["task_id"] for r in selected]:
            selected.append(row)
        if len(selected) == 6:
            break
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(11, max(3, len(selected) * 2.5)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, row in zip(axes[:, 0], selected):
        for model in config.score_models:
            scores = predictions.get((model, row["episode_id"]), {})
            keys = sorted(scores, key=int)
            ax.plot(
                [int(k) for k in keys],
                [scores[k]["success_probability"] for k in keys],
                marker="o",
                label=model,
            )
        ax.axhline(config.success_threshold, color="gray", linestyle="--")
        ax.set(
            title="%s | env=%s" % (row["episode_id"], row["env_success"]),
            xlabel="Prefix endpoint",
            ylim=(-0.03, 1.04),
        )
        ax.legend()
    fig.savefig(output / "tails.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(8, 3.5 * len(selected)),
        squeeze=False,
        constrained_layout=True,
    )
    directories = {m["episode_id"]: p for p, m, _ in episodes}
    for index, row in enumerate(selected):
        with np.load(
            directories[row["episode_id"]] / "terminal.npz", allow_pickle=False
        ) as data:
            for column, key in enumerate(("image", "image2")):
                axes[index, column].imshow(data[key])
                axes[index, column].axis("off")
                axes[index, column].set_title(
                    "%s\n%s | env=%s" % (row["episode_id"], key, row["env_success"]),
                    fontsize=9,
                )
    fig.savefig(output / "examples.png", dpi=125)
    plt.close(fig)
