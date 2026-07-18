import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
LIBERO_PLUS_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_PLUS_CLASSIFICATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "third_party"
    / "LIBERO-plus"
    / "libero"
    / "libero"
    / "benchmark"
    / "task_classification.json"
)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    out_path: str = "data/libero_eval"  # Path to save videos
    save_name: str = ""  # Name to save the evaluation results
    save_videos: bool = False  # Whether to save videos of the evaluation

    seed: int = 7  # Random Seed (for reproducibility)
    plus: bool = False  # Whether to use LIBERO-plus category logging.
    plus_summary_only: bool = False  # Summarize existing LIBERO-plus per-task logs without running evaluation.
    resume: bool = False  # Skip completed LIBERO-plus tasks already present in the per-task log.


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    assert args.save_name != "", "Please provide a save name for the evaluation run."

    if args.plus_summary_only:
        if not args.plus:
            raise ValueError("--args.plus-summary-only requires --args.plus.")
        _write_libero_plus_summary(args)
        return

    if args.resume and not args.plus:
        raise ValueError("--args.resume requires --args.plus because resume uses LIBERO-plus per-task logs.")

    if args.plus and args.task_suite_name not in LIBERO_PLUS_SUITES:
        raise ValueError(
            f"LIBERO-plus category reporting only supports {LIBERO_PLUS_SUITES}; got {args.task_suite_name}."
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    plus_task_metadata = None
    plus_task_records = []
    if args.plus:
        plus_task_metadata = _load_libero_plus_task_metadata()[args.task_suite_name]
        _validate_libero_plus_task_metadata(args.task_suite_name, task_suite, plus_task_metadata)
        if args.resume:
            plus_task_records = _load_libero_plus_resume_records(args, task_suite, plus_task_metadata)

    completed_plus_task_names = {record["task_name"] for record in plus_task_records}

    if args.save_videos:
        video_out_path = pathlib.Path(args.out_path) / "videos" / args.save_name / args.task_suite_name
        video_out_path.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes = sum(int(record["episodes"]) for record in plus_task_records)
    total_successes = sum(int(record["successes"]) for record in plus_task_records)
    remaining_task_count = num_tasks_in_suite - len(completed_plus_task_names)
    if args.resume:
        logging.info(f"Resuming {args.task_suite_name}: skipping {len(completed_plus_task_names)} completed tasks.")

    client = None
    if remaining_task_count > 0:
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    else:
        logging.info(f"All {num_tasks_in_suite} tasks in {args.task_suite_name} are already completed.")

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        if task.name in completed_plus_task_names:
            logging.info(f"Skipping completed task {task_id}: {task.name}")
            continue

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        assert client is not None
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            if args.save_videos:
                task_segment = task_description.replace(" ", "_")
                time_suffix = str(time.time())[-4:]
                imageio.mimwrite(
                    video_out_path / f"rollout_{task_segment}_{time_suffix}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        if args.plus:
            task_metadata = plus_task_metadata[task.name]
            plus_task_record = {
                "suite": args.task_suite_name,
                "task_id": task_id,
                "classification_id": task_metadata["id"],
                "task_name": task.name,
                "category": task_metadata["category"],
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": float(task_successes) / float(task_episodes),
            }
            _write_libero_plus_task_result(args, plus_task_record)
            plus_task_records.append(plus_task_record)

        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Save human-readable results.
    (pathlib.Path(args.out_path) / "results").mkdir(parents=True, exist_ok=True)
    results_extension = "md" if args.plus else "txt"
    results_path = pathlib.Path(args.out_path) / "results" / f"{args.save_name}_results.{results_extension}"
    with open(results_path, "a", encoding="utf-8") as f:
        if args.plus:
            f.write("# LIBERO-plus Suite Result\n\n")
            f.write(f"**Save name:** `{args.save_name}`\n\n")
            f.write(
                _make_markdown_table(
                    ["Suite", "Episodes", "Successes", "Success rate"],
                    [
                        [
                            args.task_suite_name,
                            total_episodes,
                            total_successes,
                            _format_rate(float(total_successes) / float(total_episodes)),
                        ]
                    ],
                )
            )
            f.write("\n")
            f.write("## Category Results\n\n")
            f.write(
                _format_libero_plus_suite_summary(
                    args.task_suite_name,
                    _aggregate_libero_plus_records(plus_task_records)[args.task_suite_name],
                    _get_libero_plus_category_counts()[args.task_suite_name],
                )
            )
        else:
            f.write(f"Task suite: {args.task_suite_name}\n")
            f.write(f"Total episodes: {total_episodes}\n")
            f.write(f"Total successes: {total_successes}\n")
            f.write(f"Success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write("\n")


def _get_libero_plus_results_dir(args: Args) -> pathlib.Path:
    results_dir = pathlib.Path(args.out_path) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _get_libero_plus_task_results_path(args: Args, suite: str) -> pathlib.Path:
    return _get_libero_plus_results_dir(args) / f"{args.save_name}_plus_{suite}_tasks.jsonl"


def _get_libero_plus_summary_path(args: Args) -> pathlib.Path:
    return _get_libero_plus_results_dir(args) / f"{args.save_name}_plus_category_summary.md"


def _load_libero_plus_task_metadata() -> dict:
    with open(LIBERO_PLUS_CLASSIFICATION_PATH, encoding="utf-8") as f:
        raw_metadata = json.load(f)

    missing_suites = [suite for suite in LIBERO_PLUS_SUITES if suite not in raw_metadata]
    if missing_suites:
        raise ValueError(f"Missing LIBERO-plus suites in {LIBERO_PLUS_CLASSIFICATION_PATH}: {missing_suites}")

    metadata = {}
    for suite in LIBERO_PLUS_SUITES:
        suite_metadata = {}
        for item in raw_metadata[suite]:
            task_name = item["name"]
            if task_name in suite_metadata:
                raise ValueError(f"Duplicate LIBERO-plus task in {suite}: {task_name}")
            suite_metadata[task_name] = {
                "id": item["id"],
                "category": item["category"],
            }
        metadata[suite] = suite_metadata
    return metadata


def _get_libero_plus_category_counts() -> dict:
    metadata = _load_libero_plus_task_metadata()
    category_counts = {}
    for suite, suite_metadata in metadata.items():
        category_counts[suite] = collections.Counter(item["category"] for item in suite_metadata.values())
    return category_counts


def _validate_libero_plus_task_metadata(suite: str, task_suite, suite_metadata: dict) -> None:
    missing_tasks = []
    for task_id in range(task_suite.n_tasks):
        task_name = task_suite.get_task(task_id).name
        if task_name not in suite_metadata:
            missing_tasks.append(task_name)

    if missing_tasks:
        sample = ", ".join(missing_tasks[:5])
        raise ValueError(
            f"{len(missing_tasks)} tasks in {suite} are missing from {LIBERO_PLUS_CLASSIFICATION_PATH}. "
            f"First missing tasks: {sample}"
        )


def _load_libero_plus_resume_records(args: Args, task_suite, suite_metadata: dict) -> list:
    results_path = _get_libero_plus_task_results_path(args, args.task_suite_name)
    if not results_path.exists():
        logging.info(f"No existing LIBERO-plus per-task log found at {results_path}; starting from task 0.")
        return []

    records = _read_libero_plus_task_results(results_path, args.task_suite_name)
    _validate_libero_plus_task_results(args.task_suite_name, records, {args.task_suite_name: suite_metadata})

    current_task_names = {task_suite.get_task(task_id).name for task_id in range(task_suite.n_tasks)}
    resume_records = []
    ignored_records = []
    for record in records:
        if record["task_name"] not in current_task_names:
            ignored_records.append(record)
            continue
        if int(record["episodes"]) != args.num_trials_per_task:
            logging.warning(
                f"Will rerun {record['task_name']}: logged episodes={record['episodes']} but "
                f"current num_trials_per_task={args.num_trials_per_task}."
            )
            ignored_records.append(record)
            continue
        resume_records.append(record)

    if ignored_records:
        logging.warning(f"Ignoring {len(ignored_records)} existing task records for resume.")
    logging.info(f"Loaded {len(resume_records)} resumable task records from {results_path}.")
    return resume_records


def _write_libero_plus_task_result(args: Args, record: dict) -> None:
    results_path = _get_libero_plus_task_results_path(args, record["suite"])
    records = []
    if results_path.exists():
        records = [
            existing_record
            for existing_record in _read_libero_plus_task_results(results_path, record["suite"])
            if existing_record["task_name"] != record["task_name"]
        ]
    records.append(record)

    with open(results_path, "w", encoding="utf-8") as f:
        for existing_record in records:
            f.write(json.dumps(existing_record, sort_keys=True) + "\n")


def _read_libero_plus_task_results(results_path: pathlib.Path, expected_suite: str) -> list:
    records_by_task = {}
    with open(results_path, encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["suite"] != expected_suite:
                raise ValueError(
                    f"Expected suite {expected_suite} in {results_path}, but line {line_number} has {record['suite']}."
                )
            record["episodes"] = int(record["episodes"])
            record["successes"] = int(record["successes"])
            records_by_task[record["task_name"]] = record
    return list(records_by_task.values())


def _validate_libero_plus_task_results(suite: str, records: list, metadata: dict) -> None:
    for record in records:
        task_name = record["task_name"]
        if task_name not in metadata[suite]:
            raise ValueError(f"Unknown LIBERO-plus task in {suite} results: {task_name}")
        expected_category = metadata[suite][task_name]["category"]
        if record["category"] != expected_category:
            raise ValueError(
                f"Category mismatch for {suite}/{task_name}: result has {record['category']}, "
                f"classification has {expected_category}."
            )


def _aggregate_libero_plus_records(records: list) -> dict:
    stats = {}
    for record in records:
        suite = record["suite"]
        category = record["category"]
        suite_stats = stats.setdefault(suite, {})
        category_stats = suite_stats.setdefault(category, {"tasks": 0, "episodes": 0, "successes": 0})
        category_stats["tasks"] += 1
        category_stats["episodes"] += int(record["episodes"])
        category_stats["successes"] += int(record["successes"])

    for suite_stats in stats.values():
        for category_stats in suite_stats.values():
            episodes = category_stats["episodes"]
            category_stats["success_rate"] = category_stats["successes"] / episodes if episodes else 0.0
    return stats


def _format_rate(success_rate: float) -> str:
    return f"{success_rate * 100:.1f}%"


def _make_markdown_table(headers: list, rows: list) -> str:
    lines = [
        "| " + " | ".join(headers) + " |\n",
        "| " + " | ".join(["---"] * len(headers)) + " |\n",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |\n" for row in rows)
    return "".join(lines)


def _empty_libero_plus_category_stats() -> dict:
    return {"tasks": 0, "episodes": 0, "successes": 0, "success_rate": 0.0}


def _sum_libero_plus_category_stats(suite_stats: dict) -> dict:
    total_stats = {"tasks": 0, "episodes": 0, "successes": 0}
    for category_stats in suite_stats.values():
        total_stats["tasks"] += int(category_stats["tasks"])
        total_stats["episodes"] += int(category_stats["episodes"])
        total_stats["successes"] += int(category_stats["successes"])
    episodes = total_stats["episodes"]
    total_stats["success_rate"] = total_stats["successes"] / episodes if episodes else 0.0
    return total_stats


def _format_libero_plus_suite_summary(suite: str, suite_stats: dict, expected_counts: dict) -> str:
    rows = []
    for category in sorted(expected_counts):
        category_stats = suite_stats.get(category, _empty_libero_plus_category_stats())
        rows.append(
            [
                category,
                category_stats["tasks"],
                expected_counts[category],
                category_stats["episodes"],
                category_stats["successes"],
                _format_rate(category_stats["success_rate"]),
            ]
        )

    suite_average = _sum_libero_plus_category_stats(suite_stats)
    rows.append(
        [
            "**Overall**",
            suite_average["tasks"],
            sum(expected_counts.values()),
            suite_average["episodes"],
            suite_average["successes"],
            _format_rate(suite_average["success_rate"]),
        ]
    )
    return (
        f"### {suite}\n\n"
        + _make_markdown_table(
            ["Category", "Logged tasks", "Expected tasks", "Episodes", "Successes", "Success rate"],
            rows,
        )
    )


def _write_libero_plus_summary(args: Args) -> None:
    metadata = _load_libero_plus_task_metadata()
    category_counts = _get_libero_plus_category_counts()
    missing_paths = [
        _get_libero_plus_task_results_path(args, suite)
        for suite in LIBERO_PLUS_SUITES
        if not _get_libero_plus_task_results_path(args, suite).exists()
    ]
    if missing_paths:
        missing_paths_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing LIBERO-plus per-task result files:\n{missing_paths_text}")

    all_records = []
    records_by_suite = {}
    for suite in LIBERO_PLUS_SUITES:
        records = _read_libero_plus_task_results(_get_libero_plus_task_results_path(args, suite), suite)
        _validate_libero_plus_task_results(suite, records, metadata)
        records_by_suite[suite] = records
        all_records.extend(records)

    stats = _aggregate_libero_plus_records(all_records)
    all_categories = sorted({category for counts in category_counts.values() for category in counts})

    lines = [
        "# LIBERO-plus Category Summary\n",
        "\n",
        f"**Save name:** `{args.save_name}`\n",
        "\n",
        "## Per-Suite Category Results\n",
        "\n",
    ]
    suite_average_rows = []
    all_case_stats = {"tasks": 0, "expected_tasks": 0, "episodes": 0, "successes": 0}
    for suite in LIBERO_PLUS_SUITES:
        suite_stats = stats.get(suite, {})
        suite_average = _sum_libero_plus_category_stats(suite_stats)
        expected_total = sum(category_counts[suite].values())
        logged_total = len(records_by_suite[suite])

        lines.append(_format_libero_plus_suite_summary(suite, suite_stats, category_counts[suite]))
        if logged_total != expected_total:
            lines.append(f"\n> WARNING: logged {logged_total} of {expected_total} expected tasks for `{suite}`.\n")
        lines.append("\n")

        suite_average_rows.append(
            [
                suite,
                suite_average["tasks"],
                expected_total,
                suite_average["episodes"],
                suite_average["successes"],
                _format_rate(suite_average["success_rate"]),
            ]
        )
        all_case_stats["tasks"] += suite_average["tasks"]
        all_case_stats["expected_tasks"] += expected_total
        all_case_stats["episodes"] += suite_average["episodes"]
        all_case_stats["successes"] += suite_average["successes"]

    all_case_success_rate = (
        all_case_stats["successes"] / all_case_stats["episodes"] if all_case_stats["episodes"] else 0.0
    )
    suite_average_rows.append(
        [
            "**All cases**",
            all_case_stats["tasks"],
            all_case_stats["expected_tasks"],
            all_case_stats["episodes"],
            all_case_stats["successes"],
            _format_rate(all_case_success_rate),
        ]
    )
    lines.extend(
        [
            "## Suite Averages\n",
            "\n",
            _make_markdown_table(
                ["Suite", "Logged tasks", "Expected tasks", "Episodes", "Successes", "Average success rate"],
                suite_average_rows,
            ),
            "\n",
            "## Four-Suite Category Averages\n",
            "\n",
        ]
    )

    category_average_rows = []
    for category in all_categories:
        suite_rates = []
        for suite in LIBERO_PLUS_SUITES:
            category_stats = stats.get(suite, {}).get(category, _empty_libero_plus_category_stats())
            suite_rates.append(category_stats["success_rate"])
        average_rate = sum(suite_rates) / len(suite_rates)
        category_average_rows.append(
            [category, _format_rate(average_rate), *[_format_rate(success_rate) for success_rate in suite_rates]]
        )
    lines.append(
        _make_markdown_table(
            ["Category", "Average success rate", "libero_spatial", "libero_object", "libero_goal", "libero_10"],
            category_average_rows,
        )
    )

    summary_path = _get_libero_plus_summary_path(args)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logging.info(f"Wrote LIBERO-plus category summary to {summary_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
