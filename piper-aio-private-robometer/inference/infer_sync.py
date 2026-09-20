import argparse
import csv
import os
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime

import numpy as np
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient, XvlaClient
from utils import (
    InferenceDataRecorder,
    check_keyboard_input,
    get_config,
    get_rollout_observation,
    handle_interactive_mode,
    inference_fn_sync,
    process_action,
    reset_observation_window,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ros_operator import RosOperator

shutdown_event = threading.Event()


class ActiveInferenceTimer:
    """Accumulate model-running wall time while excluding interactive pauses."""

    def __init__(self):
        self.started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.ended_at = None
        self._active_started_at = time.monotonic()
        self._elapsed = 0.0
        self._running = True

    def pause(self):
        if self._running:
            self._elapsed += time.monotonic() - self._active_started_at
            self._running = False
            self.ended_at = datetime.now().astimezone().isoformat(timespec="milliseconds")

    def resume(self):
        if not self._running:
            self._active_started_at = time.monotonic()
            self._running = True

    def elapsed(self):
        if self._running:
            return self._elapsed + time.monotonic() - self._active_started_at
        return self._elapsed


class SavedRolloutTimingLogger:
    """Append inference timing only after its corresponding rollout is saved."""

    FIELDNAMES = (
        "episode_index",
        "task",
        "inference_started_at",
        "inference_ended_at",
        "inference_duration_s",
        "recorded_frames",
        "hdf5_path",
    )

    def __init__(self, path, task):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.task = task
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        print(f"Saved-rollout inference timing summary: {self.path}")

    def append(self, episode_index, timer, recorded_frames, hdf5_path):
        row = {
            "episode_index": episode_index,
            "task": self.task,
            "inference_started_at": timer.started_at,
            "inference_ended_at": timer.ended_at,
            "inference_duration_s": f"{timer.elapsed():.3f}",
            "recorded_frames": recorded_frames,
            "hdf5_path": os.path.abspath(hdf5_path),
        }
        file_exists = os.path.isfile(self.path) and os.path.getsize(self.path) > 0
        with open(self.path, "a", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=self.FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            summary_file.flush()
            os.fsync(summary_file.fileno())
        print(
            f"Inference timing saved for episode {episode_index}: "
            f"{row['inference_duration_s']} seconds"
        )


def save_rollout_with_timing(recorder, timing_logger, timer):
    """Prompt for save and append timing only when the operator chooses SAVE."""
    timer.pause()
    episode_index = recorder.episode_idx
    recorded_frames = len(recorder.actions)
    hdf5_path = os.path.join(recorder.save_dir, f"episode_{episode_index}.hdf5")
    saved = recorder.save_episode()
    if saved and timing_logger is not None:
        timing_logger.append(episode_index, timer, recorded_frames, hdf5_path)
    return saved


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


class _FailurePauseMonitor:
    """Poll the local Robometer dashboard independently of policy inference."""

    def __init__(self, policy, ros_operator, command_lock, poll_interval=0.05):
        self.policy = policy
        self.ros_operator = ros_operator
        self.command_lock = command_lock
        self.poll_interval = poll_interval
        self.pause_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="robometer-failure-monitor", daemon=True
        )

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def clear_pause(self):
        self.pause_event.clear()

    def _run(self):
        while not self.stop_event.is_set() and not shutdown_event.is_set():
            if self.pause_event.is_set():
                self.stop_event.wait(self.poll_interval)
                continue
            if not self.policy.consume_failure_event(min_interval=0.0):
                self.stop_event.wait(self.poll_interval)
                continue

            # Latch first, then hold under the action-publishing lock so no
            # buffered action can be published after the hold command.
            self.pause_event.set()
            try:
                with self.command_lock:
                    self.ros_operator.stop_follower_arms()
            except Exception as exc:
                try:
                    rospy.logerr("Robometer failure hold command failed: %s", exc)
                except Exception:
                    print(f"Robometer failure hold command failed: {exc}")
            self.policy.pause_robometer()
            print("\n\033[31mRobometer detected FAILURE; robot and Robometer paused.\033[0m")


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    if args.model == "openpi":
        policy = OpenpiClient(
            host=args.host,
            port=args.port,
            image_size=args.image_size,
            prompt=config["language_instruction"],
            dashboard_port=args.robometer_dashboard_port if args.enable_robometer else None,
        )
    elif args.model == "xvla":
        policy = XvlaClient(
            host=args.host,
            port=args.port,
            prompt=config["language_instruction"],
            chunk_size=config["chunk_size"],
            dashboard_port=args.robometer_dashboard_port if args.enable_robometer else None,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Live Robometer: {'enabled' if args.enable_robometer else 'disabled'}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    left0 = config["left0"]
    right0 = config["right0"]

    ros_operator.follower_arm_publish_continuous(left0, right0)

    if not policy.standby_robometer():
        raise RuntimeError("Startup aborted: live Robometer could not enter standby")

    print("Warmup the server...")
    policy.warmup()
    print("Server warmed up")

    input("Press Enter to Continue")
    if not policy.reset_episode():
        raise RuntimeError("Startup aborted: live Robometer could not start the episode")
    task_time = time.time()
    ros_operator.follower_arm_publish_continuous(left0, right0)
    recorder = InferenceDataRecorder(args, config, shutdown_event=shutdown_event)
    timing_logger = None
    if recorder.enabled:
        timing_summary_path = args.inference_timing_summary or os.path.join(
            recorder.save_dir, "rollout_inference_times.csv"
        )
        timing_logger = SavedRolloutTimingLogger(timing_summary_path, args.task)
    command_lock = threading.Lock()
    failure_monitor = _FailurePauseMonitor(policy, ros_operator, command_lock)
    failure_monitor.start()

    try:
        # Inference loop
        while not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()
            if args.model == "xvla":
                policy.reset()
            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            force_replan = True
            episode_closed = False
            inference_timer = ActiveInferenceTimer()

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                # Check for keyboard input (space to enter interactive mode)
                failure_pause = failure_monitor.pause_event.is_set()
                key = " " if failure_pause else check_keyboard_input()
                if key == " ":
                    inference_timer.pause()
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        failure_monitor.clear_pause()
                        save_rollout_with_timing(recorder, timing_logger, inference_timer)
                        episode_closed = True
                        if not policy.reset_episode():
                            raise RuntimeError("Reset aborted: live Robometer episode could not be reset")
                        # Reset to starting position
                        ros_operator.follower_arm_publish_continuous(left0, right0)
                        input("Press Enter to Continue")
                        task_time = time.time()
                        break  # Break inner loop to restart
                    elif result == "quit":
                        save_rollout_with_timing(recorder, timing_logger, inference_timer)
                        return  # Exit the function entirely
                    if failure_pause:
                        if not policy.resume_robometer():
                            raise RuntimeError("Continue aborted: live Robometer could not be resumed")
                        failure_monitor.clear_pause()
                    force_replan = True
                    inference_timer.resume()

                # When coming to the end of the action chunk
                if force_replan or t % chunk_size == 0:
                    # Start inference
                    action_buffer = inference_fn_sync(args, config, policy, ros_operator)
                    if action_buffer is None:
                        break
                    assert action_buffer is not None, "Sync inference returned None"
                    assert (
                        action_buffer.shape[0] >= chunk_size
                    ), f"Action chunk length {action_buffer.shape[0]} is smaller than {chunk_size}"
                    force_replan = False

                # Discard an action chunk returned while failure was being detected.
                if failure_monitor.pause_event.is_set():
                    force_replan = True
                    continue

                act = action_buffer[t % chunk_size]
                observation_to_save = get_rollout_observation(args, config, ros_operator) if recorder.enabled else None
                if recorder.enabled and observation_to_save is None:
                    break

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    publish_action = ros_operator.follower_arm_publish
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    publish_action = ros_operator.follower_arm_pose_publish

                with command_lock:
                    if failure_monitor.pause_event.is_set():
                        force_replan = True
                        continue
                    publish_action(left_action, right_action)

                recorder.add_step(observation_to_save, action_to_save)
                t += 1
                print("Published Step", t)
                rate.sleep()

            if not episode_closed:
                save_rollout_with_timing(recorder, timing_logger, inference_timer)
            if shutdown_event.is_set():
                return
    finally:
        failure_active = failure_monitor.pause_event.is_set()
        failure_monitor.stop()
        if failure_active:
            ros_operator.stop_follower_arms()
        else:
            ros_operator.follower_arm_publish_continuous(left0, right0)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--leader_arm_left_topic",
        action="store",
        type=str,
        help="leader_arm_left_topic",
        default="/leader/joint_left",
        required=False,
    )
    parser.add_argument(
        "--leader_arm_right_topic",
        action="store",
        type=str,
        help="leader_arm_right_topic",
        default="/leader/joint_right",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_left_topic",
        action="store",
        type=str,
        help="follower_arm_left_topic",
        default="/follower/joint_left",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_right_topic",
        action="store",
        type=str,
        help="follower_arm_right_topic",
        default="/follower/joint_right",
        required=False,
    )
    parser.add_argument(
        "--pos_cmd_left_topic",
        action="store",
        type=str,
        help="pos_cmd_left_topic",
        default="/follower/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--pos_cmd_right_topic",
        action="store",
        type=str,
        help="pos_cmd_right_topic",
        default="/follower/pos_cmd_right",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_left_pose_topic",
        action="store",
        type=str,
        default="/follower/end_pose_euler_left",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_right_pose_topic",
        action="store",
        type=str,
        default="/follower/end_pose_euler_right",
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        nargs=7,
        help="The maximum change allowed for each joint per timestep (7 values)",
        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2],
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--save_rollout",
        action="store_true",
        help="Save rollout observations/actions to HDF5 episodes",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--save_dir",
        action="store",
        type=str,
        help="Directory used when --save_rollout is set.",
        default="",
        required=False,
    )
    parser.add_argument(
        "--inference_timing_summary",
        type=str,
        default=None,
        help="CSV path for saved rollout inference times (default: <save_dir>/rollout_inference_times.csv)",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint", "eef"],
        help="Control type for the robot arm",
        default="joint",
    )
    parser.add_argument(
        "--host",
        action="store",
        type=str,
        help="Websocket server host",
        default="127.0.0.1",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Websocket server port",
        default=8000,
        required=False,
    )
    parser.add_argument(
        "--enable_robometer",
        action="store_true",
        help="Enable Live Robometer monitoring and dashboard API calls (disabled by default)",
    )
    parser.add_argument(
        "--robometer_dashboard_port",
        type=int,
        default=8080,
        help="Live Robometer dashboard/API port used with --enable_robometer",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Desired image size for OpenPI preprocessing",
        default=(224, 224),
        required=False,
    )
    parser.add_argument(
        "--task",
        action="store",
        type=str,
        help="Task name",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openpi", "xvla"],
        help="Model to use",
        default="openpi",
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args, mode="inference")
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    # Set terminal to raw mode for non-blocking keyboard input
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
