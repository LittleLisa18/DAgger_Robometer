import argparse
import os
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
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


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


class PolicySwitcher:
    def __init__(self, student_policy, teacher_policy, initial_policy):
        if initial_policy not in {"student", "teacher"}:
            raise ValueError(f"Unknown initial_policy: {initial_policy}")
        self.policies = {
            "student": student_policy,
            "teacher": teacher_policy,
        }
        self.active_name = initial_policy

    @property
    def active_policy(self):
        return self.policies[self.active_name]

    def switch_to(self, policy_name):
        if policy_name not in self.policies:
            raise ValueError(f"Unknown policy_name: {policy_name}")
        if self.active_name == policy_name:
            print(f"Policy already set to {policy_name}")
            return False
        self.active_name = policy_name
        print(f"Switched active policy to {policy_name}")
        return True


def build_openpi_policy(host, port, dashboard_port, args, config):
    return OpenpiClient(
        host=host,
        port=port,
        image_size=args.image_size,
        prompt=config["language_instruction"],
        dashboard_port=dashboard_port,
    )


def build_policy_switcher(args, config):
    if args.model != "openpi":
        raise ValueError(f"Unknown student model: {args.model}")
    if args.teacher_model != "openpi":
        raise ValueError(f"Unknown teacher model: {args.teacher_model}")

    teacher_host = args.teacher_host if args.teacher_host is not None else args.host
    student_dashboard_port = args.student_dashboard_port if args.enable_robometer else None
    teacher_dashboard_port = args.teacher_dashboard_port if args.enable_robometer else None
    student_policy = build_openpi_policy(args.host, args.port, student_dashboard_port, args, config)
    teacher_policy = build_openpi_policy(teacher_host, args.teacher_port, teacher_dashboard_port, args, config)
    return PolicySwitcher(student_policy, teacher_policy, args.initial_policy)


def infer_active_chunk(args, config, policy_switcher, ros_operator):
    print(f"Requesting {policy_switcher.active_name} policy chunk")
    actions = inference_fn_sync(args, config, policy_switcher.active_policy, ros_operator)
    if actions is None:
        return None
    if actions.shape[0] < config["chunk_size"]:
        raise ValueError(
            f"{policy_switcher.active_name} action chunk length {actions.shape[0]} "
            f"is smaller than {config['chunk_size']}"
        )
    return actions


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    policy_switcher = build_policy_switcher(args, config)

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    left0 = config["left0"]
    right0 = config["right0"]

    ros_operator.follower_arm_publish_continuous(left0, right0)

    print("Warmup the student policy server...")
    policy_switcher.policies["student"].warmup()
    print("Student policy server warmed up")

    print("Warmup the teacher policy server...")
    policy_switcher.policies["teacher"].warmup()
    print("Teacher policy server warmed up")

    print(f"Initial active policy: {policy_switcher.active_name}")

    input("Press enter to continue")
    task_time = time.time()
    ros_operator.follower_arm_publish_continuous(left0, right0)
    recorder = InferenceDataRecorder(args, config, shutdown_event=shutdown_event)

    try:
        # Inference loop
        while not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            reset_observation_window()
            action_buffer = None
            chunk_step = chunk_size
            episode_closed = False

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                # Check for keyboard input (space to enter interactive mode)
                key = check_keyboard_input()
                if key == " ":
                    result, policy_changed = handle_interactive_mode(task_time, policy_switcher=policy_switcher)
                    if result == "reset":
                        recorder.save_episode()
                        episode_closed = True
                        policy_switcher.active_policy.reset_episode()
                        # Reset to starting position
                        ros_operator.follower_arm_publish_continuous(left0, right0)
                        if policy_switcher.active_name != "student":
                            policy_switcher.switch_to("student")
                        input("Press enter to continue")
                        task_time = time.time()
                        break  # Break inner loop to restart
                    if result == "quit":
                        recorder.save_episode()
                        return  # Exit the function entirely
                    if policy_changed:
                        # Drop any queued actions from the old policy; the next step re-infers immediately.
                        action_buffer = None
                        chunk_step = chunk_size

                # When coming to the end of the action chunk, or after a policy switch.
                if chunk_step >= chunk_size:
                    action_buffer = infer_active_chunk(args, config, policy_switcher, ros_operator)
                    if action_buffer is None:
                        break
                    chunk_step = 0

                act = action_buffer[chunk_step]
                chunk_step += 1

                observation_to_save = get_rollout_observation(args, config, ros_operator) if recorder.enabled else None
                if recorder.enabled and observation_to_save is None:
                    break

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    ros_operator.follower_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    ros_operator.follower_arm_pose_publish(left_action, right_action)
                else:
                    raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

                recorder.add_step(
                    observation_to_save,
                    action_to_save,
                    collect_label="teacher" if policy_switcher.active_name == "teacher" else "rollout",
                )
                t += 1
                print(f"Published Step {t} with {policy_switcher.active_name} policy")
                rate.sleep()

            if not episode_closed:
                recorder.save_episode()
            if shutdown_event.is_set():
                return
    finally:
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
        help="Student websocket server host",
        default="127.0.0.1",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Student websocket server port",
        default=8000,
        required=False,
    )
    parser.add_argument(
        "--teacher_host",
        action="store",
        type=str,
        help="Teacher websocket server host. Defaults to --host when omitted.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--teacher_port",
        action="store",
        type=int,
        help="Teacher websocket server port",
        default=8001,
        required=False,
    )
    parser.add_argument("--enable_robometer", action="store_true", help="Enable Live Robometer (disabled by default)")
    parser.add_argument("--student_dashboard_port", type=int, default=8080)
    parser.add_argument("--teacher_dashboard_port", type=int, default=8081)
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
        choices=["openpi"],
        help="Student model to use",
        default="openpi",
        required=False,
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        choices=["openpi"],
        help="Teacher model to use",
        default="openpi",
        required=False,
    )
    parser.add_argument(
        "--initial_policy",
        type=str,
        choices=["student", "teacher"],
        help="Policy used when inference starts",
        default="student",
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
