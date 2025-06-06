import collections
import dataclasses
import logging
import math
import pathlib
import enum

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import matplotlib.pyplot as plt
import traceback
import cv2
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

class RecoverState(enum.Enum):
    NORMAL = "normal"  # 正常执行状态
    START_RECOVER = "start_recover"  # 开始恢复
    RECOVERING = "recovering"  # 正在恢复中
    RECOVER_END = "recover_end"  # 恢复结束

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 10  # Number of rollouts per task
     
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "example/libero_test/videos"  # Path to save videos
    probs_out_path: str = "example/libero_test/probs"  # Path to save probabilities

    seed: int = 7  # Random Seed (for reproducibility)

def need_recover(average_prob: float) -> bool:
    return average_prob < 0.4  # Threshold for action probability to determine if recovery is needed

def process_observation_images(obs, resize_size,state:RecoverState):
    """处理obs中的图像
    """
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    # 在 img 上添加文字
    text = state.value  # 要添加的文字
    font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
    font_scale = 0.3  # 字体大小
    color = (255, 255, 255)  # 文字颜色（白色，RGB格式）
    thickness = 1  # 文字粗细
    position = (15, 15)  # 文字左上角位置 (x, y)

    # 使用 cv2.putText 添加文字
    img_with_text = img.copy()  # 复制图像以避免修改原图
    cv2.putText(img_with_text, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img_with_text, wrist_img

def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.probs_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        # max_steps = 520  # longest training demo has 505 steps
        max_steps = 1600
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    # task_id: int = None
    
    # if task_id is not None:
    #     num_tasks_in_suite=1
        
    # Start evaluation
    total_episodes, total_successes = 0, 0
    recovering_state = RecoverState.NORMAL
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        if task_description != "put both moka pots on the stove" :
            continue

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            state_history=collections.deque(maxlen=40)

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            display_probs = []
            recover_times=0
            action_last = LIBERO_DUMMY_ACTION

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:

                    # obs_store.append(obs)

                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    
                    img, wrist_img = process_observation_images(obs, args.resize_size,recovering_state)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        if recovering_state is not RecoverState.NORMAL:
                            recovering_state = RecoverState.NORMAL
                            logging.info("recovering end")
                            
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
                        result = client.infer(element)
                        action_chunk = result["actions"] 
                        probs = result["probs"]
                        average_prob = np.mean(probs)
                        display_probs.append(average_prob)
                        
                        # check if need recover or not
                        if need_recover(average_prob)and len(state_history)>10 and recover_times<4:
                            t += 1
                            recover_times+=1
                            recover_state=state_history.popleft()
                            recover_state = recover_state + np.random.normal(0, 0.005, len(recover_state))
                            env.regenerate_obs_from_state(recover_state)
                            recovering_state = RecoverState.START_RECOVER
                            logging.info("recover start")
                            action_plan.clear()
                            state_history.clear()
                            continue
                        else:
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                    
                    action = action_plan.popleft()
                    
                    state_history.append(env.get_sim_state())

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    print(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    logging.error(traceback.format_exc())
                    
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{task_episodes}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Save probabilities of the actions taken
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(display_probs)), display_probs, 'b-', label='Average Probability')
            plt.xlabel('Time Step')
            plt.ylabel('Average Probability')
            plt.title(f'Action Probabilities Over Time - {task_segment}')
            plt.grid(True)
            plt.legend()
            plt.savefig(pathlib.Path(args.probs_out_path) / f"probs_{task_segment}_{suffix}_{task_episodes}.png")
            plt.close()


            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
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
    logging.basicConfig(level=logging.INFO,filename="example/libero_test/client_log", filemode="w")
    tyro.cli(eval_libero)
