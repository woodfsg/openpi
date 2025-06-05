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
    num_trials_per_task: int = 1  # Number of rollouts per task
     
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "example/libero_test/videos"  # Path to save videos
    probs_out_path: str = "example/libero_test/probs"  # Path to save probabilities

    seed: int = 7  # Random Seed (for reproducibility)

def need_recover(average_prob: float) -> bool:
    return average_prob < 0.4  # Threshold for action probability to determine if recovery is needed

def process_observation_images(obs, resize_size):
    """处理obs中的图像"""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    img = cv2.resize(img, (resize_size, resize_size))
    wrist_img = cv2.resize(wrist_img, (resize_size, resize_size))
    
    return img, wrist_img

def generate_test_action_sequence():
    """生成测试用的动作序列"""
    # 这里创建一个简单的动作序列用于测试
    # 动作格式: [x, y, z, rx, ry, rz, gripper]
    actions = []
    
    # 示例动作序列：先抬起，再移动，最后放下
    # 抬起
    for i in range(15):
        actions.append([0.0, 0.0, 0.15, 0.0, 0.0, 0.0, -1.0])
    
    # 移动
    for i in range(20):
        actions.append([0.15, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    
    # 放下
    for i in range(10):
        actions.append([0.0, 0.0, 0.0, 90/20, 0.0, 0.0, -1.0])
    
    return actions

def test_action_sequence(args: Args) -> None:
    np.random.seed(args.seed)
    
    # 初始化环境
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    
    # 获取任务
    task = task_suite.get_task(0)  # 使用第一个任务
    initial_states = task_suite.get_task_init_states(0)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    
    # 生成测试动作序列
    forward_actions = generate_test_action_sequence()
    reverse_actions = [-np.array(action) for action in reversed(forward_actions)]
    
    # 记录图像
    replay_images = []
    
    # 重置环境
    env.reset()
    obs = env.set_init_state(initial_states[0])
    
    # 等待物体稳定
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    
    # 执行正向动作序列
    logging.info("执行正向动作序列...")
    for action in tqdm.tqdm(forward_actions):
        img, _ = process_observation_images(obs, args.resize_size)
        replay_images.append(img)
        obs, _, _, _ = env.step(action)
    
    size = (args.resize_size, args.resize_size) 
    # 生成纯黑图像（RGB格式，3通道）
    black_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    replay_images.append(black_img)
    
    # 执行反向动作序列
    logging.info("执行反向动作序列...")
    for action in tqdm.tqdm(reverse_actions):
        img, _ = process_observation_images(obs, args.resize_size)
        replay_images.append(img)
        obs, _, _, _ = env.step(action.tolist())
    
    # 保存视频
    task_segment = task_description.replace(" ", "_")
    imageio.mimwrite(
        pathlib.Path(args.video_out_path) / f"action_sequence_test_{task_segment}.mp4",
        [np.asarray(x) for x in replay_images],
        fps=10,
    )
    
    logging.info("测试完成！视频已保存。")

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="example/libero_test/action_sequence_test.log", filemode="w")
    tyro.cli(test_action_sequence)
