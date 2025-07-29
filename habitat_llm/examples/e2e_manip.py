import os
import sys
from typing import Any

import cv2
import hydra
import magnum as mn
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from task_3d_repr.memory.scene_mem_v3 import SceneMemV3

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
    remove_visual_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.planner.planner import Planner
from habitat_llm.utils import fix_config, setup_config
from habitat_llm.utils.habitat_viewer import HabitatViewer
from habitat_llm.utils.sim import init_agents


def load_config(config_path: str, config_name: str) -> DictConfig:
    """Load a Hydra config file."""
    with hydra.initialize_config_dir(config_path=os.path.abspath(config_path)):
        cfg = hydra.compose(config_name=config_name)
    return cfg


def get_episode_instruction(cfg: DictConfig, episode_idx: int) -> tuple[Any, str]:
    """Load the dataset and return the episode and its instruction by index."""
    # Dynamically import CollaborationDatasetV0
    sys.path.append("..")
    from habitat_llm.agent.env.dataset import CollaborationDatasetV0

    dataset = CollaborationDatasetV0(cfg.habitat.dataset)
    episode = dataset.episodes[episode_idx]
    instruction = getattr(episode, "instruction", None)
    return episode, instruction


def load_scene_memory(cfg: DictConfig, mem_path: str) -> SceneMemV3:
    """Load the scene memory from a file."""
    scene_mem = SceneMemV3(cfg.memory)
    scene_mem.load_mem(mem_path)
    return scene_mem


def print_memory_objects(scene_mem: SceneMemV3) -> None:
    """Print a summary of objects in the loaded scene memory."""
    print(f"Loaded {len(scene_mem.objects)} objects from memory.")
    for idx, obj in enumerate(scene_mem.objects):
        class_name = obj.get("class_name", ["unknown"])[0]
        n_points = obj.get("n_points", [0])[0]
        print(f"Object {idx}: class={class_name}, n_points={n_points}")


def build_environment(cfg: DictConfig, episode_idx: int) -> Any:
    """Build the environment using EnvironmentInterface and set it to the given episode."""
    # Register sensors, actions, and measures as in planner_demo.py
    keep_rgb = cfg.evaluation.use_rgb if "use_rgb" in cfg.evaluation else False
    if not cfg.evaluation.save_video and not keep_rgb:
        remove_visual_sensors(cfg)
    register_sensors(cfg)
    register_actions(cfg)
    register_measures(cfg)

    dataset = CollaborationDatasetV0(cfg.habitat.dataset)
    dataset.episodes = dataset.episodes[episode_idx : episode_idx + 1]
    env_interface = EnvironmentInterface(cfg, dataset=dataset, init_wg=False)
    env_interface.initialize_perception_and_world_graph()

    # Optionally, reset to the desired episode (if API allows)
    # This is a placeholder; actual API may differ
    # env_interface.env.reset(episode_idx=episode_idx)
    return env_interface


def execute_low_level_action(
    env_interface: Any, action_name: str = "move_forward"
) -> None:
    """Execute a low-level action in the environment and print the result."""
    # Try to execute a low-level action; the actual API may differ
    # This is a best guess based on typical Habitat APIs
    print(f"Executing action: {action_name}")
    try:
        # Try the most common API
        result = env_interface.env.step(action_name)
        print(f"Action result: {result}")
    except Exception as e:
        print(f"Failed to execute action '{action_name}': {e}")


@hydra.main(config_path="../conf")
def main(config) -> None:
    fix_config(config)

    seed = 47668090
    config = setup_config(config, seed)

    episode_idx = 17
    mem_path = "/home/yixuan/task_3d_repr/outputs/habitat/episode_17/full_pcd_454_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz"

    # Hydra disables struct for merging, so we do the same as other scripts
    OmegaConf.set_struct(config, False)
    config.hydra = {"output_subdir": None, "run": {"dir": "."}}
    OmegaConf.set_struct(config, True)

    episode, instruction = get_episode_instruction(config, episode_idx)
    print(f"Episode {episode_idx} instruction: {instruction}")

    scene_mem = load_scene_memory(config, mem_path)
    print_memory_objects(scene_mem)

    env_interface: EnvironmentInterface = build_environment(config, episode_idx)

    # initialize habitat viewer
    habitat_viewer = HabitatViewer(
        sim=env_interface.sim, agent_idx=0, camera_name="third_rgb"
    )
    habitat_viewer.run(non_blocking=True)
    vid_writer = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1024, 1024)
    )

    # initialize agent and planner
    agents = init_agents(config.evaluation.agents, env_interface)
    for agent in agents:
        agent._dry_run = env_interface._dry_run
    planner_conf = config.evaluation.planner
    planner_cls = hydra.utils.instantiate(planner_conf)
    planner: Planner = planner_cls(env_interface=env_interface)
    planner.agents = agents

    salience = scene_mem.query_text("pineapple")
    while True:
        try:
            max_salience_idx = np.argmax(salience)
            sel_obj = scene_mem.objects[max_salience_idx]
            target_pos = mn.Vector3(np.array(sel_obj["pcd"].points).mean(axis=0))
            low_level_actions = agents[0].process_high_level_action(
                "Navigate", target_pos, {}
            )
            if len(low_level_actions) > 0:
                obs, reward, done, info = env_interface.step({0: low_level_actions[0]})
                habitat_viewer.update_image(obs["third_rgb"])
                vid_writer.write(cv2.cvtColor(obs["third_rgb"], cv2.COLOR_RGB2BGR))
            else:
                break
        except KeyboardInterrupt:
            break
    vid_writer.release()
    print("Done")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
