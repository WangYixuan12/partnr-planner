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

from habitat_llm.agent.agent import Agent
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


def localize_table_in_living_room(scene: SceneMemV3, synonyms=None, eps=1e-8):
    """
    Returns: (best_idx, obj, pose_dict)
        - best_idx: index in scene.objects
        - obj: the MapObject (dict-like)
        - pose_dict: {"center": (3,), "R": (3,3), "extent": (3,)}
    """
    # 1) Robust text queries
    #    Use synonyms to cover naming variations; "bedroom" gives contextual bias.
    table_terms = [
        "table",
        "bedside table",
        "nightstand",
        "side table",
        "living room table",
    ]
    if synonyms is not None:
        table_terms = synonyms

    # Object score = max over table synonyms
    table_scores = [scene.query_text(t) for t in table_terms]
    s_table = np.max(np.stack(table_scores, axis=0), axis=0)  # (N_obj,)

    # Context score = bedroom
    s_bedroom = scene.query_text("living room")  # (N_obj,)

    # 2) Fuse scores.
    #    Use log-sum (equivalent to adding logits) for stability with your softmaxed outputs.
    fused = np.log(s_table + eps) + np.log(s_bedroom + eps)  # (N_obj,)

    # 3) Pick the best object
    best_idx = int(np.argmax(fused))
    obj = scene.objects[best_idx]

    # 4) Extract 3D localization from the stored bbox
    bbox = obj["bbox"]
    pose = {
        "center": np.asarray(bbox.center),  # (x, y, z) in world frame
        "R": np.asarray(bbox.R),  # 3x3 rotation
        "extent": np.asarray(bbox.extent),  # box size (dx, dy, dz)
    }
    return best_idx, obj, pose


def exec_skill(
    agent: Agent,
    env_interface: EnvironmentInterface,
    action_name: str,
    action_target: np.ndarray,
    habitat_viewer: HabitatViewer,
    vid_writer: cv2.VideoWriter,
) -> None:
    if action_name == "Navigate":
        while (
            np.linalg.norm(
                np.array(
                    env_interface.agents[0].articulated_agent.ee_transform().translation
                )[0::2]
                - np.array(action_target)[0::2]
            )
            > 1.0
        ):
            low_level_actions, response = agent.process_high_level_action(
                action_name, action_target, {}
            )
            if len(low_level_actions) > 0:
                obs, reward, done, info = env_interface.step({0: low_level_actions})
                habitat_viewer.update_image(obs["third_rgb"])
                vid_writer.write(cv2.cvtColor(obs["third_rgb"], cv2.COLOR_RGB2BGR))
    elif action_name == "Pick":
        while not agent.get_tool_from_name("Pick").skill.grasp_mgr.is_grasped:
            low_level_actions, response = agent.process_high_level_action(
                action_name, action_target, {}
            )
            if len(low_level_actions) > 0:
                obs, reward, done, info = env_interface.step({0: low_level_actions})
                habitat_viewer.update_image(obs["third_rgb"])
                vid_writer.write(cv2.cvtColor(obs["third_rgb"], cv2.COLOR_RGB2BGR))
    elif action_name == "Place":
        while agent.get_tool_from_name("Place").skill.grasp_mgr.is_grasped:
            low_level_actions, response = agent.process_high_level_action(
                action_name, action_target, {}
            )
            if len(low_level_actions) > 0:
                obs, reward, done, info = env_interface.step({0: low_level_actions})
                habitat_viewer.update_image(obs["third_rgb"])
                vid_writer.write(cv2.cvtColor(obs["third_rgb"], cv2.COLOR_RGB2BGR))


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

    scene_mem = load_scene_memory(config, mem_path)

    env_interface: EnvironmentInterface = build_environment(config, episode_idx)

    # initialize habitat viewer
    habitat_viewer = HabitatViewer(sim=env_interface.sim, camera_name="third_rgb")
    habitat_viewer.run(non_blocking=True)
    vid_writer = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1024, 1024)
    )

    # initialize agent and planner
    print(f"Episode {episode_idx} instruction: {instruction}")
    agents = init_agents(config.evaluation.agents, env_interface)
    for agent in agents:
        agent._dry_run = env_interface._dry_run
    # Agents:
    # Method 1: init_agents
    # Method 2: env_interface.sim.get_agent(0)
    # Method 3: env_interface.agents[0]
    planner_conf = config.evaluation.planner
    planner_cls = hydra.utils.instantiate(planner_conf)
    planner: Planner = planner_cls(env_interface=env_interface)
    planner.agents = agents

    salience = scene_mem.query_text("pineapple")
    max_salience_idx = np.argmax(salience)
    sel_obj = scene_mem.objects[max_salience_idx]
    target_pos = mn.Vector3(np.array(sel_obj["pcd"].points).mean(axis=0))
    exec_skill(
        agents[0], env_interface, "Navigate", target_pos, habitat_viewer, vid_writer
    )
    exec_skill(agents[0], env_interface, "Pick", target_pos, habitat_viewer, vid_writer)

    best_idx, obj, pose = localize_table_in_living_room(scene_mem)
    target_pos = mn.Vector3(np.array(obj["pcd"].points).mean(axis=0))
    exec_skill(
        agents[0], env_interface, "Navigate", target_pos, habitat_viewer, vid_writer
    )
    exec_skill(
        agents[0], env_interface, "Place", target_pos, habitat_viewer, vid_writer
    )
    vid_writer.release()
    print("Done")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
