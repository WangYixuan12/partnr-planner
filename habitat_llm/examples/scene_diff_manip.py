import base64
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, List

import cv2
import hydra
import magnum as mn
import numpy as np
import open3d as o3d
import openai
import torch
from omegaconf import DictConfig, OmegaConf
from task_3d_repr.memory.scene_mem_v3 import SceneMemV3, TSDFGrid

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


@dataclass
class TSDFDiffOutputs:
    """TSDFDiffOutputs"""

    added_pc: o3d.geometry.PointCloud  # voxels in target but not in source
    missing_pc: o3d.geometry.PointCloud  # voxels in source but not in target


def diff_tsdf(
    source_mem: SceneMemV3,
    target_mem: SceneMemV3,
    w_min: float = 3.0,
    conf_min: float = 0.5,
    zero_band: float = 0.25,  # surface band: |tsdf| < zero_band
    free_band: float = 0.5,  # free/away from surface: tsdf > free_band
    cluster_eps: float = 0.08,
    cluster_min_pts: int = 80,
    move_pair_max_dist: float = 0.6,
    size_ratio: float = 3.0,
) -> TSDFDiffOutputs:
    """Diff two scene memories based on TSDF."""
    A = source_mem.tsdf
    B = target_mem.tsdf
    assert A is not None and B is not None, "Both scenes must have TSDF fused."
    # Enforce shared grid (origin, dims, voxel)
    assert (
        np.allclose(A.origin, B.origin)
        and np.all(A.dims == B.dims)
        and A.voxel == B.voxel
    )

    tsdf_A, w_A, v_A = A.tsdf, A.w, A.v
    tsdf_B, w_B, v_B = B.tsdf, B.w, B.v
    common = (
        (w_A >= w_min)
        & (w_B >= w_min)
        & (w_A / v_A >= conf_min)
        & (w_B / v_B >= conf_min)
    )

    diff_thresh = 0.02
    missing_mask = common & (tsdf_A < -diff_thresh) & (tsdf_B > diff_thresh)
    added_mask = common & (tsdf_A > diff_thresh) & (tsdf_B < -diff_thresh)

    # Turn masks into point clouds at voxel centers
    def mask_to_pcd(mask: np.ndarray, grid: TSDFGrid) -> o3d.geometry.PointCloud:
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return o3d.geometry.PointCloud()
        centers = grid.origin + (idx + 0.5) * grid.voxel  # (N,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers.astype(np.float32))
        return pcd

    added_pc = mask_to_pcd(added_mask, A)
    missing_pc = mask_to_pcd(missing_mask, A)

    return TSDFDiffOutputs(added_pc=added_pc, missing_pc=missing_pc)


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


def draw_pts_bbox_on_img(
    image: np.ndarray, K: np.ndarray, world_T_cam: np.ndarray, pts: np.ndarray
) -> np.ndarray:
    """Project points to image, find min/max, and draw a bbox on the image"""
    if len(pts) == 0:
        return image

    cam_T_world = np.linalg.inv(world_T_cam)

    # Transform world points to camera coordinates
    cam_pts = np.dot(cam_T_world[:3, :3], pts.T) + cam_T_world[:3, 3:4]
    cam_pts = cam_pts.T

    # Project to image coordinates
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    img_pts = np.stack(
        [
            fx * cam_pts[:, 0] / cam_pts[:, 2] + cx,
            fy * cam_pts[:, 1] / cam_pts[:, 2] + cy,
        ],
        axis=1,
    )
    img_pts = img_pts.astype(int)  # (N, 2)

    # Filter points within image bounds
    h, w = image.shape[:2]
    valid_mask = (
        (img_pts[:, 0] >= 0)
        & (img_pts[:, 0] < w)
        & (img_pts[:, 1] >= 0)
        & (img_pts[:, 1] < h)
    )
    valid_pts = img_pts[valid_mask]

    if len(valid_pts) == 0:
        return image

    # Find bounding box
    x_min, y_min = valid_pts.min(axis=0) - h // 30
    x_max, y_max = valid_pts.max(axis=0) + h // 30

    # Draw bounding box
    line_width = h // 100
    image = cv2.rectangle(
        image.copy(), (x_min, y_min), (x_max, y_max), (0, 0, 255), line_width
    )

    return image


def encode_image(image: np.ndarray) -> str:
    # save image to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=".")
    cv2.imwrite(tmp_file.name, image)
    with open(tmp_file.name, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ask_gpt_for_comp(
    img_ls_1: List[np.ndarray], img_ls_2: List[np.ndarray], label: str
) -> str:
    """Ask GPT for comparison of two images"""
    client = openai.OpenAI()
    num_img_1 = len(img_ls_1)
    num_img_2 = len(img_ls_2)
    prompt = [
        {
            "type": "input_text",
            "text": f"The first {num_img_1} images are objects highlighted in scene 1, the second {num_img_2} images are objects highlighted in scene 2. Please decide if the object in scene 1 is {label} in scene 2 or not. Return true or false.",
        }
    ]
    for i in range(num_img_1):
        prompt.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encode_image(img_ls_1[i])}",
            }
        )
    for i in range(num_img_2):
        prompt.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encode_image(img_ls_2[i])}",
            }
        )
    response = client.responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": prompt}],
    )
    return response.output_text


def find_objects_from_diff(
    tsdf_diff: TSDFDiffOutputs, scene_mem_1: SceneMemV3, scene_mem_2: SceneMemV3
) -> dict:
    obj_diff: dict = {}
    obj_diff["missing"] = []
    obj_diff["added"] = []
    obj_diff["moved"] = []

    # for missing_pc, 1) cluster 2) find the center 3) match with sim memory
    missing_pc = tsdf_diff.missing_pc
    missing_pc_points = np.asarray(missing_pc.points)
    missing_pc_labels = np.array(missing_pc.cluster_dbscan(eps=0.1, min_points=3))
    for label in np.unique(missing_pc_labels):
        if label < 0:
            continue
        pts = missing_pc_points[missing_pc_labels == label]

        # project those points to the corresponding frame and
        # ask GPT to decide if it is really missing or not
        center = pts.mean(axis=0)
        ix, iy, iz = scene_mem_1.tsdf.coord_to_idx(center[None])[0]
        frame_idx_1 = scene_mem_1.tsdf.frame_idx[ix, iy, iz]
        frame_idx_2 = scene_mem_2.tsdf.frame_idx[ix, iy, iz]
        img_ls_1 = []
        img_ls_2 = []
        for i1, i2 in zip(frame_idx_1, frame_idx_2):
            if i1 < 0 or i2 < 0:
                continue
            rgbd_1 = scene_mem_1.tsdf.rgbd_history[i1]
            rgbd_2 = scene_mem_2.tsdf.rgbd_history[i2]
            K_1 = scene_mem_1.tsdf.K_history[i1]
            K_2 = scene_mem_2.tsdf.K_history[i2]
            world_T_cam_1 = scene_mem_1.tsdf.world_T_cam_history[i1]
            world_T_cam_2 = scene_mem_2.tsdf.world_T_cam_history[i2]
            rgb_1 = (rgbd_1[..., :3] * 255).astype(np.uint8)
            bgr_1 = cv2.cvtColor(rgb_1, cv2.COLOR_RGB2BGR)
            img_1 = draw_pts_bbox_on_img(bgr_1, K_1, world_T_cam_1, pts)
            rgb_2 = (rgbd_2[..., :3] * 255).astype(np.uint8)
            bgr_2 = cv2.cvtColor(rgb_2, cv2.COLOR_RGB2BGR)
            img_2 = draw_pts_bbox_on_img(bgr_2, K_2, world_T_cam_2, pts)
            img_ls_1.append(img_1)
            img_ls_2.append(img_2)
        comp_res = ask_gpt_for_comp(img_ls_1, img_ls_2, "missing")
        if comp_res == "true":
            obj_diff["missing"].append(center)

    # for added_pc, 1) cluster 2) find the center 3) match with sim memory
    added_pc = tsdf_diff.added_pc
    added_pc_points = np.asarray(added_pc.points)
    added_pc_labels = np.array(added_pc.cluster_dbscan(eps=0.05, min_points=20))
    for label in np.unique(added_pc_labels):
        if label < 0:
            continue
        pts = added_pc_points[added_pc_labels == label]
        center = pts.mean(axis=0)
        obj_diff["added"].append(center)

    return obj_diff


@hydra.main(config_path="../conf")
def main(config) -> None:
    fix_config(config)

    seed = 47668090
    config = setup_config(config, seed)

    episode_idx = 17

    config.memory.verbose = False
    path_1 = "/home/yixuan/task_3d_repr/outputs/2025-08-12/18-30-34/full_pcd_250_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz"  # normal_map conf with local range  # noqa
    scene_mem_1 = SceneMemV3(config.memory)
    scene_mem_1.load_mem(path_1)

    path_2 = "/home/yixuan/task_3d_repr/outputs/2025-08-12/16-53-53/full_pcd_168_none_overlap_maskconf0.95_simsum1.2_dbscan.1_merge20_masksub.pkl.gz"  # normal_map conf with local range  # noqa
    scene_mem_2 = SceneMemV3(config.memory)
    scene_mem_2.load_mem(path_2)

    tsdf_diff = diff_tsdf(scene_mem_1, scene_mem_2, w_min=20, conf_min=0.2)
    object_diff = find_objects_from_diff(tsdf_diff, scene_mem_1, scene_mem_2)

    # Hydra disables struct for merging, so we do the same as other scripts
    OmegaConf.set_struct(config, False)
    config.hydra = {"output_subdir": None, "run": {"dir": "."}}
    OmegaConf.set_struct(config, True)

    episode, instruction = get_episode_instruction(config, episode_idx)

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

    # salience = scene_mem.query_text("pineapple")
    # max_salience_idx = np.argmax(salience)
    # sel_obj = scene_mem.objects[max_salience_idx]
    # target_pos = mn.Vector3(np.array(sel_obj["pcd"].points).mean(axis=0))
    target_pos = mn.Vector3(object_diff["missing"][0])
    exec_skill(
        agents[0], env_interface, "Navigate", target_pos, habitat_viewer, vid_writer
    )
    exec_skill(agents[0], env_interface, "Pick", target_pos, habitat_viewer, vid_writer)

    # best_idx, obj, pose = localize_table_in_living_room(scene_mem)
    # target_pos = mn.Vector3(np.array(obj["pcd"].points).mean(axis=0))
    # exec_skill(
    #     agents[0], env_interface, "Navigate", target_pos, habitat_viewer, vid_writer
    # )
    # exec_skill(
    #     agents[0], env_interface, "Place", target_pos, habitat_viewer, vid_writer
    # )
    vid_writer.release()
    print("Done")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))
    main()
