#!/usr/bin/env python3
"""
Interactive Robot Controller - A GUI controller for habitat-sim robots.

This application provides a standalone GUI window for controlling habitat-sim robots
with keyboard controls for movement.

Controls:
- WASD keys: Move robot in the plane (forward/left/backward/right)
- QE keys: Rotate robot left/right
- I/K keys: Move head up/down (pitch)
- J/L keys: Rotate head left/right (yaw)
- 4/6 keys: Rotate head roll (less used)
- R key: Reset robot to default position
- ESC: Exit application
"""

import os
import sys
import time

import numpy as np

# Add habitat-lab to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../third_party/habitat-lab")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import hydra
from yixuan_utilities.hdf5_utils import save_dict_to_hdf5

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import cprint, fix_config, setup_config
from habitat_llm.utils.robot_controller import RobotController


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def data_collection(config):
    fix_config(config)
    # Setup a seed
    seed = 47668090
    time.time()
    # Setup config
    config = setup_config(config, seed)

    # Initialize habitat-sim environment
    env_interface = None

    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)
    episode_id = 17
    dataset.episodes = dataset.episodes[episode_id : episode_id + 1]

    # Register sensors, actions, and measures
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Initialize environment interface
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    # Create controller with simulator
    controller = RobotController(
        env_interface=env_interface, agent_idx=0, camera_names=["third_rgb", "head_rgb"]
    )
    obs_hist = controller.run(save_views=["head_rgb", "head_depth"])
    # for i in range(len(obs_hist["head_rgb"])):
    #     head_rgb = obs_hist["head_rgb"][i]
    #     head_depth = obs_hist["head_depth"][i]
    #     head_depth_vis = np.clip(head_depth, 0, 2) / 2.0
    #     colormap = cm.get_cmap("viridis")
    #     head_depth_vis = colormap(head_depth_vis[:, :, 0])[:, :, :3] * 255
    #     head_depth_vis = head_depth_vis.astype(np.uint8)
    #     concat_img = np.concatenate([head_rgb, head_depth_vis], axis=1)
    #     concat_img = cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("concat_img", concat_img)
    #     cv2.waitKey(30)
    for k in obs_hist:
        obs_hist[k] = np.stack(obs_hist[k], axis=0)
    config_dict = {
        "head_rgb": {
            "dtype": "uint8",
            "chunks": (
                1,
                obs_hist["head_rgb"].shape[1],
                obs_hist["head_rgb"].shape[2],
                3,
            ),
        },
        "head_depth": {
            "dtype": "float32",
            "chunks": (
                1,
                obs_hist["head_depth"].shape[1],
                obs_hist["head_depth"].shape[2],
                1,
            ),
        },
        "head_rgb_extrinsic": {
            "dtype": "float32",
            "chunks": (1, 4, 4),
        },
        "head_rgb_intrinsic": {
            "dtype": "float32",
            "chunks": (1, 3, 3),
        },
    }
    os.system("mkdir -p data/my_data")
    save_dict_to_hdf5(obs_hist, config_dict, f"data/my_data/episode_{episode_id}.hdf5")

    return 0


if __name__ == "__main__":
    cprint(
        "\nStart of the Interactive Robot Controller.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run controller with Hydra config
    data_collection()

    cprint(
        "\nEnd of the Interactive Robot Controller.",
        "blue",
    )
