#!/usr/bin/env python3
"""
Simple Environment class for Genesis backend.
Manages scene, robot, and provides environment interface.
"""

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import torch
import transforms3d as t3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import genesis as gs
from robot.robot import Robot, RobotConfig
from utils.mesh_utils import process_mesh_for_physics


class GraspMode(Enum):
    """Grasping modes"""

    REALISTIC = "realistic"
    MAGIC = "magic"


class MovementMode(Enum):
    """Movement modes"""

    REALISTIC = "realistic"
    MAGIC = "magic"


@dataclass
class EnvironmentConfig:
    """Configuration for Genesis environment"""

    dt: float = 1.0 / 30.0  # 30 FPS
    show_viewer: bool = False
    grasp_mode: GraspMode = GraspMode.REALISTIC
    movement_mode: MovementMode = MovementMode.REALISTIC
    robot_config: Optional[RobotConfig] = None
    scene_instance_path: Optional[str] = None  # Path to .scene_instance.json
    stage_mesh_path: Optional[str] = None  # Path to .glb stage mesh


class Environment:
    """Simple environment class for Genesis backend"""

    def __init__(self, config: EnvironmentConfig):
        """
        Initialize Genesis environment.

        Args:
            config: Environment configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        viewer_opts = gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=(8, 8, 6),
            camera_lookat=(0, 0, 0),
            camera_up=(0, 1, 0),
            camera_fov=40,
            enable_interaction=True,
        )
        self.scene = gs.Scene(
            show_viewer=config.show_viewer, viewer_options=viewer_opts
        )
        self.robot = None
        self.cameras: dict[str, gs.vis.camera.Camera] = {}
        self.step_count = 0
        self.max_steps = 1000

        self.scene.add_entity(gs.morphs.Plane(pos=np.array([0.0, 0.0, -0.02])))

        # Load scene and objects if specified
        if config.scene_instance_path and config.stage_mesh_path:
            self._load_scene_with_objects(
                config.stage_mesh_path,
                config.scene_instance_path,
                show_viewer=config.show_viewer,
            )

        if config.robot_config is not None:
            self.robot = Robot(config.robot_config, self.scene)

        self._setup_cameras()
        self.scene.build()

        if self.robot is not None:
            if self.config.robot_config.initial_position is not None:
                self.robot.entity.set_pos(self.config.robot_config.initial_position)
            if self.config.robot_config.initial_orientation is not None:
                self.robot.entity.set_quat(self.config.robot_config.initial_orientation)

    def _setup_cameras(self) -> None:
        """Setup cameras for observations"""
        # Robot camera (if robot exists)
        if self.robot is not None:
            robot_camera = self.scene.add_camera(
                model="pinhole",
                res=(640, 480),
                pos=(0.0, 0.0, 0.0),
                lookat=(1.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
                fov=60,
                GUI=False,
            )
            self.cameras["robot"] = robot_camera

        # Third person camera
        third_camera = self.scene.add_camera(
            model="pinhole",
            res=(640, 480),
            pos=(-2.0, 0.0, 1.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=60,
            GUI=False,
        )
        self.cameras["third_person"] = third_camera

    def _load_scene_with_objects(
        self, stage_mesh_path: str, scene_instance_path: str, show_viewer: bool
    ) -> None:
        """
        Load a Genesis scene from a stage mesh and a scene_instance.json file.
        Args:
            stage_mesh_path: Path to the .glb mesh for the room
            scene_instance_path: Path to the .scene_instance.json file
            show_viewer: Whether to enable the interactive viewer
        """
        # Add the main room mesh
        room_pos = np.array([0.0, 0.0, 0.0])
        room_quat = t3.euler.euler2quat(np.pi / 2.0, 0.0, 0.0)
        if stage_mesh_path.endswith(".glb"):
            # Process mesh for physics (decompress + convex decomposition if needed)
            processed_mesh_path = process_mesh_for_physics(
                stage_mesh_path,
                max_hulls=10000,
                resolution=100000,
            )
            room_mesh = gs.morphs.Mesh(
                file=processed_mesh_path,
                pos=room_pos,
                quat=room_quat,
                scale=1.0,
                fixed=True,
                decompose_object_error_threshold=0.0,
                convexify=True,
            )
        else:
            room_mesh = gs.morphs.Mesh(
                file=stage_mesh_path,
                pos=room_pos,
                quat=room_quat,
                scale=1.0,
                fixed=True,
                decompose_object_error_threshold=0.0,
                convexify=True,
            )
        self.scene.add_entity(morph=room_mesh)
        # Load and place all objects from the scene_instance.json
        with open(scene_instance_path, "r") as f:
            scene_data = json.load(f)
        for obj in scene_data.get("object_instances", []):
            template = obj["template_name"]
            mesh_path = f"data/hssd-hab/objects/{template[0]}/{template}.glb"
            # TODO: temporary fix
            if not os.path.exists(mesh_path):
                continue
            processed_mesh_path = process_mesh_for_physics(
                mesh_path,
                max_hulls=1000,
                resolution=100000,
            )
            obj_pos = tuple(obj.get("translation", [0, 0, 0]))
            obj_quat = tuple(obj.get("rotation", [1, 0, 0, 0]))  # wxyz
            room_t_obj = np.eye(4)
            room_t_obj[:3, :3] = t3.quaternions.quat2mat(obj_quat)
            room_t_obj[:3, 3] = obj_pos
            world_t_room = np.eye(4)
            world_t_room[:3, :3] = t3.quaternions.quat2mat(room_quat)
            world_t_room[:3, 3] = room_pos
            world_t_obj = world_t_room @ room_t_obj
            pos = world_t_obj[:3, 3]
            quat = t3.quaternions.mat2quat(world_t_obj[:3, :3])
            scale = obj.get("non_uniform_scale", 1.0)
            motion_type = obj.get("motion_type", "static")
            fixed = motion_type == "static"
            mesh = gs.morphs.Mesh(
                file=processed_mesh_path,
                pos=pos,
                quat=quat,
                scale=scale,
                fixed=fixed,
                decompose_object_error_threshold=0.01,
                convexify=True,
            )
            self.scene.add_entity(morph=mesh)

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state.

        Returns:
            Dictionary containing initial observations
        """
        self.scene.reset()
        self.step_count = 0

        # Reset robot if exists
        if self.robot is not None:
            self.robot.reset()

        return self.get_observations()

    def step(self, actions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Execute actions and step simulation.

        Args:
            actions: Dictionary of actions for each agent

        Returns:
            Dictionary containing observations, rewards, done flag, and info
        """
        # Apply actions
        for agent_name, action in actions.items():
            if agent_name == "robot" and self.robot is not None:
                action_type = "joint"
                if self.config.movement_mode == MovementMode.MAGIC:
                    action_type = "magic"
                self.robot.apply_action(action, action_type)

        # Step simulation
        self.scene.step()
        self.step_count += 1

        # Get observations
        observations = self.get_observations()

        # Check if episode is done
        done = self.step_count >= self.max_steps

        # Calculate reward (simplified)
        reward = 0.0

        # Info dictionary
        info = {
            "step_count": self.step_count,
            "grasp_mode": self.config.grasp_mode.value,
            "movement_mode": self.config.movement_mode.value,
        }

        return {
            "observations": observations,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def get_observations(self) -> Dict[str, Any]:
        """
        Get observations from environment.

        Returns:
            Dictionary containing all observations
        """
        observations = {}

        # Robot observations
        if self.robot is not None:
            robot_obs = self.robot.get_observations()
            observations["robot"] = robot_obs

        # Camera observations
        for camera_name, camera in self.cameras.items():
            try:
                rgb = camera.get_rgb()
                depth = camera.get_depth()
                semantic = camera.get_semantic()

                observations[camera_name] = {
                    "rgb": rgb,
                    "depth": depth,
                    "semantic": semantic,
                }
            except Exception:
                # Camera might not be ready yet
                observations[camera_name] = {
                    "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                    "depth": np.zeros((480, 640), dtype=np.float32),
                    "semantic": np.zeros((480, 640), dtype=np.uint8),
                }

        return observations

    def set_grasp_mode(self, mode: GraspMode) -> None:
        """Set grasping mode"""
        self.config.grasp_mode = mode

    def set_movement_mode(self, mode: MovementMode) -> None:
        """Set movement mode"""
        self.config.movement_mode = mode

    def close(self) -> None:
        """Close environment and cleanup resources"""
