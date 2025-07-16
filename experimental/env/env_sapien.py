#!/usr/bin/env python3
"""
Simple Environment class for SAPIEN backend.
Manages scene, robot, and provides environment interface.
"""

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import transforms3d as t3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import sapien.core as sapien
from robot.robot_sapien import Robot, RobotConfig
from sapien.utils import Viewer
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
    """Configuration for SAPIEN environment"""

    dt: float = 1.0 / 30.0  # 30 FPS
    show_viewer: bool = False
    grasp_mode: GraspMode = GraspMode.REALISTIC
    movement_mode: MovementMode = MovementMode.REALISTIC
    robot_config: Optional[RobotConfig] = None
    scene_instance_path: Optional[str] = None  # Path to .scene_instance.json
    stage_mesh_path: Optional[str] = None  # Path to .glb stage mesh


class Environment:
    """Simple environment class for SAPIEN backend"""

    def __init__(self, config: EnvironmentConfig):
        """
        Initialize SAPIEN environment.

        Args:
            config: Environment configuration
        """
        self.config = config

        # Initialize SAPIEN engine
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        # Create scene
        self.scene = self.engine.create_scene()
        self.scene.set_timestep(config.dt)

        # # Add ground plane
        # self.scene.add_ground(altitude=0.0)

        # Add some lights so that you can observe the scene
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        # Initialize components
        self.robot = None
        self.cameras: dict = {}
        self.step_count = 0
        self.max_steps = 1000

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

        # Create viewer if requested
        if config.show_viewer:
            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(8, 8, 6)
            self.viewer.set_camera_rpy(0, -0.5, 0.785)

    def _setup_cameras(self) -> None:
        """Setup cameras for observations"""
        # Robot camera (if robot exists)
        if self.robot is not None:
            robot_camera = self.scene.add_camera(
                name="robot_camera",
                width=640,
                height=480,
                fovy=60,
                near=0.1,
                far=100.0,
            )
            self.cameras["robot"] = robot_camera

        # Third person camera
        third_camera = self.scene.add_camera(
            name="third_person_camera",
            width=640,
            height=480,
            fovy=60,
            near=0.1,
            far=100.0,
        )
        self.cameras["third_person"] = third_camera

    def _load_scene_with_objects(
        self, stage_mesh_path: str, scene_instance_path: str, show_viewer: bool
    ) -> None:
        """
        Load a SAPIEN scene from a stage mesh and a scene_instance.json file.
        Args:
            stage_mesh_path: Path to the .glb mesh for the room
            scene_instance_path: Path to the .scene_instance.json file
            show_viewer: Whether to enable the interactive viewer
        """
        # Add the main room mesh
        room_pos = np.array([0.0, 0.0, 0.0])
        room_quat = t3.euler.euler2quat(np.pi / 2.0, 0.0, 0.0)

        # Process mesh for physics (decompress + convex decomposition if needed)
        processed_mesh_path = process_mesh_for_physics(
            stage_mesh_path,
            max_hulls=10000,
            resolution=100000,
        )

        # Load mesh as SAPIEN actor (simplified for testing)
        # Note: In a real implementation, you'd properly load the mesh
        builder = self.scene.create_actor_builder()
        # For now, create a simple box as placeholder
        builder.add_convex_collision_from_file(processed_mesh_path)
        builder.add_visual_from_file(processed_mesh_path)
        room_actor = builder.build_static()
        room_actor.set_pose(sapien.Pose(room_pos, room_quat))

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
            obj.get("non_uniform_scale", 1.0)
            motion_type = obj.get("motion_type", "static")
            fixed = motion_type == "static"

            builder = self.scene.create_actor_builder()
            # For now, create simple shapes as placeholders
            builder.add_convex_collision_from_file(processed_mesh_path)
            builder.add_visual_from_file(processed_mesh_path)

            actor = builder.build_static() if fixed else builder.build()

            actor.set_pose(sapien.Pose(pos, quat))

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
        self.scene.update_render()
        if self.config.show_viewer:
            self.viewer.render()
        self.step_count += 1

        # Get observations
        observations = self.get_observations()

        # Check if episode is done
        done = self.step_count >= self.max_steps

        # Calculate reward (placeholder)
        reward = 0.0

        return {
            "observations": observations,
            "reward": reward,
            "done": done,
            "info": {"step_count": self.step_count},
        }

    def get_observations(self) -> Dict[str, Any]:
        """
        Get current observations from environment.

        Returns:
            Dictionary containing observations
        """
        observations: dict = {
            "step_count": self.step_count,
            "cameras": {},
        }

        # Get robot observations if robot exists
        if self.robot is not None:
            observations["robot"] = self.robot.get_observations()

        # Get camera observations (simplified for now)
        for camera_name in list(self.cameras.keys()):
            # For now, just add placeholder camera data
            # In a real implementation, you'd properly render camera images
            observations["cameras"][camera_name] = {
                "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
                "depth": np.zeros((480, 640), dtype=np.float32),
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
        if hasattr(self, "viewer"):
            self.viewer.close()
        self.engine = None
