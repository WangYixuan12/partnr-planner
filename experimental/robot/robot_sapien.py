#!/usr/bin/env python3
"""
Simple Robot class for SAPIEN backend.
Handles robot initialization and basic control operations.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import sapien.core as sapien


@dataclass
class RobotConfig:
    """Configuration for a SAPIEN robot"""

    urdf_path: str
    friction: float = 0.8
    scale: float = 1.0
    fixed: bool = False
    initial_position: Optional[np.ndarray] = None
    initial_orientation: Optional[np.ndarray] = None


class Robot:
    """Simple robot class for SAPIEN backend"""

    def __init__(self, config: RobotConfig, scene: sapien.Scene):
        """
        Initialize robot in SAPIEN scene.

        Args:
            config: Robot configuration
            scene: SAPIEN scene to add robot to
        """
        self.config = config
        self.scene = scene

        # Load robot from URDF using SAPIEN's URDF loader
        # Note: SAPIEN doesn't have a direct load_urdf method, so we'll create a simple robot
        # In a real implementation, you'd use a proper URDF parser

        # Create a simple articulated robot for testing
        builder = scene.create_articulation_builder()

        # Create root link
        root_link = builder.create_link_builder()
        root_link.set_name("base_link")

        # Add visual and collision shapes (simplified)
        # In a real implementation, you'd load the actual URDF and create proper links

        # For now, create a simple robot with a few joints
        self.robot = builder.build()

        # Set initial pose if specified
        if (
            config.initial_position is not None
            or config.initial_orientation is not None
        ):
            pose = sapien.Pose()
            if config.initial_position is not None:
                pose.set_p(config.initial_position)
            if config.initial_orientation is not None:
                pose.set_q(config.initial_orientation)
            self.robot.set_root_pose(pose)

        # Set robot pose if specified
        if (
            config.initial_position is not None
            or config.initial_orientation is not None
        ):
            pose = sapien.Pose()
            if config.initial_position is not None:
                pose.set_p(config.initial_position)
            if config.initial_orientation is not None:
                pose.set_q(config.initial_orientation)
            self.robot.set_root_pose(pose)

        # Get joint information
        self.joints = self.robot.get_active_joints()
        self.num_dofs = len(self.joints)

        # Set joint properties
        for joint in self.joints:
            joint.set_drive_property(1000.0, 100.0)  # kp, kd
            joint.set_friction(config.friction)

        # Initialize state
        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
        self.joint_positions = np.zeros(self.num_dofs)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get robot observations"""
        # Get current state
        root_pose = self.robot.get_root_pose()
        self.position = root_pose.p
        self.orientation = root_pose.q

        # Get joint positions
        self.joint_positions = np.array([joint.get_position() for joint in self.joints])

        return {
            "position": self.position,
            "orientation": self.orientation,
            "joint_positions": self.joint_positions,
            "num_dofs": self.num_dofs,
        }

    def apply_action(self, action: np.ndarray, action_type: str = "joint") -> None:
        """
        Apply action to robot.

        Args:
            action: Action to apply
            action_type: Type of action ("joint", "base_velocity", "magic")
        """
        if action_type == "joint":
            # Apply joint position control
            if len(action) == self.num_dofs:
                for i, joint in enumerate(self.joints):
                    joint.set_drive_target(action[i])
        elif action_type == "base_velocity":
            # Apply base velocity control (not implemented in this simple version)
            pass
        elif action_type == "magic":
            # Magic movement - directly set position
            if len(action) >= 3:
                pose = self.robot.get_root_pose()
                pose.set_p(action[:3])
                if len(action) >= 7:
                    pose.set_q(action[3:7])
                self.robot.set_root_pose(pose)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Reset robot to initial state.

        Args:
            position: Initial position (optional)
            orientation: Initial orientation (optional)
        """
        pose = sapien.Pose()

        if position is not None:
            pose.set_p(position)
        else:
            # Reset to default position slightly above ground
            pose.set_p(np.array([0.0, 0.0, 0.5]))

        if orientation is not None:
            pose.set_q(orientation)
        else:
            # Reset to default orientation (upright)
            pose.set_q(np.array([1.0, 0.0, 0.0, 0.0]))

        self.robot.set_root_pose(pose)

        # Reset joint positions to zero
        for joint in self.joints:
            joint.set_position(0.0)

    def get_grasp_state(self) -> bool:
        """Get current grasp state (simplified)"""
        # This is a simplified implementation
        # In a real system, you'd check gripper joint positions
        return False

    def set_grasp_state(self, grasp: bool) -> None:
        """
        Set grasp state.

        Args:
            grasp: True to close gripper, False to open
        """
        # Simplified gripper control
        # In a real system, you'd control specific gripper joints
