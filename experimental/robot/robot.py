#!/usr/bin/env python3
"""
Simple Robot class for Genesis backend.
Handles robot initialization and basic control operations.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import genesis as gs
import numpy as np


@dataclass
class RobotConfig:
    """Configuration for a Genesis robot"""

    urdf_path: str
    friction: float = 0.8
    scale: float = 1.0
    fixed: bool = False


class Robot:
    """Simple robot class for Genesis backend"""

    def __init__(self, config: RobotConfig, scene: gs.Scene):
        """
        Initialize robot in Genesis scene.

        Args:
            config: Robot configuration
            scene: Genesis scene to add robot to
        """
        self.config = config
        self.scene = scene

        # Create robot morphology using URDF
        self.morph = gs.morphs.URDF(
            file=config.urdf_path, scale=config.scale, fixed=config.fixed
        )

        # Create robot material using Tool material for robots
        self.material = gs.materials.Tool(friction=config.friction)

        # Create robot surface
        self.surface = gs.surfaces.Default()

        # Add robot to scene (URDF robots do not need material or surface)
        self.entity = self.scene.add_entity(morph=self.morph)

        # Initialize state
        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion
        self.joint_positions = np.zeros(self.entity.n_dofs)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get robot observations"""
        # Get current state
        self.position = self.entity.get_pos()
        self.orientation = self.entity.get_quat()
        self.joint_positions = self.entity.get_dofs_position()

        return {
            "position": self.position,
            "orientation": self.orientation,
            "joint_positions": self.joint_positions,
            "num_dofs": self.entity.n_dofs,
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
            self.entity.control_dofs_position(action)
        elif action_type == "base_velocity":
            # Apply base velocity control
            self.entity.control_base_velocity(action)
        elif action_type == "magic":
            # Magic movement - directly set position
            if len(action) >= 3:
                self.entity.set_pos(action[:3])
            if len(action) >= 7:
                self.entity.set_quat(action[3:7])
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
        if position is not None:
            self.entity.set_pos(position)
        if orientation is not None:
            self.entity.set_quat(orientation)

        # Reset joint positions to zero
        self.entity.set_dofs_position(np.zeros(self.entity.n_dofs))

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
