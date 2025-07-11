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
    initial_position: Optional[np.ndarray] = None
    initial_orientation: Optional[np.ndarray] = None


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
        # URDF robots don't need materials and surfaces
        self.entity = self.scene.add_entity(morph=self.morph)

        # Get joint names and DOF indices for control
        self.dofs_idx = list(range(self.entity.n_dofs))

        # # Set control gains for better stability
        # kp = np.array([1000.0] * self.entity.n_dofs)  # Position gains
        # kv = np.array([100.0] * self.entity.n_dofs)   # Velocity gains

        # self.entity.set_dofs_kp(kp, self.dofs_idx)
        # self.entity.set_dofs_kv(kv, self.dofs_idx)

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
            # Apply joint position control using Genesis pattern
            # Set control command before stepping (will be executed in next step)
            self.entity.control_dofs_position(action, self.dofs_idx)
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
        else:
            # Reset to default position slightly above ground
            self.entity.set_pos(np.array([0.0, 0.0, 0.5]))

        if orientation is not None:
            self.entity.set_quat(orientation)
        else:
            # Reset to default orientation (upright)
            self.entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

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
