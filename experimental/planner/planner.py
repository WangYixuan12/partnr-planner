#!/usr/bin/env python3
"""
Simple Planner class for Genesis backend.
Handles basic planning tasks using different strategies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
from env.env import Environment


class PlannerType(Enum):
    """Planner types"""

    RANDOM = "random"
    HEURISTIC = "heuristic"
    LLM = "llm"


@dataclass
class PlannerConfig:
    """Configuration for planner"""

    planner_type: PlannerType
    max_steps: int = 100
    target_position: Optional[np.ndarray] = None
    target_object: Optional[str] = None


class Planner:
    """Simple planner class for Genesis backend"""

    def __init__(self, config: PlannerConfig, env: Environment):
        """
        Initialize planner.

        Args:
            config: Planner configuration
            env: Environment to plan for
        """
        self.config = config
        self.env = env
        self.step_count = 0
        self.current_plan: list = []

    def plan(
        self, instruction: str, observations: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Generate plan based on instruction and observations.

        Args:
            instruction: Natural language instruction
            observations: Current environment observations

        Returns:
            Dictionary of actions for each agent
        """
        if self.config.planner_type == PlannerType.RANDOM:
            return self._random_plan(observations)
        elif self.config.planner_type == PlannerType.HEURISTIC:
            return self._heuristic_plan(instruction, observations)
        elif self.config.planner_type == PlannerType.LLM:
            return self._llm_plan(instruction, observations)
        else:
            raise ValueError(f"Unknown planner type: {self.config.planner_type}")

    def _random_plan(self, observations: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate random actions"""
        if "robot" not in observations:
            return {}

        robot_obs = observations["robot"]
        num_dofs = robot_obs["num_dofs"]

        # Generate random joint actions
        action = np.random.uniform(-0.1, 0.1, num_dofs)

        return {"robot": action}

    def _heuristic_plan(
        self, instruction: str, observations: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Generate heuristic-based plan"""
        if "robot" not in observations:
            return {}

        robot_obs = observations["robot"]
        num_dofs = robot_obs["num_dofs"]

        # Simple heuristic: move towards target if specified
        if self.config.target_position is not None:
            current_pos = robot_obs["position"]
            target_pos = self.config.target_position

            # Calculate direction to target
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance > 0.1:  # If not close enough
                # Normalize direction
                direction = direction / distance

                # Convert to joint action (simplified)
                action = np.zeros(num_dofs)
                # Set first few DOFs to move in direction
                action[:3] = direction * 0.1

                return {"robot": action}

        # Default: small random action
        action = np.random.uniform(-0.05, 0.05, num_dofs)
        return {"robot": action}

    def _llm_plan(
        self, instruction: str, observations: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Generate plan using LLM (placeholder implementation).
        In a real implementation, this would use an actual LLM.
        """
        if "robot" not in observations:
            return {}

        robot_obs = observations["robot"]
        num_dofs = robot_obs["num_dofs"]

        # Parse instruction for basic commands
        instruction_lower = instruction.lower()

        if "move" in instruction_lower or "go" in instruction_lower:
            # Move action
            action = np.random.uniform(-0.1, 0.1, num_dofs)
            return {"robot": action}
        elif "grasp" in instruction_lower or "pick" in instruction_lower:
            # Grasp action
            action = np.zeros(num_dofs)
            # Set gripper to close (simplified)
            if num_dofs > 0:
                action[-1] = 1.0  # Assume last DOF is gripper
            return {"robot": action}
        elif "place" in instruction_lower or "put" in instruction_lower:
            # Place action
            action = np.zeros(num_dofs)
            # Set gripper to open (simplified)
            if num_dofs > 0:
                action[-1] = -1.0  # Assume last DOF is gripper
            return {"robot": action}
        else:
            # Default action
            action = np.random.uniform(-0.05, 0.05, num_dofs)
            return {"robot": action}

    def update(self, observations: Dict[str, Any], reward: float, done: bool) -> None:
        """
        Update planner with new information.

        Args:
            observations: Current observations
            reward: Current reward
            done: Whether episode is done
        """
        self.step_count += 1

        # Update plan based on new information
        if done:
            self.current_plan = []

    def reset(self) -> None:
        """Reset planner state"""
        self.step_count = 0
        self.current_plan = []

    def is_done(self) -> bool:
        """Check if planning is complete"""
        return self.step_count >= self.config.max_steps
