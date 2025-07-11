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
import tkinter as tk

import magnum as mn
import numpy as np
from PIL import Image, ImageTk

# Add habitat-lab to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../third_party/habitat-lab")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import hydra
from yixuan_utilities.kinematics_helper import KinHelper

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import cprint, fix_config, setup_config


class RobotController:
    """
    An interactive GUI controller for habitat-sim robots with keyboard controls.

    This provides a standalone window for controlling robots with keyboard movement controls.
    """

    def __init__(
        self,
        env_interface: EnvironmentInterface,
        agent_idx=0,
        camera_names=None,
    ):
        # Robot state
        self._robot_position = mn.Vector3(0, 0, 0)
        self._robot_rotation = mn.Quaternion()
        self._robot_yaw = 0.0

        # Head control state
        self._head_alpha = 0.0  # Up/down rotation (pitch)
        self._head_beta = 0.0  # Left/right rotation (yaw)
        self._head_gamma = 0.0  # Roll rotation (less used)

        # Control parameters
        self._movement_speed = 0.1
        self._rotation_speed = 0.05
        self._head_rotation_speed = 0.05

        # Input state
        self._keys_pressed: set[str] = set()

        # Recording state
        self._is_recording = False

        # Habitat-sim setup
        # self._sim = sim
        self._env_interface = env_interface
        self._agent_idx = agent_idx
        self._camera_names = camera_names

        # Tkinter setup
        self._window_size = (1024 * len(self._camera_names), 1024)
        self._root = tk.Tk()
        self._root.title("Robot Controller")
        self._root.geometry(f"{self._window_size[0]}x{self._window_size[1]}")

        # Create canvas for image display
        self._canvas = tk.Canvas(
            self._root,
            width=self._window_size[0],
            height=self._window_size[1],
            bg="black",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._photo = None

        # IK solver
        self._kin_helper = KinHelper("vega")

        if self._camera_names is None:
            self._camera_names = ["third_rgb", "head_rgb"]

    def _reset_robot_and_camera(self) -> None:
        """Reset robot to default position."""
        self._robot_position = mn.Vector3(0, 0, 0)
        self._robot_yaw = 0.0
        self._head_alpha = 0.0
        self._head_beta = 0.0
        self._head_gamma = 0.0

    def get_head_state(self) -> dict:
        """Get current head rotation state.

        Returns:
            dict: Current head rotation angles in radians
        """
        return {
            "alpha": self._head_alpha,  # pitch (up/down)
            "beta": self._head_beta,  # yaw (left/right)
            "gamma": self._head_gamma,  # roll
        }

    def _handle_robot_movement(self) -> dict:
        """Handle keyboard-based robot movement.

        Returns:
            dict: Observation from the environment after applying actions
        """
        actions = np.zeros(38)

        # Forward/backward movement (W/S)
        if "w" in self._keys_pressed:
            actions[7] = 10.0
        if "s" in self._keys_pressed:
            actions[7] = -10.0

        # Rotation (A/D for yaw)
        if "a" in self._keys_pressed:
            actions[8] = 10.0
        if "d" in self._keys_pressed:
            actions[8] = -10.0

        # Rotate head left/right (J/L) - yaw control
        if "j" in self._keys_pressed:
            self._head_beta += self._head_rotation_speed
        if "l" in self._keys_pressed:
            self._head_beta -= self._head_rotation_speed

        # Rotate head up/down (up/down) - pitch control
        if "up" in self._keys_pressed:
            self._head_gamma -= self._head_rotation_speed
        if "down" in self._keys_pressed:
            self._head_gamma += self._head_rotation_speed

        # Roll head left/right (left/right) - roll control (less used)
        if "right" in self._keys_pressed:
            self._head_alpha += self._head_rotation_speed
        if "left" in self._keys_pressed:
            self._head_alpha -= self._head_rotation_speed

        # Apply head movements using IK solver
        if any(
            key in self._keys_pressed
            for key in ["i", "k", "j", "l", "left", "right", "up", "down"]
        ):
            # Access the articulated agent to get current joint positions
            articulated_agent = self._env_interface.sim.get_agent_data(
                self._agent_idx
            ).articulated_agent
            current_qpos = np.array(articulated_agent.sim_obj.joint_positions)
            joint_names = [
                "B_wheel_j1",
                "B_wheel_j2",
                "R_wheel_j1",
                "R_wheel_j2",
                "L_wheel_j1",
                "L_wheel_j2",
                "torso_j1",
                "torso_j2",
                "torso_j3",
                "L_arm_j1",
                "L_arm_j2",
                "L_arm_j3",
                "L_arm_j4",
                "L_arm_j5",
                "L_arm_j6",
                "L_arm_j7",
                "R_arm_j1",
                "R_arm_j2",
                "R_arm_j3",
                "R_arm_j4",
                "R_arm_j5",
                "R_arm_j6",
                "R_arm_j7",
                "head_j1",
                "head_j2",
                "head_j3",
            ]
            current_qpos = self._kin_helper.convert_to_sapien_joint_order(
                current_qpos, joint_names
            )

            curr_head_pose = self._kin_helper.compute_fk_from_link_idx(
                current_qpos, [26]
            )[0]
            target_head_pose = curr_head_pose.copy()
            if "i" in self._keys_pressed:
                target_head_pose[:3, 3] += 0.01 * np.array([0, 0, 1])
            if "k" in self._keys_pressed:
                target_head_pose[:3, 3] -= 0.01 * np.array([0, 0, 1])

            # Apply rotations based on alpha (pitch), beta (yaw), gamma (roll)
            # Alpha controls up/down (pitch around X axis)
            # Beta controls left/right (yaw around Z axis)
            # Gamma controls roll (roll around Y axis)

            # Create rotation matrix from euler angles
            cos_a, sin_a = np.cos(self._head_alpha), np.sin(self._head_alpha)
            cos_b, sin_b = np.cos(self._head_beta), np.sin(self._head_beta)
            cos_g, sin_g = np.cos(self._head_gamma), np.sin(self._head_gamma)

            # Rotation matrix for alpha (pitch around X)
            R_x = np.array([[1, 0, 0], [0, cos_a, -sin_a], [0, sin_a, cos_a]])

            # Rotation matrix for beta (yaw around Z)
            R_z = np.array([[cos_b, -sin_b, 0], [sin_b, cos_b, 0], [0, 0, 1]])

            # Rotation matrix for gamma (roll around Y)
            R_y = np.array([[cos_g, 0, sin_g], [0, 1, 0], [-sin_g, 0, cos_g]])

            # Combined rotation (following sapien example order)
            R_combined = R_z @ R_y @ R_x
            target_head_pose[:3, :3] = R_combined

            # Define active joint mask for head control
            # Based on sapien example, we activate specific joints for head control
            # Adjust these indices based on the actual robot configuration
            active_qmask = np.zeros(26, dtype=bool)
            active_qmask[3] = True  # torso joint
            active_qmask[7:10] = True  # head joints
            active_qmask[12] = True  # additional head joint
            active_qmask[15] = True  # additional head joint

            # Compute IK for head pose (eef_idx=26 for head end effector)
            new_qpos = self._kin_helper.compute_ik_from_mat(
                current_qpos,
                target_head_pose,
                eef_idx=26,
                active_qmask=active_qmask,
            )
            new_qpos = self._kin_helper.convert_from_sapien_joint_order(
                new_qpos, joint_names
            )
            self._env_interface.sim.get_agent_data(
                self._agent_idx
            ).articulated_agent.sim_obj.joint_positions = new_qpos

        # Update agent state in simulator
        curr_qpos = self._env_interface.sim.get_agent_data(
            self._agent_idx
        ).articulated_agent.sim_obj.joint_positions
        self._env_interface.env.step(actions.astype(np.float32))
        self._env_interface.sim.get_agent_data(
            self._agent_idx
        ).articulated_agent.sim_obj.joint_positions = curr_qpos
        obs, reward, done, info = self._env_interface.env.step(
            np.zeros(38).astype(np.float32)
        )
        return obs

    def _display_observation_tkinter(self, image_data):
        """Display the observation image using Tkinter."""
        # Convert image data to RGB format if needed
        if image_data.dtype != np.uint8:
            image_data = (image_data * 255).astype(np.uint8)

        # Ensure image is in RGB format (3 channels)
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            # Already RGB
            pass
        elif len(image_data.shape) == 3 and image_data.shape[2] == 4:
            # RGBA, convert to RGB
            image_data = image_data[:, :, :3]
        else:
            # Single channel, repeat to RGB
            image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_data)

        # Resize to fit window
        pil_image = pil_image.resize(self._window_size, Image.Resampling.LANCZOS)

        # Convert to PhotoImage for Tkinter
        self._photo = ImageTk.PhotoImage(pil_image)

        # Update canvas
        if self._canvas:
            self._canvas.delete("all")
            self._canvas.create_image(
                self._window_size[0] // 2, self._window_size[1] // 2, image=self._photo
            )

    def _key_press(self, event):
        """Handle key press events."""
        key = event.keysym.lower()
        self._keys_pressed.add(key)

        if key == "r":
            self._reset_robot_and_camera()
        elif key == "c":
            self._is_recording = True
            print("Recording started - press 'x' to stop")
        elif key == "x":
            self._is_recording = False
            print("Recording stopped")
        elif key == "escape":
            self._root.quit()

    def _key_release(self, event):
        """Handle key release events."""
        key = event.keysym.lower()
        if key in self._keys_pressed:
            self._keys_pressed.remove(key)

    def run(self, save_views=None) -> dict:
        """Main run loop with Tkinter GUI (blocking mode)."""
        print("Starting Robot Controller with Tkinter")
        print("Controls: WASD to move robot, QE to rotate, R to reset, ESC to exit")
        print("Head controls: I/K for up/down, J/L for left/right, 4/6 for roll")
        print("Recording: C to start recording, X to stop recording")

        # Bind events
        self._canvas.bind("<KeyPress>", self._key_press)
        self._canvas.bind("<KeyRelease>", self._key_release)

        # Focus on canvas for keyboard input
        self._canvas.focus_set()

        self.obs_hist: dict = {}
        if save_views is not None:
            for camera_name in save_views:
                self.obs_hist[camera_name] = []

        def update():
            """Update function called by Tkinter."""
            # Handle robot movement
            obs = self._handle_robot_movement()
            images = []
            for camera_name in self._camera_names:
                if camera_name in obs:
                    # Get the image data from observation
                    image_data = obs[camera_name]
                    images.append(image_data)

                if (
                    save_views is not None
                    and self._is_recording
                    and len(self._keys_pressed) > 0
                ):
                    for camera_name in save_views:
                        self.obs_hist[camera_name].append(obs[camera_name])

            image_data = np.concatenate(images, axis=1)

            # Display using Tkinter
            self._display_observation_tkinter(image_data)

            # Schedule next update
            self._root.after(16, update)  # ~60 FPS

        # Start update loop
        update()

        # Start Tkinter main loop
        self._root.mainloop()

        return self.obs_hist


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def run_robot_controller(config):
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

    # Register sensors, actions, and measures
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Initialize environment interface
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    # Get the simulator from the environment interface
    # sim = env_interface.env.env.env._env.sim

    print("Starting Robot Controller...")

    # Create controller with simulator
    controller = RobotController(
        env_interface=env_interface, agent_idx=0, camera_names=["third_rgb", "head_rgb"]
    )
    controller.run()

    # Main loop
    while True:
        time.sleep(1.0 / 30.0)  # 30 FPS
        if controller.is_thread_running():
            # Get observation and update controller
            # obs = sim.get_sensor_observations()
            obs = env_interface.parse_observations(
                env_interface.step({0: np.zeros(38)})[0]
            )
            if "third_rgb" in obs:
                controller.update_image(obs["third_rgb"])
        else:
            print("Controller stopped")
            break

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
    run_robot_controller()

    cprint(
        "\nEnd of the Interactive Robot Controller.",
        "blue",
    )
