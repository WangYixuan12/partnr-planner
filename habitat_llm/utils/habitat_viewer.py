#!/usr/bin/env python3
"""
Simple Basic Habitat Viewer - A lightweight GUI for habitat-sim with camera controls.

This application provides a standalone GUI window for viewing habitat-sim environments
with mouse and keyboard camera controls for zoom, rotate, and translate.

Controls:
- Mouse drag (left button): Rotate camera around target
- Mouse scroll: Zoom in/out
- WASD keys: Move camera position
- QE keys: Move camera up/down
- R key: Reset camera to default position
- ESC: Exit application
"""

import os
import sys
import threading
import time
import tkinter as tk
from multiprocessing import Event
from queue import Empty, Full, Queue

import magnum as mn
import numpy as np
from PIL import Image, ImageTk

# Add habitat-lab to path for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../third_party/habitat-lab")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import hydra

from habitat_llm.agent.env import (
    EnvironmentInterface,
    register_actions,
    register_measures,
    register_sensors,
)
from habitat_llm.agent.env.dataset import CollaborationDatasetV0
from habitat_llm.utils import cprint, fix_config, setup_config


class HabitatViewer:
    """
    A simple GUI viewer for habitat-sim environments with camera controls using Tkinter.

    This provides a standalone window for viewing and navigating habitat-sim
    environments with mouse and keyboard camera controls.
    """

    def __init__(self, sim=None, camera_name="third_rgb"):
        # Camera state
        self._camera_position = mn.Vector3(0, 2, -5)
        self._camera_target = mn.Vector3(0, 0, 0)
        self._camera_up = mn.Vector3(0, 1, 0)

        # Camera control parameters
        self._rotation_sensitivity = 0.005
        self._zoom_sensitivity = 0.1
        self._move_sensitivity = 0.1
        self._camera_distance = 5.0
        self._min_distance = 0.5
        self._max_distance = 50.0

        # Input state
        self._mouse_position = (0, 0)
        self._mouse_delta = (0, 0)
        self._mouse_scroll = 0.0
        self._keys_pressed = set()
        self._mouse_buttons_pressed = set()

        # Habitat-sim setup
        self._sim = sim
        self._camera_name = camera_name

        # Tkinter setup
        self._root = None
        self._canvas = None
        self._photo = None
        self._window_size = (1024, 1024)

        # Inter-process communication
        self._image_queue = Queue(maxsize=1)
        self._control_queue = Queue(maxsize=10)
        self._stop_event = Event()

        # Threading communication
        self._gui_thread = None
        self._gui_queue = Queue(maxsize=20)
        self._gui_ready = threading.Event()

    def _mouse_callback(self, event):
        """Tkinter mouse callback for handling mouse events."""
        if event.type == tk.EventType.Motion:
            old_x, old_y = self._mouse_position
            self._mouse_position = (event.x, event.y)
            self._mouse_delta = (event.x - old_x, event.y - old_y)

            # Handle rotation if left button is pressed
            if event.state & 0x100:  # Button1Mask
                self._handle_camera_rotation()

        elif event.type == tk.EventType.ButtonPress and event.num == 1:
            self._mouse_buttons_pressed.add("left")

        elif event.type == tk.EventType.ButtonRelease and event.num == 1:
            if "left" in self._mouse_buttons_pressed:
                self._mouse_buttons_pressed.remove("left")

        elif event.type == tk.EventType.MouseWheel:
            # Handle scroll wheel for zoom (Windows)
            if event.delta > 0:
                self._mouse_scroll = 1.0  # Zoom in
            else:
                self._mouse_scroll = -1.0  # Zoom out
            self._handle_camera_zoom()

        elif event.type == tk.EventType.ButtonPress:
            # Handle scroll wheel for zoom (Linux)
            if event.num == 4:  # Scroll up
                self._mouse_scroll = 1.0  # Zoom in
                self._handle_camera_zoom()
            elif event.num == 5:  # Scroll down
                self._mouse_scroll = -1.0  # Zoom out
                self._handle_camera_zoom()

    def _handle_camera_rotation(self) -> None:
        """Handle mouse-based camera rotation."""
        if "left" in self._mouse_buttons_pressed:
            dx, dy = self._mouse_delta
            # Calculate rotation angles
            yaw_delta = -dx * self._rotation_sensitivity
            pitch_delta = -dy * self._rotation_sensitivity

            # Get camera direction and right vector
            camera_dir = (self._camera_target - self._camera_position).normalized()
            camera_right = mn.math.cross(camera_dir, self._camera_up).normalized()

            # Apply yaw rotation (around up vector)
            yaw_rotation = mn.Quaternion.rotation(mn.Rad(yaw_delta), self._camera_up)
            camera_dir = yaw_rotation.transform_vector(camera_dir)

            # Apply pitch rotation (around right vector)
            pitch_rotation = mn.Quaternion.rotation(mn.Rad(pitch_delta), camera_right)
            camera_dir = pitch_rotation.transform_vector(camera_dir)

            # Update camera position to maintain distance
            self._camera_position = (
                self._camera_target - camera_dir * self._camera_distance
            )

    def _handle_camera_zoom(self) -> None:
        """Handle mouse scroll for zoom."""
        if self._mouse_scroll != 0:
            # Zoom by changing distance
            zoom_factor = 1.0 - self._mouse_scroll * self._zoom_sensitivity
            self._camera_distance *= zoom_factor
            self._camera_distance = np.clip(
                self._camera_distance, self._min_distance, self._max_distance
            )

            # Update camera position to maintain target
            camera_dir = (self._camera_target - self._camera_position).normalized()
            self._camera_position = (
                self._camera_target - camera_dir * self._camera_distance
            )

    def _handle_camera_movement(self) -> None:
        """Handle keyboard-based camera movement."""
        move_delta = self._move_sensitivity

        # Get camera coordinate system
        camera_dir = (self._camera_target - self._camera_position).normalized()
        camera_right = mn.math.cross(camera_dir, self._camera_up).normalized()
        camera_up = mn.math.cross(camera_right, camera_dir).normalized()

        # Forward/backward movement (W/S)
        if "w" in self._keys_pressed:
            self._camera_position += camera_dir * move_delta
            self._camera_target += camera_dir * move_delta
        if "s" in self._keys_pressed:
            self._camera_position -= camera_dir * move_delta
            self._camera_target -= camera_dir * move_delta

        # Left/right movement (A/D)
        if "a" in self._keys_pressed:
            self._camera_position -= camera_right * move_delta
            self._camera_target -= camera_right * move_delta
        if "d" in self._keys_pressed:
            self._camera_position += camera_right * move_delta
            self._camera_target += camera_right * move_delta

        # Up/down movement (Q/E)
        if "q" in self._keys_pressed:
            self._camera_position += camera_up * move_delta
            self._camera_target += camera_up * move_delta
        if "e" in self._keys_pressed:
            self._camera_position -= camera_up * move_delta
            self._camera_target -= camera_up * move_delta

    def _update_camera_transform(self) -> mn.Matrix4:
        """Update and return the camera transform matrix."""
        return mn.Matrix4.look_at(
            self._camera_position, self._camera_target, self._camera_up
        )

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

        if key == "escape":
            self._root.quit()

    def _is_camera_updated(self):
        return (
            self._mouse_delta != (0, 0)
            or self._mouse_scroll != 0
            or "w" in self._keys_pressed
            or "s" in self._keys_pressed
            or "a" in self._keys_pressed
            or "d" in self._keys_pressed
            or "q" in self._keys_pressed
            or "e" in self._keys_pressed
        )

    def _key_release(self, event):
        """Handle key release events."""
        key = event.keysym.lower()
        if key in self._keys_pressed:
            self._keys_pressed.remove(key)

    def run_blocking(self):
        """Main run loop with Tkinter GUI."""
        print("Starting Habitat-Sim Viewer with Tkinter")
        print(
            "Controls: Mouse drag to rotate, mouse scroll to zoom, WASD/QE to move, R to reset, ESC to exit"
        )

        # Create Tkinter window
        self._root = tk.Tk()
        self._root.title("Habitat-Sim Viewer")
        self._root.geometry(f"{self._window_size[0]}x{self._window_size[1]}")

        # Create canvas for image display
        self._canvas = tk.Canvas(
            self._root,
            width=self._window_size[0],
            height=self._window_size[1],
            bg="black",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Bind events
        self._canvas.bind("<Motion>", self._mouse_callback)
        self._canvas.bind("<Button-1>", self._mouse_callback)
        self._canvas.bind("<ButtonRelease-1>", self._mouse_callback)

        # Mouse wheel bindings for different platforms
        self._canvas.bind("<MouseWheel>", self._mouse_callback)  # Windows
        self._canvas.bind("<Button-4>", self._mouse_callback)  # Linux scroll up
        self._canvas.bind("<Button-5>", self._mouse_callback)  # Linux scroll down
        self._root.bind("<MouseWheel>", self._mouse_callback)  # Windows (root level)
        self._root.bind(
            "<Button-4>", self._mouse_callback
        )  # Linux scroll up (root level)
        self._root.bind(
            "<Button-5>", self._mouse_callback
        )  # Linux scroll down (root level)

        self._root.bind("<KeyPress>", self._key_press)
        self._root.bind("<KeyRelease>", self._key_release)

        # Focus on canvas for keyboard input
        self._canvas.focus_set()

        def update():
            """Update function called by Tkinter."""
            # If there is camera motion
            if self._is_camera_updated():
                # Handle camera controls
                self._handle_camera_zoom()
                self._handle_camera_movement()

                # Update camera transform
                camera_transform = self._update_camera_transform()

                # Set camera transform
                rot = mn.Quaternion.from_matrix(camera_transform.rotation_scaling())
                trans = self._camera_position
                self._sim._sensors[
                    self._camera_name
                ]._sensor_object.node.translation = trans
                self._sim._sensors[self._camera_name]._sensor_object.node.rotation = rot
                self._sim._sensors[
                    self._camera_name
                ]._sensor_object.node.transformation = mn.Matrix4.from_(
                    rot.to_matrix(), trans
                )

            # Render observation
            obs = self._sim.get_sensor_observations()
            if self._camera_name in obs:
                # Get the image data from observation
                image_data = obs[self._camera_name]

                # Display using Tkinter
                self._display_observation_tkinter(image_data)

            # Reset input deltas
            self._mouse_delta = (0, 0)
            self._mouse_scroll = 0.0

            # Schedule next update
            self._root.after(16, update)  # ~60 FPS

        # Start update loop
        update()

        # Start Tkinter main loop
        self._root.mainloop()

    def start_thread(self):
        """Start the viewer in a separate thread without blocking."""

        def gui_thread():
            """GUI thread function."""
            # Create Tkinter window
            self._root = tk.Tk()
            self._root.title("Habitat-Sim Viewer")
            self._root.geometry(f"{self._window_size[0]}x{self._window_size[1]}")

            # Create canvas for image display
            self._canvas = tk.Canvas(
                self._root,
                width=self._window_size[0],
                height=self._window_size[1],
                bg="black",
            )
            self._canvas.pack(fill=tk.BOTH, expand=True)

            # Bind events
            self._canvas.bind("<Motion>", self._mouse_callback)
            self._canvas.bind("<Button-1>", self._mouse_callback)
            self._canvas.bind("<ButtonRelease-1>", self._mouse_callback)

            # Mouse wheel bindings for different platforms
            self._canvas.bind("<MouseWheel>", self._mouse_callback)  # Windows
            self._canvas.bind("<Button-4>", self._mouse_callback)  # Linux scroll up
            self._canvas.bind("<Button-5>", self._mouse_callback)  # Linux scroll down
            self._root.bind(
                "<MouseWheel>", self._mouse_callback
            )  # Windows (root level)
            self._root.bind(
                "<Button-4>", self._mouse_callback
            )  # Linux scroll up (root level)
            self._root.bind(
                "<Button-5>", self._mouse_callback
            )  # Linux scroll down (root level)

            self._root.bind("<KeyPress>", self._key_press)
            self._root.bind("<KeyRelease>", self._key_release)

            # Focus on canvas for keyboard input
            self._canvas.focus_set()

            # Signal that GUI is ready
            self._gui_ready.set()

            def update():
                """Update function called by Tkinter."""
                # If there is camera motion
                if self._is_camera_updated():
                    # Handle camera controls
                    self._handle_camera_zoom()
                    self._handle_camera_movement()

                    # Update camera transform
                    camera_transform = self._update_camera_transform()

                    # Set camera transform
                    rot = mn.Quaternion.from_matrix(camera_transform.rotation_scaling())
                    trans = self._camera_position
                    self._sim._sensors[
                        self._camera_name
                    ]._sensor_object.node.translation = trans
                    self._sim._sensors[
                        self._camera_name
                    ]._sensor_object.node.rotation = rot
                    self._sim._sensors[
                        self._camera_name
                    ]._sensor_object.node.transformation = mn.Matrix4.from_(
                        rot.to_matrix(), trans
                    )

                # Get image from main thread
                try:
                    image_data = self._gui_queue.get_nowait()
                    self._display_observation_tkinter(image_data)
                except Empty:
                    pass

                # Reset input deltas
                self._mouse_delta = (0, 0)
                self._mouse_scroll = 0.0

                # Schedule next update
                self._root.after(16, update)  # ~60 FPS

            # Start update loop
            update()

            # Start Tkinter main loop
            self._root.mainloop()

        # Start GUI thread
        self._gui_thread = threading.Thread(target=gui_thread, name="HabitatViewerGUI")
        self._gui_thread.daemon = True
        self._gui_thread.start()

        # Wait for GUI to be ready
        self._gui_ready.wait(timeout=5)

        print("Started Habitat-Sim Viewer in separate thread")
        return self._gui_thread

    def run(self, non_blocking=True):
        if non_blocking:
            self.start_thread()
        else:
            self.run_blocking()

    def update_image(self, image_data):
        """Update the image in the GUI thread (non-blocking)."""
        try:
            # Send image to GUI thread
            self._gui_queue.put_nowait(image_data)
        except Full:
            # Queue is full, skip this frame
            pass

    def stop_thread(self):
        """Stop the viewer thread."""
        if self._gui_thread and self._gui_thread.is_alive():
            if self._root:
                self._root.quit()
            self._gui_thread.join(timeout=5)
            print("Stopped Habitat-Sim Viewer thread")
        else:
            print("No viewer thread to stop")

    def is_thread_running(self):
        """Check if the viewer thread is running."""
        return self._gui_thread and self._gui_thread.is_alive()


# Method to load agent planner from the config
@hydra.main(config_path="../conf")
def run_viewer(config):
    fix_config(config)
    # Setup a seed
    seed = 47668090
    time.time()
    # Setup config
    config = setup_config(config, seed)

    # Initialize habitat-sim environment
    env_interface = None
    sim = None

    # Create dataset
    dataset = CollaborationDatasetV0(config.habitat.dataset)

    # Register sensors, actions, and measures
    register_sensors(config)
    register_actions(config)
    register_measures(config)

    # Initialize environment interface
    env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

    # Get the simulator from the environment interface
    sim = env_interface.env.env.env._env.sim

    # Create viewer with simulator
    app = HabitatViewer(sim=sim, camera_name="gui_rgb")
    app.run(non_blocking=False)

    while True:
        time.sleep(1.0 / 30.0)
        print("Checking if viewer is running")
        obs = sim.get_sensor_observations()
        app.update_image(obs["gui_rgb"])


if __name__ == "__main__":
    cprint(
        "\nStart of the Simple Basic Habitat Viewer.",
        "blue",
    )

    if len(sys.argv) < 2:
        cprint("Error: Configuration file path is required.", "red")
        sys.exit(1)

    # Run viewer with Hydra config
    run_viewer()

    cprint(
        "\nEnd of the Simple Basic Habitat Viewer.",
        "blue",
    )
