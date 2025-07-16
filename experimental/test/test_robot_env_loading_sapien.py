#!/usr/bin/env python3
"""
Test for robot and environment loading in SAPIEN planner structure.
Verifies that robot is created in the scene properly.
"""

import os
import sys
import time

import numpy as np
import transforms3d as t3

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from env.env_sapien import Environment, EnvironmentConfig, GraspMode, MovementMode
from robot.robot_sapien import RobotConfig


def test_robot_loading():
    """Test robot loading and entity creation"""
    print("🔧 Testing robot loading...")

    # Create robot configuration with actual URDF path
    robot_config = RobotConfig(
        urdf_path="/home/yixuan/partnr-planner/data/robots/vega-urdf/vega_no_effector.urdf",
        fixed=False,  # Allow robot to move
        initial_position=np.array([0.0, 0.0, 0.5]),  # Position above ground
        initial_orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Upright orientation
    )
    print(f"  ✓ Robot config created: {robot_config}")

    # Create environment with robot
    env_config = EnvironmentConfig(
        show_viewer=False,  # Disable viewer for testing
        robot_config=robot_config,
    )
    env = Environment(env_config)
    print("  ✓ Environment created with robot")

    # Verify robot exists in environment
    assert env.robot is not None, "Robot should be created in environment"
    assert env.robot.num_dofs > 0, "Robot should have DOFs"

    print(f"  ✓ Robot created with {env.robot.num_dofs} DOFs")

    # Step simulation a few times to test stability
    for _i in range(100):
        env.step({"robot": np.zeros(env.robot.num_dofs)})
        time.sleep(0.01)

    print("  ✅ Robot loaded and entity created in scene")
    return True


def test_env_robot_creation():
    """Test environment setup with robot"""
    print("🌍 Testing environment with robot creation...")

    # Create robot and environment config
    robot_config = RobotConfig(
        urdf_path="/home/yixuan/partnr-planner/data/robots/vega-urdf/vega_no_effector.urdf",
        fixed=False,  # Allow robot to move
        initial_position=np.array([0.0, 0.0, 0.5]),  # Position above ground
        initial_orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Upright orientation
    )
    env_config = EnvironmentConfig(robot_config=robot_config)
    print(f"  ✓ Environment config created: grasp_mode={env_config.grasp_mode.value}")

    # Create environment
    env = Environment(env_config)
    print("  ✓ Environment created with robot")

    # Verify robot exists in environment
    assert env.robot is not None, "Robot should be created in environment"
    assert env.robot.num_dofs > 0, "Robot should have DOFs"

    print(f"  ✅ Robot created in environment scene (DOFs: {env.robot.num_dofs})")

    # Step simulation to test physics
    for _ in range(100):
        env.step({"robot": np.zeros(env.robot.num_dofs)})
        time.sleep(0.01)

    return True


def test_environment_modes():
    """Test environment mode switching"""
    print("🔄 Testing environment mode switching...")

    # Create environment with realistic modes
    robot_config = RobotConfig(
        urdf_path="/home/yixuan/partnr-planner/data/robots/vega-urdf/vega_no_effector.urdf",
        fixed=False,  # Allow robot to move
        initial_position=np.array([0.0, 0.0, 0.5]),  # Position above ground
        initial_orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Upright orientation
    )
    env_config = EnvironmentConfig(
        robot_config=robot_config,
        grasp_mode=GraspMode.REALISTIC,
        movement_mode=MovementMode.REALISTIC,
    )
    env = Environment(env_config)

    print(
        f"  📋 Initial modes: grasp={env.config.grasp_mode.value}, movement={env.config.movement_mode.value}"
    )

    # Switch to magic modes
    env.set_grasp_mode(GraspMode.MAGIC)
    env.set_movement_mode(MovementMode.MAGIC)

    print(
        f"  📋 Switched modes: grasp={env.config.grasp_mode.value}, movement={env.config.movement_mode.value}"
    )

    # Verify mode changes
    assert env.config.grasp_mode == GraspMode.MAGIC, "Grasp mode should be MAGIC"
    assert (
        env.config.movement_mode == MovementMode.MAGIC
    ), "Movement mode should be MAGIC"

    print("  ✅ Environment mode switching works correctly")
    return True


def example_realistic_scene_with_viewer():
    """Example: Load a realistic SAPIEN scene with viewer enabled"""
    print("🔄 Testing realistic scene with viewer...")
    # Paths to the stage mesh and scene instance json
    stage_mesh_path = "data/hssd-hab/stages/108736824_177263559.glb"
    scene_instance_path = (
        "data/hssd-hab/scenes-partnr-filtered/108736824_177263559.scene_instance.json"
    )
    # Optionally, add a robot
    # quat = t3.euler.euler2quat(3.0*np.pi/2.0, 0.0, 0.0)
    quat = t3.euler.euler2quat(0.0, 0.0, 0.0)
    robot_config = RobotConfig(
        urdf_path="/home/yixuan/partnr-planner/data/robots/vega-urdf/vega_no_effector.urdf",
        fixed=False,  # Allow robot to move
        initial_position=np.array([-5.0, 4.0, 0.01]),  # Position above ground
        initial_orientation=quat,  # Upright orientation
    )
    env_config = EnvironmentConfig(
        show_viewer=True,
        scene_instance_path=scene_instance_path,
        stage_mesh_path=stage_mesh_path,
        robot_config=robot_config,
    )
    env = Environment(env_config)
    print(
        "Scene loaded and viewer enabled. Interact with the window to inspect the scene."
    )
    # Step the simulation for a while
    for _ in range(10000):
        env.step({"robot": np.zeros(env.robot.num_dofs)})
        time.sleep(0.01)
    env.close()
    print("  ✅ Realistic scene with viewer example completed")


def main():
    """Run all tests with clear visualization"""
    print("🚀 SAPIEN Planner Robot/Env Loading Tests")
    print("=" * 50)

    tests = [
        # ("Robot Loading", test_robot_loading),
        # ("Environment Robot Creation", test_env_robot_creation),
        # ("Environment Modes", test_environment_modes),
        ("Realistic Scene with Viewer", example_realistic_scene_with_viewer)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))

    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")

    print(f"\n📈 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! SAPIEN planner structure is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    main()
