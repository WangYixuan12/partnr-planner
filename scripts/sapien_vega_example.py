import numpy as np
import sapien
from yixuan_utilities.kinematics_helper import KinHelper


def demo(fix_root_link, balance_passive_force):
    kin_helper = KinHelper("vega")
    scene = sapien.Scene()
    scene.add_ground(-0.1)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    robot = loader.load("data/robots/vega-urdf/vega_no_effector.urdf")

    for link in robot.get_links():
        for shape in link.get_collision_shapes():
            shape.set_collision_groups([1, 1, 17, 0])
    robot.set_root_pose(
        sapien.Pose([0, 0, 0], [1, 0, 0, 0])
    )  # origin, identity rotation quaternion

    init_qpos = np.zeros(robot.dof)
    # final_qpos = np.zeros(robot.dof)
    # final_qpos[10] = 1.57/2.0 # j1
    # final_qpos[11] = -1.57/2.0 # j1
    # final_qpos[18] = 1.57/2.0 # j4
    # final_qpos[19] = 1.57/2.0 # j4
    curr_t = 0.0
    full_cycle = 100.0

    robot.set_qpos(init_qpos)
    active_joints = robot.get_active_joints()

    init_head_pose = (
        robot.get_links()[26].get_entity_pose().to_transformation_matrix().copy()
    )
    target_head_pose = init_head_pose.copy()
    target_head_pose[2, 3] += 0.4
    alpha = 0.2
    beta = 0.4

    for joint_idx, joint in enumerate(active_joints):
        if "torso" in joint.name:
            joint.set_drive_property(stiffness=4000, damping=500, mode="acceleration")
        else:
            joint.set_drive_property(
                stiffness=4000, damping=500, force_limit=1000, mode="force"
            )
        joint.set_drive_target(init_qpos[joint_idx])
    while not viewer.closed:
        # curr_t changes from 0 to full_cycle and then decrease gradually to 0 and loop
        ratio = curr_t / full_cycle - int(curr_t / full_cycle / 2.0) * 2.0
        if ratio > 1.0:
            ratio = 2.0 - ratio
        curr_alpha = alpha * ratio
        curr_beta = beta * ratio
        curr_head_pose = target_head_pose.copy()
        curr_head_pose[:3, 3] = (
            init_head_pose[:3, 3] * (1 - ratio) + target_head_pose[:3, 3] * ratio
        )
        # alpha and beta define the rotation of the head
        # and curr_head_pose[:3, 0] * curr_head_pose[:3, 1] should be 0
        curr_head_pose[:3, 0] = np.array(
            [
                np.cos(curr_alpha) * np.cos(curr_beta),
                np.cos(curr_alpha) * np.sin(curr_beta),
                -np.sin(curr_alpha),
            ]
        )
        curr_head_pose[:3, 1] = np.array([-np.sin(curr_beta), np.cos(curr_beta), 0])
        curr_head_pose[:3, 2] = np.cross(curr_head_pose[:3, 0], curr_head_pose[:3, 1])
        init_qpos = robot.get_qpos()
        eef_idx = 26
        active_qmask = np.zeros(robot.dof, dtype=bool)
        active_qmask[3] = True
        active_qmask[7:10] = True
        active_qmask[12] = True
        active_qmask[15] = True
        curr_qpos = kin_helper.compute_ik_from_mat(
            init_qpos, curr_head_pose, eef_idx=eef_idx, active_qmask=active_qmask
        )

        # curr_qpos = init_qpos + (final_qpos - init_qpos) * ratio
        curr_t += 1.0
        robot.set_qpos(curr_qpos)
        for _ in range(4):  # render every 4 steps
            if balance_passive_force:
                qf = robot.compute_passive_force(
                    gravity=True,
                    coriolis_and_centrifugal=True,
                )
                robot.set_qf(qf)
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-root-link", action="store_true")
    parser.add_argument("--balance-passive-force", action="store_true")
    parser.parse_args()

    demo(
        # fix_root_link=args.fix_root_link,
        # balance_passive_force=args.balance_passive_force,
        fix_root_link=True,
        balance_passive_force=True,
    )


if __name__ == "__main__":
    main()
