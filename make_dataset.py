import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(Path().resolve())
sys.path.append(os.path.join(Path().resolve(), "DigiLeTs/scripts"))

import glob
import cv2
import copy

from data import read_original_data
from env import DrawingEnv_Moc1

from tqdm import tqdm


def fill_traj(target_position, initial_position, norm, z_pos=0):
    positions = [initial_position]
    d_pos = target_position[:2] - positions[-1][:2]
    while np.linalg.norm(d_pos) > norm:
        pos = positions[-1][:2] + (d_pos / np.linalg.norm(d_pos) * norm)
        positions.append(np.hstack([pos, z_pos]))
        d_pos = target_position[:2] - positions[-1][:2]
    return positions


def get_trajectory(instance, initial_position):
    # initial_position = np.array([0.8, 0.2, 0])
    norm = np.linalg.norm(instance[1:, :2] - instance[:-1, :2], axis=-1)
    norm_mean = norm[norm > 0].mean()
    trajectory = fill_traj(target_position=instance[0], initial_position=initial_position, norm=norm_mean)

    for i in range(0, len(instance) - 1):
        if (instance[i][5] == 0 and instance[i + 1][5] == 0) and (instance[i + 1][3] == 0):
            position = [np.array((instance[i + 1][0], instance[i + 1][1], 1))]
        else:
            position = fill_traj(target_position=instance[i + 1], initial_position=trajectory[-1], norm=norm_mean)
        trajectory += position

    trajectory += fill_traj(target_position=initial_position, initial_position=trajectory[-1], norm=norm_mean)
    trajectory = np.array(trajectory)
    return trajectory


def get_image_sequence(instance, size=256, line_width=3, threshold=None):
    initial_position = np.array((0.8, 0.8, 0))
    params = dict(
        size=256,
        line_color=(50, 50, 50),
        line_width=3,
        digit_area=[30, 36, 158, 164],
        initial_position=initial_position,
        max_step=2e2,
    )
    instance[:, 1] = 1 - instance[:, 1]
    trajectory = get_trajectory(instance=instance, initial_position=initial_position)

    env = DrawingEnv_Moc1(params)
    observation = env.init()
    images = [observation["image"]]
    mask = [observation["mask"]]
    actions = []
    positions = [observation["position"]]
    for t in range(0, len(trajectory) - 1):
        observation, reward, done, info = env.step(trajectory[t])
        images.append(observation["image"])
        positions.append(observation["position"])
        mask.append(observation["mask"])
        action = trajectory[t + 1][:2] - trajectory[t][:2]
        actions.append(action)

    images = np.array(images).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    positions = np.array(positions)
    observations = dict(image=images, mask=mask, position=positions)
    actions = np.array(actions)
    rewards = np.zeros(len(actions))
    dones = np.zeros(len(actions))
    dones[-1] = 1
    return observations, actions, rewards, dones


def main():
    data_dir = "DigiLeTs/data/preprocessed/complete"
    filenames = glob.glob(os.path.join(data_dir, "*_preprocessed"))

    for idx in tqdm(range(len(filenames))):
        filename = filenames[idx]
        basename = os.path.basename(filename)
        participant = read_original_data(filename)
        if idx < 70:
            save_folder = "dataset/Drawing/realistic/train"
        else:
            save_folder = "dataset/Drawing/realistic/validation"
        for s, symbol in enumerate(participant["trajectories"]):
            if s > 9:
                # 数字以外は省略
                break
            for i, _instance in enumerate(symbol):

                instance = _instance[: participant["lengths"][s, i]]
                observations, actions, rewards, dones = get_image_sequence(instance, size=64, line_width=3)
                dataset = dict()
                dataset["image"] = observations["image"]
                dataset["mask"] = observations["mask"]
                dataset["position"] = observations["position"]
                dataset["action"] = actions
                dataset["reward"] = rewards
                dataset["done"] = dones

                save_foldername = "{}/{}".format(save_folder, s)
                save_filename = "{}/{}_{}.npy".format(save_foldername, basename, i)
                # print(save_filename)
                os.makedirs(save_foldername, exist_ok=True)
                np.save(save_filename, dataset)


if __name__ == "__main__":
    main()
