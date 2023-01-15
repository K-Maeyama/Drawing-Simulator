import os
import sys
from pathlib import Path

sys.path.append(Path().resolve())

import numpy as np
import glob
from tqdm import tqdm

from DigiLeTs.scripts.data import read_original_data
from env import DrawingEnv_Moc1


def fill_traj(target_position, initial_position, norm, z_pos=0):
    positions = [initial_position]
    d_pos = target_position[:2] - positions[-1][:2]
    while np.linalg.norm(d_pos) > norm:
        pos = positions[-1][:2] + (d_pos / np.linalg.norm(d_pos) * norm)
        positions.append(np.hstack([pos, z_pos]))
        d_pos = target_position[:2] - positions[-1][:2]
    return positions


def get_trajectory(instance, initial_position):
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

    observations = dict()
    env = DrawingEnv_Moc1(params)
    observation = env.init()
    for key in observation.keys():
        observations[key] = [observation[key]]
    actions = []
    for t in range(0, len(trajectory) - 1):
        observation, reward, done, info = env.step(trajectory[t])
        for key in observation.keys():
            observations[key].append(observation[key])
        action = trajectory[t + 1][:2] - trajectory[t][:2]
        actions.append(action)

    for key in observations.keys():
        if ("image" in key) or ("mask" == key):
            observations[key] = np.array(observations[key]).astype(np.uint8)
        else:
            observations[key] = np.array(observations[key])
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
                for key in observations.keys():
                    dataset[key] = observations[key]
                dataset["action"] = actions
                dataset["reward"] = rewards
                dataset["done"] = dones
                dataset["digit"] = np.ones_like(dones) * s
                dataset["digit_onehot"] = np.identity(10, dtype=np.float32)[dataset["digit"].astype(np.int16)]

                save_foldername = "{}/{}".format(save_folder, s)
                save_filename = "{}/{}_{}.npy".format(save_foldername, basename, i)
                os.makedirs(save_foldername, exist_ok=True)
                np.save(save_filename, dataset)


if __name__ == "__main__":
    main()
