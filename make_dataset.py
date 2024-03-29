import os
import sys
from pathlib import Path

sys.path.append(Path().resolve())

import numpy as np
import glob
from tqdm import tqdm
import copy

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


def get_image_sequence(instance, params, draw_line=True):
    params = copy.deepcopy(params)
    instance[:, 1] = 1 - instance[:, 1]
    trajectory = get_trajectory(instance=instance, initial_position=params["initial_position"])
    if not draw_line:
        trajectory[:, 2] = 0

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
        # action = trajectory[t + 1][:2] - trajectory[t][:2]
        action = trajectory[t + 1] - trajectory[t]
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


def get_tau(position):
    d_pos = position[1:, :2] - position[:-1, :2]
    norm_body = np.linalg.norm(d_pos, axis=-1)
    norm_digit = copy.deepcopy(norm_body)
    norm_digit[position[1:, 2] == 0] = 0
    total_norm_body = [0]
    total_norm_digit = [0]
    for t in range(len(norm_body)):
        total_norm_body.append(norm_body[:t].sum())
        total_norm_digit.append(norm_digit[:t].sum())
    tau_norm_body = total_norm_body / total_norm_body[-1]
    tau_norm_digit = total_norm_digit / total_norm_digit[-1]

    timestep_body = np.ones(len(norm_body))
    timestep_digit = np.ones(len(norm_digit))
    timestep_digit[position[1:, 2] == 0] = 0
    total_timestep_body = [0]
    total_timestep_digit = [0]
    for t in range(len(timestep_body)):
        total_timestep_body.append(timestep_body[:t].sum())
        total_timestep_digit.append(timestep_digit[:t].sum())
    tau_time_body = total_timestep_body / total_timestep_body[-1]
    tau_time_digit = total_timestep_digit / total_timestep_digit[-1]
    tau = dict(
        tau_norm_body=np.expand_dims(tau_norm_body, -1),
        tau_norm_digit=np.expand_dims(tau_norm_digit, -1),
        tau_time_body=np.expand_dims(tau_time_body, -1),
        tau_time_digit=np.expand_dims(tau_time_digit, -1),
    )
    return tau


def main():
    data_dir = "DigiLeTs/data/preprocessed/complete"
    filenames = glob.glob(os.path.join(data_dir, "*_preprocessed"))

    params = dict(
        size=256,
        line_color=(50, 50, 50),
        line_width=3,
        digit_area=[30, 36, 158, 164],
        initial_position=np.array((0.8, 0.8, 0)),
        max_step=2e2,
    )

    for idx in tqdm(range(len(filenames))):
        filename = filenames[idx]
        basename = os.path.basename(filename)
        participant = read_original_data(filename)
        if idx < 70:
            save_folder = "dataset/Drawing/realistic_no_line/train"
        else:
            save_folder = "dataset/Drawing/realistic_no_line/validation"
        for s, symbol in enumerate(participant["trajectories"]):
            if s > 9:
                # 数字以外は省略
                break
            for i, _instance in enumerate(symbol):
                # 各々，同じsymbolを5回ずつ書いている
                instance = _instance[: participant["lengths"][s, i]]
                observations, actions, rewards, dones = get_image_sequence(instance, params=params, draw_line=False)
                dataset = dict()
                for key in observations.keys():
                    dataset[key] = observations[key]
                tau = get_tau(observations["position"])
                for key in tau.keys():
                    dataset[key] = tau[key]
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
