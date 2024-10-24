import itertools
import random

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# test
def fk_indep(conf):
    Nsegs = len(conf["k"])
    assert len(conf["l"]) == Nsegs and len(conf["alpha"]) == Nsegs
    assert all(conf["l"] > 0) and all(conf["k"] >= 0)

    poses = np.zeros((6,))
    Told = np.eye(4)
    for seg in range(Nsegs):
        T = trans_mat(conf["l"][seg], conf["k"][seg], conf["alpha"][seg])
        Told = np.matmul(Told, T)
        pos = Told[0:3, -1]
        rot = R.from_matrix(Told[0:3, 0:3])
        euler_angles = rot.as_euler('zyx', degrees=False)
        pose_temp = np.hstack((pos, euler_angles))
        poses = np.vstack((poses, pose_temp))

    return poses


def fk_dep(act, R):
    Nsegs = act.shape[0]

    conf = {}
    conf["l"] = np.array([])
    conf["k"] = np.array([])
    conf["alpha"] = np.array([])

    for seg in range(Nsegs):
        l = np.mean(act[seg, :])
        conf["l"] = np.append(conf["l"], l)
        k = 2 * np.sqrt(
            act[seg, 0] ** 2 + act[seg, 1] ** 2 + act[seg, 2] ** 2 - act[seg, 0] * act[seg, 1] - act[seg, 1] * act[
                seg, 2] - act[seg, 0] * act[seg, 2]) / (R * np.sum(act[seg, :]))
        conf["k"] = np.append(conf["k"], k)
        alpha = math.atan2(np.sqrt(3) * (act[seg, 1] + act[seg, 2] - 2 * act[seg, 0]), 3 * (act[seg, 1] - act[seg, 2]))
        conf["alpha"] = np.append(conf["alpha"], alpha)

    return conf


def trans_mat(l, k, alpha):
    if k == 0:
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, l],
            [0, 0, 0, 1]
        ])
    else:
        T = np.array([
            [np.cos(alpha), -np.sin(alpha) * np.cos(l * k), np.sin(alpha) * np.sin(l * k),
             np.sin(alpha) * (1 - np.cos(l * k)) / k],
            [np.sin(alpha), np.cos(alpha) * np.cos(l * k), -np.cos(alpha) * np.sin(l * k),
             -np.cos(alpha) * (1 - np.cos(l * k)) / k],
            [0, np.sin(l * k), np.cos(l * k), np.sin(l * k) / k],
            [0, 0, 0, 1]
        ])
    return T


def rotz(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])


def conf_to_mat(conf):
    confmat = np.zeros((len(conf["l"]), 3))
    for seg in range(len(conf["l"])):
        confmat[seg, 0] = conf["l"][seg]
        confmat[seg, 1] = conf["k"][seg]
        confmat[seg, 2] = conf["alpha"][seg]
    return confmat


def plot_robot(act, radius, conf, poses):
    # Poses of segments' end points
    X, Y, Z = poses[:, 0], poses[:, 1], poses[:, 2]
    normals = np.zeros((poses.shape[0], 3))
    tangents = np.zeros((poses.shape[0], 3))
    binormals = np.zeros((poses.shape[0], 3))
    for i in range(poses.shape[0]):
        rotm = R.from_euler('zyx', poses[i, 3:6], degrees=False)
        rotm = rotm.as_matrix()
        tangents[i, :] = rotm[:, 0]
        binormals[i, :] = rotm[:, 1]
        normals[i, :] = rotm[:, -1]
    U, V, W = normals[:, 0], normals[:, 1], normals[:, 2]
    tU, tV, tW = tangents[:, 0], tangents[:, 1], tangents[:, 2]
    bU, bV, bW = binormals[:, 0], binormals[:, 1], binormals[:, 2]

    Npoints = 100
    # Centerline
    segs = np.zeros((poses.shape[0] - 1, 3, Npoints))
    Told = np.eye(4)
    for seg in range(segs.shape[0]):
        counter = 0
        k = conf["k"][seg]
        alpha = conf["alpha"][seg]
        rotm = Told[0:3, 0:3]
        for l in np.linspace(0, conf["l"][seg], Npoints, endpoint=True):
            if k == 0:
                segs[seg, :, counter] = rotm.dot([
                    0,
                    0,
                    l
                ]) + poses[seg, 0:3]
            else:
                segs[seg, :, counter] = rotm.dot([
                    np.sin(alpha) * (1 - np.cos(l * k)) / k,
                    -np.cos(alpha) * (1 - np.cos(l * k)) / k,
                    np.sin(l * k) / k
                ]) + poses[seg, 0:3]
            counter += 1
        T = trans_mat(conf["l"][seg], conf["k"][seg], conf["alpha"][seg])
        Told = np.matmul(Told, T)

    # Actuators
    actuators = {"bases": np.zeros((poses.shape[0], 3, Npoints))}

    angles = np.linspace(0, 2 * np.pi, Npoints, endpoint=True)
    circ = np.vstack((
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros((len(angles, )))
    ))
    for sec in range(actuators["bases"].shape[0]):
        center = poses[sec, 0:3]
        rotm = R.from_euler('zyx', poses[sec, 3:6], degrees=False)
        rotm = rotm.as_matrix()
        actuators["bases"][sec, :, :] = center.reshape(-1, 1) + np.matmul(rotm, circ)

    actuators["tendons"] = np.zeros((poses.shape[0] - 1, 3, 3, Npoints))
    tend_angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    Told = np.eye(4)
    for seg in range(actuators["tendons"].shape[0]):
        k = conf["k"][seg]
        alpha = conf["alpha"][seg]
        rotm = Told[0:3, 0:3]
        T = trans_mat(conf["l"][seg], k, alpha)
        for tendon in range(actuators["tendons"].shape[1]):
            r_new = (1 / k - radius * np.cos(tend_angles[tendon]))
            k_new = 1 / r_new
            Pinit = Told[0:3, -1] + rotm.dot([
                radius * np.cos(tend_angles[tendon]),
                radius * np.sin(tend_angles[tendon]),
                0
            ])
            counter = 0
            for l in np.linspace(0, act[seg, tendon], Npoints, endpoint=True):
                if k == 0:
                    actuators["tendons"][seg, tendon, :, counter] = rotm.dot([
                        0,
                        0,
                        l
                    ]) + Pinit
                else:
                    actuators["tendons"][seg, tendon, :, counter] = rotm.dot([
                        np.sin(alpha) * (1 - np.cos(l * k_new)) / k_new,
                        -np.cos(alpha) * (1 - np.cos(l * k_new)) / k_new,
                        np.sin(l * k_new) / k_new
                    ]) + Pinit
                counter += 1
        Told = np.matmul(Told, T)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.1, color="r")
    ax.quiver(X, Y, Z, tU, tV, tW, length=0.1, color="g")
    ax.quiver(X, Y, Z, bU, bV, bW, length=0.1, color="b")
    for seg in range(segs.shape[0]):
        ax.plot(segs[seg, 0, :], segs[seg, 1, :], segs[seg, 2, :])
        for tendon in range(actuators["tendons"].shape[1]):
            ax.plot(actuators["tendons"][seg, tendon, 0, :], actuators["tendons"][seg, tendon, 1, :],
                    actuators["tendons"][seg, tendon, 2, :], color="k")
    for base in range(actuators["bases"].shape[0]):
        ax.plot(actuators["bases"][base, 0, :], actuators["bases"][base, 1, :], actuators["bases"][base, 2, :],
                color="k")
    ax.grid(True)
    ax.set_xlim([-.5, .5])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([0, .5])
    ax.set_box_aspect([1.0, 1.0, .5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def flatten_array(arr):
    return ' '.join(map(str, arr.flatten()))


def generateRandomDataset(num_simulation):
    R = .02

    # Definisci il range per ogni coordinata tra 0.1 e 1
    x_range = np.round(np.random.uniform(0.0, 1, num_simulation), 3)
    y_range = np.round(np.random.uniform(0.0, 1, num_simulation), 3)
    z_range = np.round(np.random.uniform(0.0, 1, num_simulation), 3)

    # Genera tutte le possibili combinazioni
    all_combinations = list(itertools.product(x_range, y_range, z_range))
    random.shuffle(all_combinations)

    # Apri il file per scrivere i risultati
    with open('dataset.txt', 'w') as f:
        # Scrivi l'intestazione
        f.write(" actuation (3 values -xyz) poses (6 values -xyz, zyx euler angles) configuration (3 values xyz)  \n")

        # Itera su tutte le combinazioni, recuperandone al massimo 100: DA CAMBIARE IN FASE DI TRAINING DEFINITIVO
        for i, combination in enumerate(all_combinations[:100000]):
            act = np.array([list(combination)])

            conf = fk_dep(act, R)
            poses = np.round(fk_indep(conf), 3)
            configuration = np.round(conf_to_mat(conf), 3)

            # Scrivi i risultati nel file in formato tabellare.
            # Il vettore poses contiene le 3 dimensioni della posizione del punto più i valori degli angoli
            # di Eulero zyx da analizzare in una seconda fase
            f.write(f"{flatten_array(act)} {flatten_array(poses)} {flatten_array(configuration)} \n")


def plotSimulation(actuator):
    actuator = np.array([
        [.1, .1, .156],
        # [.2, .18, .2],
        # [.2, .2, .18]
    ])
    R = .02

    # conf = {}
    # conf["l"] = np.array([1, 1, 1])
    # conf["k"] = np.array([np.pi/2, np.pi/2, np.pi])
    # conf["alpha"] = np.array([0, -np.pi/2, -np.pi/2])

    conf = fk_dep(actuator, R)
    poses = fk_indep(conf)
    plot_robot(actuator, R, conf, poses)

    print("Poses:")
    print(poses)
    print("Configuration:")
    print(conf_to_mat(conf))


def main():
    # num simulazioni in input
    generateRandomDataset(100)
    act = np.array([
        [.1, .1, .156],
        # [.2, .18, .2],
        # [.2, .2, .18]
    ])
    plotSimulation(act)


if __name__ == "__main__":
    main()
