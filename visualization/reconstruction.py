from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_reconstructed(*points3d):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, P in enumerate(points3d):
        ax.plot(P[:, 0], P[:, 1], P[:, 2], '.', label=str(i))
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(elev=135, azim=90)
        ax.legend()
    ax.set_aspect('equal', 'datalim')
    plt.show()

