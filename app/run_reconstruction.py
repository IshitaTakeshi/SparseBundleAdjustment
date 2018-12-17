from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sfm.sba import SBA
from sfm.lm import LevenbergMarquardt
from sfm.camera import CameraParameters

from dataset.sba import load_sba
from app.optimizers import optimize_lm, optimize_scipy

def load_observation(filename):
    # observation.shape == (n_viewpoints, n_3dpoints, 2)
    observation = np.load(filename)
    # observation.shape == (n_3dpoints, n_viewpoints, 2)
    observation = np.swapaxes(observation, 0, 1)
    return observation


def approximate_camera_parameters(image_shape):
    H, W = image_shape[0:2]
    return CameraParameters(focal_length=W, offset=[W / 2, H / 2], skew=0)


def main():
    camera_parameters = CameraParameters(focal_length=1, offset=0)
    observation, n_3dpoints, n_viewpoints = load_observation(sys.argv[1])

    print("n_3dpoints: ", n_3dpoints)
    print("n_viewpoints: ", n_viewpoints)

    sba = SBA(camera_parameters, n_3dpoints, n_viewpoints)

    points3d, poses = optimize_scipy(sba, observations.flatten())
    # points3d, poses = optimize_lm(sba, observation)

    print("points3d.shape", points3d.shape)
    print("poses.shape", poses.shape)
    plot_reconstructed(points3d)


main()
