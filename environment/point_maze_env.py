from environments.maze_env import MazeEnv
from environments.point import PointEnv


class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
