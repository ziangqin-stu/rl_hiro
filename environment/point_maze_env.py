from environment.maze_env import MazeEnv
from environment.point import PointEnv


class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
