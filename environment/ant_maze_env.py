from environments.maze_env import MazeEnv
from environments.ant import AntEnv


class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv
