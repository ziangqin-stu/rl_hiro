from environment.maze_env import MazeEnv
from environment.ant import AntEnv


class AntMazeEnv(MazeEnv):
    MODEL_CLASS = AntEnv
