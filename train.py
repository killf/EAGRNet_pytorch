from solver import Solver
from config import Config

if __name__ == '__main__':
    cfg = Config()

    solver = Solver(cfg)
    solver.train()
