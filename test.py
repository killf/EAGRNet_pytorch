from solver import Solver
from config import Config

if __name__ == '__main__':
    cfg = Config()

    solver = Solver(cfg)
    solver.sample(sample_dir="/data/face/parsing/dataset/testset_210720_aligned", result_folder="result")
