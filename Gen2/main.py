import numpy as np
import matplotlib.pyplot as plt
import pickle, os, argparse

from config import get_args
from roadmap_cal import Calculator, Server

if __name__ == '__main__':
    args = get_args()
    print(args)

    calculator = Calculator(args.plot, args.segment_num)
    calculator.load_data(args.gps_path, args.aligned_path, args.roadmap_path)
    matrix, freq, length = calculator.calculate_all()
    # print(matrix, freq, length)
    # print(length)

    generator = Server(args.roadmap_path, args.output_path,
                        args.segment_num, args.plot)
    generator.load(matrix, freq, length)
    generator.trajs_generate(args.gen_num)
    generator.save_trajs()
    print('Generation finished!')
