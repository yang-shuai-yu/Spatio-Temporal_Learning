import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Generate Trajectories')
    parser.add_argument('--gps_path', type=str, default='data/gps/valgps', help='Path to the gps file')
    parser.add_argument('--aligned_path', type=str, default='data/align_data/aligned_gps/aligned_valgps', help='Path to the aligned file')
    parser.add_argument('--roadmap_path', type=str, default='data/align_data/porto_roadmap_edge', help='Path to the roadmap file')
    parser.add_argument('--segment_num', type=int, default=11095, help='Number of segments')
    parser.add_argument('--output_path', type=str, default='result/gen_valgps', help='Path to the output file')
    parser.add_argument('--plot', type=bool, default=True, help='Plot the data')
    parser.add_argument('--gen_num', type=int, default=1000, help='Number of generated trajectories')
    args = parser.parse_args()
    return args