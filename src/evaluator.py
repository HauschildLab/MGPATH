"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
import argparse

import numpy
import pandas

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--data_dir_s', type=str, default=None,\
                    help='data directory for the low resolution embedding')
    parser.add_argument('--data_dir_l', type=str, default=None,\
                    help='data directory for the high resolution embedding')
    parser.add_argument('--data_graph_dir_s', type=str, default=None,\
                    help='data directory for the low spatial graph features')
    parser.add_argument('--data_graph_dir_l', type=str, default=None,\
                    help='data directory for the high spatial graph features')
    parser.add_argument('--k_start', type=int, default=0, help='start fold')
    parser.add_argument('--k_end', type=int, default=5, help='end fold')
    parser.add_argument('--task', type=str,\
                        help='task name, such as task_tcga_lung_subtyping')
    parser.add_argument("--text_prompt_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    args.text_prompt = numpy.array(pandas.read_csv(args.text_prompt_path,\
                                                    header=None)).squeeze()