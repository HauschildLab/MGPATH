"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
import os
import argparse

import numpy
import pandas

import utils
from eval import Evaluator

def eval(
    seed: int,
    config: dict,
):
    evaluator = Evaluator(seed, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--seed', type=int, default=2024, help='seed number')
    parser.add_argument('--k_start', type=int, default=0, help='start fold')
    parser.add_argument('--k_end', type=int, default=5, help='end fold')
    parser.add_argument('--task', type=str,\
                        help='task name, such as task_tcga_lung_subtyping')
    parser.add_argument("--text_prompt_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,\
            help='path to the config yaml file, such as yaml/tcga_lung.yml')
    parser.add_argument("--checkpoint_dir", type=str, default=None,\
            help='path to the checkpoint directory."\
                            "The checkpoint directory should contain"\
                                    " the model checkpoints for each fold')
    parser.add_argument("--splits_dir", type=str, default=None,\
                            help='path to the splits directory containing'\
                                                ' the splits for each fold')
    parser.add_argument("--output_dir", type=str, default=None)


    args = parser.parse_args()
    utils.create_dir(args.output_dir)

    args.text_prompt = utils.read_text_prompt_file(args.text_prompt_path)
    configs = utils.read_yaml_file(args.config)
    eval(args.seed, configs)
