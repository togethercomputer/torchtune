import argparse
import wandb
import os
from gridsearcher import GridSearcher, GSExe, GSKeyValSep


def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, required=True)
    parser.add_argument('--exact', type=int, required=False, default=0, choices=[0, 1])
    # parser.add_argument('--exclude', default='', type=str, required=False) # 'gpu266,gpu275,gpu276,gpu277'
    # parser.add_argument('--cpus_per_task', default=10, type=int, required=False)
    # parser.add_argument('--time', default='10-00:00:00', type=str, required=False)
    # parser.add_argument('--mem', default='100G', type=str, required=False)
    # parser.add_argument('--partition', default='gpu100', type=str, required=False)
    # parser.add_argument('--gres', default='gpu:H100:1', type=str, required=False)
    args = parser.parse_args()
    return args

def main():
    """
        This script iterates through all wandb runs in a project.
        For each run, we extract information, such as project name, group, job_type, name and run id.
        This information is sent to the SBATCH script that actually runs core_eval.py to evaluate the model (trained in this wandb run).
        The model was saved at the location ROOT / wandb_project / wandb_group / wandb_job_type / wandb_name / huggingface

        To use this script, run **pip3 install gridsearcher**
    """
    args = get_arg_parse()
    group_filters = args.filter.split(',')
    exact_filter = bool(args.exact)

    gpus = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]

    # llama_version = 2; llama_size = 7
    # llama_version = 3; llama_size = 8

    # task = 'gsm8k'
    # task = 'viggo'
    # task = 'sql'
    # task = 'math'

    BATCH_SIZE = 64 # if llama_version == 2 else 32
    # ROOT = '/mnt/beegfs/alistgrp/imodoran/results'
    ROOT = '/data/imodoranu/results/'

    MODELS = [
        # (2, 7),
        # (3, 8),
        (3.2, 1),
    ] # (version, size)

    TASKS = [
        # 'math',
        'gsm8k',
        # 'viggo',
        # 'sql',
    ]

    core_eval_args = []

    for llama_version, llama_size in MODELS:
        for task in TASKS:
            # wandb_project = f'ionut_clr-adamw_llama{llama_version}-{llama_size}b_{task}'
            # wandb_project = f'imodoranu_frugal-micro-adam_llama{llama_version}-{llama_size}b_{task}'
            wandb_project = f'ionut_torchtune_llama{llama_version}-{llama_size}B_full_ft_multi_gpu'

            api = wandb.Api()
            runs = api.runs(f'ist/{wandb_project}')

            for run in runs:
                # run_str = f'{run.group}/{run.job_type}'
                if run.state != 'finished':
                    # print(f'skipping run {run_str} because it is not finished')
                    continue
                if 'eval/acc' in run.summary:
                    # print(f'skipping run {run_str} because model was already trained')
                    continue
                if not any([(gf == run.group) if exact_filter else (gf in run.group) for gf in group_filters]):
                    # print(f'skipping run {run_str} because group does not contain any string from "{group_filters}"')
                    continue

                core_eval_args.append(
                    ','.join([
                        f'root={ROOT}',
                        f'wandb_project={wandb_project}',
                        f'wandb_group={run.group}',
                        f'wandb_job_type={run.job_type}',
                        f'wandb_name={run.name}',
                        f'wandb_run_id={run.id}',
                        f'task={task}',
                        f'batch_size={BATCH_SIZE}',
                    ])
                )
            # end for run
        # end for task
    # end for (version, size)

    gs = GridSearcher(
        # script=f'/nfs/scistore19/alistgrp/imodoran/workplace/M-FAC_extensions/llm-foundry/scripts/train/evaluation/core_eval.py',
        script=f'/home/imodoranu/workplace/projects/torchtune/evaluation/core_eval.py',
        exe=GSExe.PYTHON,
        key_value_separator=GSKeyValSep.EQUAL,
        use_dashes=True,
        defaults=dict())

    state_finished_file = './gridsearcher_output/state.finished'
    if os.path.isfile(state_finished_file):
        os.remove(state_finished_file)

    gs.run(
        launch_blocking=0,
        torchrun=0,
        scheduling=dict(
            distributed_training=False,
            max_jobs_per_gpu=1,
            gpus=gpus,
            params_values={
                'csv_args': core_eval_args
            }
        ),
        param_name_for_exp_root_folder='out_folder',
        exp_folder='./gridsearcher_output')
    # for i, core_arg in enumerate(core_eval_args):
    #     print(f'{i}) {core_arg}')

if __name__ == '__main__':
    main()

"""
    Script usage:
    py main_scrape_eval.py --filter clr_adamw
"""
