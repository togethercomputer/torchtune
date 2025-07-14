import argparse
import wandb
import os
from gridsearcher import GridSearcher, GSExe, GSKeyValSep


def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, required=True)
    parser.add_argument('--exact', type=int, required=False, default=0, choices=[0, 1])
    parser.add_argument('--shots', type=int, required=False, default=0)
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

    BATCH_SIZE = 64 # if llama_version == 2 else 32
    # ROOT = '/mnt/beegfs/alistgrp/imodoran/results'
    ROOT = '/data/imodoranu/results/'

    MODELS = [
        # (2, 7),
        # (3, 8),
        # (3.1, 8),
        (3.2, 1),
    ] # (version, size)

    TASKS = [
        # 'hendrycks_math',
        'gsm8k',
        # 'viggo',
        # 'sql',
    ]

    wandb_entity = 'ionutmodo'

    core_eval_args = []

    for llama_version, llama_size in MODELS:
        for task in TASKS:
            # wandb_project = f'ionut_clr-adamw_llama{llama_version}-{llama_size}b_{task}'
            # wandb_project = f'imodoranu_frugal-micro-adam_llama{llama_version}-{llama_size}b_{task}'
            # wandb_project = f'ionut_torchtune_llama{llama_version}-{llama_size}B_full_ft_multi_gpu'
            wandb_project = f'ionut_torchtune_llama{llama_version}-{llama_size}B_full_ft_multi_gpu_coswmp'

            api = wandb.Api()
            runs = api.runs(f'{wandb_entity}/{wandb_project}')

            for run in runs:
                # run_str = f'{run.group}/{run.job_type}'
                if run.state != 'finished':
                    # print(f'skipping run {run_str} because it is not finished')
                    continue
                if 'eval/gsm8k/flexible' in run.summary:
                    # print(f'skipping run {run_str} because model was already evaluated')
                    continue
                if not any([(gf == run.group) if exact_filter else (gf in run.group) for gf in group_filters]):
                    # print(f'skipping run {run_str} because group does not contain any string from "{group_filters}"')
                    continue

                core_eval_args.append(
                    ','.join([
                        f'root={ROOT}',
                        f'wandb_entity={wandb_entity}',
                        f'wandb_project={wandb_project}',
                        f'wandb_group={run.group}',
                        f'wandb_job_type={run.job_type}',
                        f'wandb_name={run.name}',
                        f'wandb_run_id={run.id}',
                        f'shots={args.shots}',
                        f'task={task}',
                        f'batch_size={BATCH_SIZE}',
                    ])
                )
                # break
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
    python3 main_scrape_eval_allocated.py --exact=1 --filter=adamw_bs=2_gas=32_cg=1_split=train_1M
    python3 main_scrape_eval_allocated.py --exact=1 --filter=frugal-micro-adam-sgd=1_bs=2_gas=32_cg=1_split=train_1M_ng=10_k=0.02_norm=none
"""
