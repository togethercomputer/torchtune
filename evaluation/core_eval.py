import sys
sys.path.append('/home/imodoranu/workplace/projects/torchtune/evaluation')
import os
import time
import json
import argparse
import wandb
from evaluation import evaluate_model

def convert_seconds_to_hours_minutes_seconds(ss):
    ss = int(ss)
    hh = ss // 3600
    ss %= 3600
    mm = ss // 60
    ss %= 60
    return f'{hh}h {mm}m {ss}s'

def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_args',       type=str, default=None, required=False,  help='The root folder that contains the experiment')
    parser.add_argument('--root',           type=str, default=None, required=False, help='The root folder that contains the experiment')
    parser.add_argument('--wandb_entity',   type=str, default=None, required=False, help='The wandb project inside "ist" owner.')
    parser.add_argument('--wandb_project',  type=str, default=None, required=False, help='The wandb project inside "ist" owner.')
    parser.add_argument('--wandb_group',    type=str, default=None, required=False, help='The wandb group in the project.')
    parser.add_argument('--wandb_job_type', type=str, default=None, required=False, help='The wandb job type')
    parser.add_argument('--wandb_name',     type=str, default=None, required=False, help='The name for the experiment in wandb runs')
    parser.add_argument('--wandb_run_id',   type=str, default=None, required=False, help='The run_id of the wandb job.')
    parser.add_argument('--task',           type=str, default=None, required=False, help='The task to evaluate the model on')
    parser.add_argument('--batch_size',     type=int, default=None, required=False, help='The batch size to use for evaluation.')
    parser.add_argument('--out_folder',     type=str, default=None, required=False, help='A placeholder for GridSearcher')
    args = parser.parse_args()

    # --arg will contain all other arguments below it, but separated by a space
    if args.csv_args is not None:
        for item in args.csv_args.split(','):
            split = item.split('=')
            key = split[0]
            value = '='.join(split[1:])
            setattr(args, key, value)
    return args

def model_eval(args):
    print(f'\nRunning eval on task {args.task}\n')

    ft_model_path = os.path.join(
        args.root,
        args.wandb_project,
        args.wandb_group,
        args.wandb_job_type,
        args.wandb_name,
        # 'huggingface'
        'epoch_0',
    )

    print(f'{ft_model_path}')

    if not os.path.isdir(ft_model_path):
        print('Path not found, exiting')
        return None

    print('Evaluating model...')

    eval_metrics_file = os.path.join(ft_model_path, 'eval', f'{args.task}_eval_metrics.json')

    print(f'\n\n{eval_metrics_file=}\n\n')

    if os.path.isfile(eval_metrics_file):
        print(f'Evaluation metrics file found: {eval_metrics_file}')
        with open(eval_metrics_file, 'r') as file:
            data = json.load(file)
            if args.task == 'gsm8k':
                eval_data = {
                    # f'eval/{args.task}/acc': data['results']['gsm8k']['acc'],
                    # f'eval/{args.task}/acc-std': data['results']['gsm8k']['acc_stderr'],
                    f'eval/{args.task}/strict': data[args.task]['exact_match,strict-match'],
                    f'eval/{args.task}/strict-std': data[args.task]['exact_match_stderr,strict-match'],
                    f'eval/{args.task}/flexible': data[args.task]['exact_match,flexible-match'],
                    f'eval/{args.task}/flexible-std': data[args.task]['exact_match_stderr,flexible-match'],
                    f'eval/{args.task}/elapsed': 'unknown'
                }
            elif args.task in ['viggo', 'sql']:
                eval_data = {
                    f'eval/{args.task}/acc': data['acc'],
                    f'eval/{args.task}/acc-std': data['acc_std'],
                    f'eval/{args.task}/elapsed': 'unknown'
                }
            elif args.task == 'math':
                print(f'Evaluation for MATH dataset:')
                print(data)
            else:
                raise NotImplementedError(f'Reading eval data for task {args.task} is not implemented yet!')
    else:
        print(f'Evaluation metrics file not found: {eval_metrics_file}')
        time_eval_start = time.time()
        # acc, acc_std = evaluate_model(
        strict, strict_std, flexible, flexible_std = evaluate_model(
            ft_model_path=ft_model_path,
            task=args.task,
            batch_size=args.batch_size)
        time_eval_end = time.time()

        eval_data = {
            # f'eval/{args.task}/acc': acc,
            # f'eval/{args.task}/acc-std': acc_std,
            f'eval/{args.task}/strict': strict,
            f'eval/{args.task}/strict-std': strict_std,
            f'eval/{args.task}/flexible': flexible,
            f'eval/{args.task}/flexible-std': flexible_std,
            f'eval/{args.task}/elapsed': convert_seconds_to_hours_minutes_seconds(time_eval_end - time_eval_start)
        }
    # end if-else
    print(f'{eval_data=}')
    return eval_data

def upload_data_to_run(eval_data, args):
    if eval_data is not None:
        api = wandb.Api()
        run = api.run(f'{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}')
        for k, v in eval_data.items():
            if k not in run.summary:
                run.summary[k] = v
        run.summary.update()

        ### initial version that increases Runtime
        # run = wandb.init(entity='ist', project=args.wandb_project, id=args.wandb_run_id, resume='allow')
        # run.log(data)
        # run.finish()

def main():
    args = get_arg_parse()
    eval_data = model_eval(args)
    upload_data_to_run(eval_data, args)


if __name__ == '__main__':
    main()
