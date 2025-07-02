from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help="HuggingFace dataset name, such as 'nvidia/OpenMathInstruct-2'")
args = parser.parse_args()

dataset = load_dataset(args.dataset)