import torch
import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
from utils import *
from datasets import load_dataset, Dataset
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
# from trl import RewardTrainer, RewardConfig
from methods import RewardTrainer, RewardConfig

import pprint
import random
import wandb

parser = argparse.ArgumentParser(description='reward model train')
# model and dataset name
parser.add_argument('--model', type=str, default="qwen1.7b", help='model name')
parser.add_argument('--dataset', type=str, default="hh", help='dataset name, hh, tldr, ufb')
parser.add_argument('--noise_rate', type=float, default=0, help='noise rate')
# loss setting
parser.add_argument('--loss', type=str, default='sigmoid', help='sigmoid, unbiased')
parser.add_argument('--para', type=float, default=0, help='loss parameter')
# train setting
parser.add_argument('--batch_size', type=int, default=128, help='total batch size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=3, help='epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=10, help='max grad norm')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='lr scheduler type')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')

parser.add_argument('--debug', default=False, action="store_true")
args = parser.parse_args()

# uniform format
if args.noise_rate in [0, 0.0]:
    args.noise_rate = 0
if args.para in [0, 0.0]:
    args.para = 0

gpu_nums = torch.cuda.device_count()
print(gpu_nums)
per_device_train_batch_size = int(args.batch_size / gpu_nums / args.gradient_accumulation_steps)


# load model
model_name = return_model_name(args.model)
sft_model_path = f"./models/sft_{args.model}_{args.dataset}"
if not os.path.exists(sft_model_path):
    raise ValueError('No exist sft model, train sft model first')
reward_model = AutoModelForSequenceClassification.from_pretrained(
    sft_model_path,
    num_labels=1,  # Reward model outputs a single score
    trust_remote_code=True,
    dtype=torch.bfloat16
    # device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# dataset process
preference_train_dataset, preference_test_dataset = return_preference_dataset(args.dataset)

# datasets.disable_caching()
random.seed(args.seed)

if args.noise_rate > 0:
    preference_train_dataset = preference_train_dataset.map(lambda x: add_preference_noise(x, noise_rate=args.noise_rate))

if args.debug:
    preference_train_dataset = preference_train_dataset.select(range(1000))
    args.epochs = 1

if args.dataset == 'tldr':
    args.max_length = 512
elif args.dataset == 'hh':
    args.max_length = 512
elif args.dataset == 'ufb':
    args.max_length = 1024


# wandb
if int(os.environ.get("RANK", "0")) == 0:
    wandb.init(
        entity="your name",
        project="your name",

    )

# config
reward_args = RewardConfig(
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim="adamw_torch_fused",
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="no",
    bf16=True,
    report_to="wandb",
    max_grad_norm=args.max_grad_norm,
    seed=args.seed,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    max_length=args.max_length,
    gradient_checkpointing=True,
    loss_type=args.loss
)
if args.loss == 'sigmoid':
    reward_args.label_smoothing = args.para
elif args.loss == 'robust':
    reward_args.label_smoothing = args.para
elif args.loss == 'unbiased':
    reward_args.unbiased_a = args.para
elif args.loss == 'normal_unbiased':
    reward_args.unbiased_a = args.para    

reward_trainer = RewardTrainer(
    model=reward_model,
    processing_class=tokenizer,
    train_dataset=preference_train_dataset,
    eval_dataset=preference_test_dataset,
    args=reward_args,
)

if int(os.environ.get("RANK", "0")) == 0:
    print('--- Training data example ---')
    print('--- chosen response ---\n' + tokenizer.decode(reward_trainer.train_dataset[0]['chosen_input_ids']))
    print('--- rejected response ---\n' + tokenizer.decode(reward_trainer.train_dataset[0]['rejected_input_ids']))

# train and save
reward_trainer.train()
reward_model_path = f"./models/reward_{args.model}_{args.dataset}_{args.loss}_para={args.para}_noise={args.noise_rate}"
reward_trainer.save_model(reward_model_path)

if int(os.environ.get("RANK", "0")) == 0:
    print("reward model train end")