import os
import torch
import argparse
from utils import *
from datasets import load_dataset, Dataset
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# from trl import SFTTrainer, DPOTrainer, DPOConfig
from methods import DPOTrainer, DPOConfig
import pprint
import random
import wandb


parser = argparse.ArgumentParser(description='dpo-base train')
# model and dataset name
parser.add_argument('--model', type=str, default="qwen1.7b", help='model name')
parser.add_argument('--dataset', type=str, default="shp", help='dataset name, hh, tldr, orca, ufb, shp')
parser.add_argument('--noise_rate', type=float, default=0, help='noise rate')
# loss setting
parser.add_argument('--loss', type=str, default='sigmoid', help='the loss functions, sigmoid, robust, unbiased...')
parser.add_argument('--para', type=float, default=0, help='loss parameter')
# parser.add_argument('--beta', type=float, default=0.1, help='loss parameter')
# train setting
parser.add_argument('--batch_size', type=int, default=128, help='total batch size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient accumulation steps')
parser.add_argument('--precompute_ref_batch_size_multiple', type=int, default=4, help='')
parser.add_argument('--epochs', type=int, default=3, help='epochs')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=10, help='max grad norm')
# parser.add_argument('--max_length', type=int, default=1024, help='max length')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='lr scheduler type')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
# parser.add_argument('--optim', type=str, default="adamw_bnb_8bit", help='optim')
parser.add_argument('--debug', default=False, action="store_true")
args = parser.parse_args()


# uniform format
if args.noise_rate in [0, 0.0]:
    args.noise_rate = 0
if args.para in [0, 0.0]:
    args.para = 0

gpu_nums = torch.cuda.device_count() 
# print(gpu_nums)
per_device_train_batch_size = int(args.batch_size / gpu_nums / args.gradient_accumulation_steps)

dataset_name = return_dataset_name(args.dataset)
model_name = return_model_name(args.model)


sft_model_path = f"./models/sft_{args.model}_{args.dataset}"
if not os.path.exists(sft_model_path):
    raise ValueError('No exist sft model, train sft model first')

dpo_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True, dtype=torch.bfloat16)
ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# dataset process
preference_train_dataset, preference_test_dataset = return_preference_dataset(args.dataset)
if args.debug:
    preference_train_dataset = preference_train_dataset.select(range(1000))

# datasets.disable_caching()
random.seed(args.seed)

if args.noise_rate > 0:
    preference_train_dataset = preference_train_dataset.map(lambda x: add_preference_noise(x, noise_rate=args.noise_rate))

if args.dataset == 'tldr':
    args.max_length = 512
    args.beta = 0.5
elif args.dataset == 'hh':
    args.max_length = 512
    args.beta = 0.1
elif args.dataset == 'ufb':
    args.max_length = 1024
    args.beta = 0.1


# wandb
if int(os.environ.get("RANK", "0")) == 0:
    wandb.init(
        entity="your name",
        project="your name",
    )

# config and train
dpo_args = DPOConfig(
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim=args.optim,
    optim="adamw_torch_fused",
    num_train_epochs=args.epochs,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    eval_on_start=True,
    max_length=args.max_length,
    save_strategy="no",
    bf16=True,
    report_to="wandb",
    max_grad_norm=args.max_grad_norm,
    loss_type=args.loss,
    learning_rate=args.lr,
    seed=args.seed,
    beta=args.beta,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    precompute_ref_log_probs=True,
    precompute_ref_batch_size=per_device_train_batch_size*args.precompute_ref_batch_size_multiple,
    gradient_checkpointing=True
)
if args.loss == 'sigmoid':
    dpo_args.label_smoothing = args.para
elif args.loss == 'robust':
    dpo_args.label_smoothing = args.para
elif args.loss == 'drdpo':
    dpo_args.drdpo_beta = args.para
elif args.loss == 'unbiased':
    dpo_args.unbiased_a = args.para
elif args.loss == 'normal_unbiased':
    dpo_args.unbiased_a = args.para
    
dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=ref_model,
    processing_class=tokenizer,
    train_dataset=preference_train_dataset,
    eval_dataset=preference_test_dataset,
    args=dpo_args,
)
if int(os.environ.get("RANK", "0")) == 0:
    print('--- prompt example ---\n' + tokenizer.decode(dpo_trainer.train_dataset[0]['prompt_input_ids']))
    print('--- chosen response ---\n' + tokenizer.decode(dpo_trainer.train_dataset[0]['chosen_input_ids']))
    print('--- rejected response ---\n' + tokenizer.decode(dpo_trainer.train_dataset[0]['rejected_input_ids']))


dpo_model_path = f"./models/dpo_{args.model}_{args.dataset}_{args.loss}_para={args.para}_noise={args.noise_rate}"
dpo_trainer.train()
dpo_trainer.save_model(dpo_model_path)
if int(os.environ.get("RANK", "0")) == 0:
    print("train end")