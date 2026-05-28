import random
import numpy as np
from datasets import load_dataset, Dataset
from trl import extract_prompt
import logging
import datasets


def dpo_tldr(data_format1):

  data_format2 = {
      "prompt": [{"role": "user", "content": data_format1["prompt"]}],
      "chosen": [{"role": "assistant", "content": data_format1["chosen"]}],
      "rejected": [{"role": "assistant", "content": data_format1["rejected"]}],
  }
  return data_format2

def add_preference_noise(example, noise_rate=0):
    """
    以一定概率交换 chosen 和 rejected, 模拟人工标注噪声。
    """
    if random.random() < noise_rate:
        # 交换 chosen 和 rejected
        example['chosen'], example['rejected'] = example['rejected'], example['chosen']
    return example


def get_logger(filename):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=filename, format=head, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def return_dataset_name(dataset):
    if dataset == 'hh':
        dataset_name = "trl-lib/hh-rlhf-helpful-base"
    elif dataset == 'descriptiveness':
        dataset_name = "trl-lib/lm-human-preferences-descriptiveness"
    elif dataset == 'sentiment':
        dataset_name = "trl-lib/lm-human-preferences-sentiment"
    elif dataset == 'tldr':
        dataset_name = "trl-lib/tldr-preference"
    elif dataset == 'ufb':
        dataset_name = "trl-lib/ultrafeedback_binarized"
    elif dataset == 'orca':
        dataset_name = "Intel/orca_dpo_pairs"
    elif dataset == 'ultrachet':
        dataset_name = 'HuggingFaceH4/ultrachat_200k'
    elif dataset == 'capybara':
        dataset_name = "trl-lib/Capybara"
    elif dataset == 'shp':
        dataset_name = "stanfordnlp/SHP"
    return dataset_name

def return_model_name(model):

    if model == 'qwen0.6b':
        model_name = "Qwen/Qwen3-0.6B-Base"
        tokenizer_name = "Qwen/Qwen3-0.6B-Base"
    if model == 'qwen1.7b':
        model_name = "Qwen/Qwen3-1.7B-Base"
        tokenizer_name = "Qwen/Qwen3-1.7B-Base"
    if model == 'qwen4b':
        model_name = "Qwen/Qwen3-4B-Base"
        tokenizer_name = "Qwen/Qwen3-4B-Base"
    if model == 'qwen8b':
        model_name = "Qwen/Qwen3-8B-Base"
        tokenizer_name = "Qwen/Qwen3-8B-Base"

    if model == 'llama3b':
        model_name = 'unsloth/Llama-3.2-3B'
        tokenizer_name = 'unsloth/Llama-3.2-3B-Instruct'
    if model == 'llama8b':
        model_name = 'unsloth/Meta-Llama-3.1-8B'
        tokenizer_name = 'unsloth/Meta-Llama-3.1-8B-Instruct'
    return model_name, tokenizer_name

def return_preference_dataset(dataset):
    dataset_name = return_dataset_name(dataset)

 
    if dataset == 'tldr':
        preference_train_dataset = load_dataset(dataset_name, split="train")
        # preference_train_dataset = preference_train_dataset.map(dpo_tldr)

        preference_test_dataset = load_dataset(dataset_name, split="validation")
        df = preference_test_dataset.to_pandas()
        df_deduplicated = df.drop_duplicates(subset=['prompt'], keep='first')
        preference_test_dataset = Dataset.from_pandas(df_deduplicated, preserve_index=False) 
        preference_test_dataset = preference_test_dataset.select(range(1000))
        # preference_test_dataset = preference_test_dataset.map(dpo_tldr)
    elif dataset == 'hh':
        preference_train_dataset = load_dataset(dataset_name, split="train")
        preference_test_dataset = load_dataset(dataset_name, split="test[:1000]")
    elif dataset == 'ufb':
        preference_train_dataset = load_dataset(dataset_name, split="train")
        preference_train_dataset = preference_train_dataset.map(extract_prompt)

        preference_test_dataset = load_dataset(dataset_name, split="test")
        preference_test_dataset = preference_test_dataset.map(extract_prompt)
    elif dataset == "capybara":
        preference_train_dataset = load_dataset(dataset_name, split="train")
        preference_test_dataset = load_dataset(dataset_name, split="test")
   

    return preference_train_dataset, preference_test_dataset




















if __name__ == "__main__":

    # dataset_name = "trl-lib/hh-rlhf-helpful-base"
    # dpo_dataset = load_dataset(dataset_name, split="train")
    # sft_dataset = dpo_dataset.map(sft_hh, remove_columns=["prompt", "chosen", "rejected"])
    # print(sft_dataset[0])

    # dataset = load_dataset(dataset_name, split="test")
    # dataset = dataset.map(concat_prompt_hh, remove_columns=["chosen", "rejected"])
    # print(dataset[0])

    dataset_name = "trl-lib/ultrafeedback_binarized"
    dpo_dataset = load_dataset(dataset_name, split="train")
    # dpo_dataset = dpo_dataset.map(dpo_hh)
    print(dpo_dataset[0])
  
