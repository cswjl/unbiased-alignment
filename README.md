# Unbiased Alignment for Large Language Models with Noisy Preferences

This repository is the official pytorch code of the **Unbiased Alignment** [ICML2026] 


## How to use

Major version:\
trl Version: 0.24.0\
torch Version: 2.8.0+cu128

Step 1: train SFT model:
```console
python3 train_sft.py 
```

Step 2: train DPO model or Reward model:
```console
python3 train_dpo.py 
```
or 
```console
python3 train_reward.py 
```
In addition, you can easily integrate the loss we proposed into your code.
```console
if self.args.loss_type == 'unbiased':
    prob = (torch.exp(rewards_chosen - rewards_rejected) + self.args.unbiased_a) / (torch.exp(rewards_chosen - rewards_rejected) + 1)
    loss = - torch.log(prob).mean()

elif self.args.loss_type == 'normal_unbiased':
    sqrt_a = self.args.unbiased_a ** 0.5
    normal = (1 + sqrt_a) / (1 - sqrt_a)
    prob = (torch.exp(rewards_chosen - rewards_rejected) + self.args.unbiased_a) / (torch.exp(rewards_chosen - rewards_rejected) + 1)
    loss = - torch.log(prob).mean() * normal
```
If you have any question, you can contact cswjl@stu.hit.edu.cn