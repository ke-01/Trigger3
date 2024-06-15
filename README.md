# Implementation of Trigger<sup>3</sup>
This is the implementation of the paper "Trigger<sup>3</sup>: Refining Query Correction via Adaptive Model Selector" based on PyTorch.


## Dataset
Check folder `dataset` for details.

## Satisfy the requirements
Because the base model environment conflicts, you need to check it according to the requirements in the corresponding folder.

```
# e.g. GECToR
conda create -n GECToR python=3.8
conda activate GECToR
pip install -r requirements_gector.txt
```

## Train and evaluate our framework:

### Traditional correction models
```bash
# GECToR
sh model/GECToR/train_gector.sh

# BART
sh model/BART/train_bart.sh

# mT5
sh model/mT5/train_mt5.sh
```

Check folder `model/GECToR, model/BART, model/mT5` for details.

### LLMs
The fine-tuing process of LLMs is based on the open-sourced [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

The Qwen1.5-7b-chat and Baichuan2-7b-chat can be downloaded from [huggingface](https://huggingface.co/models).

```bash
# Qwen
sh model/LLaMA-Factory/sft_qwen.sh

# Baichuan
sh model/LLaMA-Factory/sft_baichuan.sh
```

Check folder `model/LLaMA-Factory` for details.

### Trigger

```bash
# train dataset construct
sh ChERRANT/qq_train_trigger_char.sh
sh ChERRANT/qq_train_trigger_data_construct.sh

# train
sh model/Trigger/train_trigger.sh
```


Check folder `model/Trigger` for details.

### Inference

```bash
sh model/Trigger/Trigger3.sh
```

Check folder `model/Trigger` for details.

### Test

```bash
sh ChERRANT/qq_test.sh
```

## Reference Repositories
[MuCGEC](https://github.com/HillZhang1999/MuCGEC)

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)


### Environments
We conducted the experiments based on the following environments:
* CUDA Version: 11.8
* OS: Ubuntu 18.04.4 LTS
* GPU: The NVIDIA Tesla V100 GPUs
* CPU: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz