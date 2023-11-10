# 🦖 X—LLM: Simple & Cutting Edge LLM Finetuning

<div align="center">

[![Build](https://github.com/KompleteAI/xllm/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/KompleteAI/xllm/actions/workflows/build.yaml)
[![Github: License](https://img.shields.io/github/license/KompleteAI/xllm.svg?color=63C462)](https://github.com/KompleteAI/xllm/blob/main/LICENSE)
[![Github: Release](https://img.shields.io/github/v/release/kompleteai/xllm.svg)](https://github.com/KompleteAI/xllm/releases)

[![PyPI - Version](https://img.shields.io/pypi/v/xllm.svg?logo=pypi&label=PyPI&logoColor=gold&color=63C462)](https://pypi.org/project/xllm/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/xllm.svg?color=63C462&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/xllm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xllm?logo=python&label=Python&logoColor=gold&color=63C462)](https://pypi.org/project/xllm/)

[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/modelfront/predictor/blob/master/.pre-commit-config.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/KompleteAI/xllm/graph/badge.svg?token=ZOBMDXVW4B)](https://codecov.io/gh/KompleteAI/xllm)

Easy & cutting edge LLM finetuning using the most advanced methods (QLoRA, DeepSpeed, GPTQ, Flash Attention 2, FSDP,
etc)

Developed by [@BobaZooba](https://t.me/BobaZooba) | [CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing) | [LinkedIn](https://www.linkedin.com/in/boriszubarev/) | [bobazooba@gmail.com](mailto:bobazooba@gmail.com)

</div>

# Why you should use X—LLM 🪄

Are you using **Large Language Models (LLMs)** for your work and want to train them more efficiently with advanced methods? Wish to focus on the data and improvements rather than time-consuming coding repetitive for LLM training?

**X—LLM** is your solution. It's a user-friendly library that streamlines training optimization, so you can **focus on enhancing your models and data**. Equipped with **cutting-edge training techniques**, X—LLM is engineered for efficiency by engineers who understand your needs.


**X—LLM** is ideal whether you're **gearing up for production** or need a **fast prototyping tool**.

## Features

- Hassle-free training for Large Language Models
- Seamless integration of new data and data processing
- Effortless expansion of the library
- Speed up your training, while simultaneously reducing model sizes
- Each checkpoint is saved immediately to the 🤗 HuggingFace Hub
- Easy-to-use integration with your existing project
- Customize almost any part of your training with ease
- Track your training progress using `W&B`
- Supported many 🤗 Transformers models
  like `Yi-34B`, `Mistal AI`, `Llama 2`, `Zephyr`, `OpenChat`, `Falcon`, `Phi`, `Qwen`, `MPT` and many more
- Benefit from cutting-edge advancements in LLM training optimization
  - QLoRA and fusing
  - Flash Attention 2
  - Gradient checkpointing
  - bitsandbytes
  - GPTQ (including post-training quantization)
  - DeepSpeed
  - FSDP
  - And many more

# Quickstart 🦖

### Installation

X—LLM is tested on Python 3.8+, PyTorch 2.0.1+ and CUDA 11.8.

```bash
pip install xllm
```

Version which include `deepspeed`, `flash-attn` and `auto-gptq`:

```bash
pip install xllm[train]
```

Default `xllm` version recommended for local development, `xllm[train]` recommended for training.

#### Training recommended environment

CUDA version: `11.8`  
Docker: `huggingface/transformers-pytorch-gpu:latest`

## Fast prototyping ⚡

```python
from xllm import Config
from xllm.datasets import GeneralDataset
from xllm.experiments import Experiment

# 1. Init Config which controls the internal logic of xllm
config = Config(model_name_or_path="facebook/opt-350m")

# 2. Prepare the data
train_data = ["Hello!"] * 100

# 3. Load the data
train_dataset = GeneralDataset.from_list(data=train_data)

# 4. Init Experiment
experiment = Experiment(config=config, train_dataset=train_dataset)

# 5. Build Experiment from Config: init tokenizer and model, apply LoRA and so on
experiment.build()

# 6. Run Experiment (training)
experiment.run()

# 7. [Optional] Fuse LoRA layers
experiment.fuse_lora()

# 8. [Optional] Push fused model (or just LoRA weight) to the HuggingFace Hub
experiment.push_to_hub(repo_id="YOUR_NAME/MODEL_NAME")
```

<details>
  <summary>LoRA</summary>

#### Simple

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta", 
    apply_lora=True,
)
```

#### Advanced

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    apply_lora=True,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout=0.05,
    raw_lora_target_modules="all",  # Names of modules to apply LoRA. A comma-separated string, for example: "k,q,v" or "all".
)
```

</details>

<details>
  <summary>QLoRA</summary>


#### Simple

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    apply_lora=True,
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
)
```

#### Advanced

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    stabilize=True,
    apply_lora=True,
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
```

</details>

<details>
  <summary>Stabilize training</summary>

This technique helps to translate some operations into `fp32` for learning stability. It is also useful to use together with LoRA and GPUs that support `bfloat16`.

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    stabilize=True,
)
```

</details>

<details>
  <summary>Push checkpoints to the HuggingFace Hub</summary>

Before that, you must log in to `Huggingface Hub` or add an `API Token` to the environment variables.

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    push_to_hub=True,
    hub_private_repo=True,
    hub_model_id="BobaZooba/AntModel-7B-XLLM-Demo-LoRA",
    save_steps=25,
)
```

- Checkpoints will be saved locally and in Huggingface Hub each `save_steps`
- If you train a model with `LoRA`, then only `LoRA` weights will be saved

</details>

<details>
  <summary>Report to W&B</summary>

Before that, you must log in to `W&B` or add an `API Token` to the environment variables.

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    report_to_wandb=True,
    wandb_project="xllm-demo",
    wandb_entity="bobazooba",
)
```

</details>

<details>
  <summary>Gradient checkpointing</summary>

This will help to use `less GPU memory` during training, that is, you will be able to learn more than without this technique. The disadvantages of this technique is slowing down the forward step, that is, `slowing down training`.

You will be training larger models (for example 7B in colab), but at the expense of training speed.

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    use_gradient_checkpointing=True,
)
```
</details>

<details>
  <summary>Flash Attention 2</summary>

This speeds up training and GPU memory consumption, but it does not work with all models and GPUs. You also need to install `flash-attn` for this. This can be done using:  

`pip install xllm[train]`

```python
config = Config(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    use_flash_attention_2=True,
)
```
</details>

<details>
  <summary>Combine all</summary>

Features:
- QLoRA
- Gradient checkpointing
- Flash Attention 2
- Stabilize training
- Push checkpoints to HuggingFace Hub
- W&B report

```python
config = Config(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    use_gradient_checkpointing=True,
    stabilize=True,
    use_flash_attention_2=True,
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
    apply_lora=True,
    warmup_steps=1000,
    max_steps=10000,
    logging_steps=1,
    save_steps=1000,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    max_length=2048,

    tokenizer_padding_side="right",  # good for llama2

    push_to_hub=False,
    hub_private_repo=True,
    hub_model_id="BobaZooba/SupaDupaLlama-7B-LoRA",

    report_to_wandb=False,
    wandb_project="xllm-demo",
    wandb_entity="bobazooba",
)
```

</details>

<details>
  <summary>Fuse</summary>

This operation is only for models with a LoRA adapter.

You can explicitly specify to fuse the model after training.

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta", 
    apply_lora=True,
    fuse_after_train=True,
)
```

Even when you are using QLoRa

```python
config = Config(
    model_name_or_path="HuggingFaceH4/zephyr-7b-beta",
    apply_lora=True,
    load_in_4bit=True,
    prepare_model_for_kbit_training=True,
    fuse_after_train=True,
)
```

Or you can fuse the model yourself after training.

```python
experiment.fuse_lora()
```

</details>

<details>
  <summary>DeepSpeed</summary>

`DeepSpeed` is needed for training models on `multiple GPUs`. `DeepSpeed` allows you to `efficiently manage the resources of several GPUs during training`. For example, you can `distribute the gradients and the state of the optimizer to several GPUs`, rather than storing a complete set of gradients and the state of the optimizer on each GPU. Starting training using `DeepSpeed` can only happen from the `command line`.

`train.py`
```python
from xllm.core.config import Config
from xllm.cli.train import cli_run_train

if __name__ == '__main__':
    cli_run_train(config_cls=Config)
```

Run train
```bash
deepspeed --num_gpus=8 train.py --deepspeed_stage 2
```

</details>

### Colab notebooks

| Name      | Comment                                                                            | Link |
|-----------|------------------------------------------------------------------------------------|------|
| X—LLM Prototyping    | In this notebook you will learn the basics of the library                          | <a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/drive/1zsNmJFns1PKZy5VE5p5nsQL-mZF7SwHf?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Llama2 & Mistral AI efficient fine-tuning | 7B model training in colab using QLoRA, bnb int4, gradient checkpointing and X—LLM | <a target="_blank" href="https://colab.research.google.com/github/https://colab.research.google.com/drive/1CNNB_HPhQ8g7piosdehqWlgA30xoLauP?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Production solution 🚀

Run the existing project

Using X—LLM to train a model is simple and involves these few steps:

1. `Download` — Get the data and the model ready by downloading and preparing them. Saves data locally
   to `config.train_local_path_to_data` and `config.eval_local_path_to_data` if you are using eval dataset.
2. `Train` — Use the data prepared in the previous step to train the model.
3. `Fuse` — If you used LoRA during the training, fuse LoRA.
4. `GPTQ Quantization` — Make your model take less space by quantizing it.

Remember, all tasks in X—LLM start from the command line. So, when you're all set to go, launching your full project
will look something like this:

<details>
  <summary>Example how to run your project</summary>

1. Downloading and preparing data and model

    ```bash
    python3 MY_PROJECT/cli/download.py \
      --dataset_key MY_DATASET \
      --model_name_or_path mistralai/Mistral-7B-v0.1 \
      --path_to_env_file ./.env
    ```

2. Run train using DeepSpeed on multiple GPUs

    ```bash
    deepspeed --num_gpus=8 MY_PROJECT/cli/train.py \
      --use_gradient_checkpointing True \
      --deepspeed_stage 2 \
      --stabilize True \
      --model_name_or_path mistralai/Mistral-7B-v0.1 \
      --use_flash_attention_2 False \
      --load_in_4bit True \
      --apply_lora True \
      --raw_lora_target_modules all \
      --per_device_train_batch_size 8 \
      --warmup_steps 1000 \
      --save_total_limit 0 \
      --push_to_hub True \
      --hub_model_id MY_HF_HUB_NAME/LORA_MODEL_NAME \
      --hub_private_repo True \
      --report_to_wandb True \
      --path_to_env_file ./.env
    ```

3. Fuse LoRA
    ```bash
    python3 MY_PROJECT/cli/fuse.py \
      --model_name_or_path mistralai/Mistral-7B-v0.1 \
      --lora_hub_model_id MY_HF_HUB_NAME/LORA_MODEL_NAME \
      --hub_model_id MY_HF_HUB_NAME/MODEL_NAME \
      --hub_private_repo True \
      --force_fp16 True \
      --fused_model_local_path ./fused_model/ \
      --path_to_env_file ./.env
     
4. [Optional] GPTQ quantization of the trained model with fused LoRA
   ```bash
    python3 MY_PROJECT/cli/gptq_quantize.py \
      --model_name_or_path ./fused_model/ \
      --apply_lora False \
      --stabilize False \
      --quantization_max_samples 100000 \
      --quantized_model_path ./quantized_model/ \
      --prepare_model_for_kbit_training False \
      --quantized_hub_model_id MY_HF_HUB_NAME/MODEL_NAME_GPTQ \
      --quantized_hub_private_repo True \
      --low_cpu_mem_usage \
      --path_to_env_file ./.env
    ```

</details>

Right now, the X—LLM library lets you use only the [SODA dataset](https://huggingface.co/datasets/allenai/soda). We've
set it up this way for demo purposes, but we're planning to add more datasets soon. You'll need to figure out how to
download and handle your dataset. Simply put, you take care of your data, and X—LLM handles the rest. We've done it this
way on purpose, to give you plenty of room to get creative and customize to your heart's content.

## Build your own project

To set up your own project using X—LLM, you need to do two things:

1. Implement your dataset (figure out how to download and handle it)
2. Add X—LLM's command-line tools into your project

Once that's done, your project will be good to go, and you can start running the steps you need (like download, train,
and so on).

To get a handle on building your project with X—LLM, check out the materials below.

## Useful materials

- [Docs](https://github.com/KompleteAI/xllm/blob/main/DOCS.md): here, we go into detail about everything the library can
  do
- [Demo project](https://github.com/KompleteAI/xllm-demo): here's a step-by-step example of how to use X—LLM and fit it
  into your own project
- [Template project](https://github.com/KompleteAI/xllm-template): here's a template, a kickoff point you can use for
  your projects
- [How to implement dataset](!link)
- [How to add CLI tools to your project](!link)
- [Demo project](https://github.com/KompleteAI/xllm-demo): here's a step-by-step example of how to use X—LLM and fit it
  into your own project
- [Docs](https://github.com/KompleteAI/xllm/blob/main/DETAILED-GUIDE.md): here, we go into detail about everything the
  library can do

# Config 🔧

The X—LLM library uses a single config setup for all steps like downloading and training. It's designed in a way that
lets you easily understand the available features and what you can adjust. The config has control almost over every
single part of each step. Thanks to the config, you can pick your dataset, set your collator, manage the type of
quantization during training, decide if you want to use lore, if you need to load a checkpoint in HuggingFace Hub, and a
lot more.

Config path: `src.xllm.core.config.Config`

## Useful materials

- [Important config fields for different steps](!link)
- [How do I choose the methods for training?](!link)
- [Detailed description of all config fields](!link)

# Customization options 🛠

You have the flexibility to tweak many aspects of your model's training: data, how data is processed, trainer, config,
how the model is loaded, what happens before and after training, and so much more.

At the very least, you'll need to implement your dataset. For everything else, we've got ready-to-use components.
You can entirely switch out some components like the dataset, collator, trainer, and experiment.
For some components like experiment and config, you have the option to just build on what's already there.

### Useful materials

- [How to implement dataset](!link)
- [How to implement collator](!link)
- [How to implement trainer](!link)
- [How to implement experiment](!link)
- [How to extend config](!link)

# Projects using X—LLM 🏆

Building something cool with [X—LLM](https://github.com/KompleteAI/xllm)? Kindly reach out to me
at [bobazooba@gmail.com](mailto:bobazooba@gmail.com). I'd love to hear from you.

## Hall of Fame

Write to us so that we can add your project.

- [Shurale7B-v1](https://huggingface.co/KompleteAI/Shurale7b-v1)
- [Shurale7B-v1-GPTQ](https://huggingface.co/KompleteAI/Shurale7b-v1-GPTQ)

### Tale Quest

Please support my project for more updates!

<a href="https://www.buymeacoffee.com/talequest" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>


## Badge

Consider adding a badge to your model card.

```bash
[<img src="https://github.com/KompleteAI/xllm/blob/main/static/images/xllm-badge.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/KompleteAI/xllm)
```

[<img src="https://github.com/KompleteAI/xllm/blob/main/static/images/xllm-badge.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/KompleteAI/xllm)

# Testing 🧪

At the moment, we don't have Continuous Integration tests that utilize a GPU. However, we might develop these kinds of
tests in the future. It's important to note, though, that this would require investing time into their development, as
well as funding for machine maintenance.

# Future Work 🔮

- Add more tests
- GPU CI using RunPod
- Add runpod deployment
- Add multipacking
- Fix caching in CI
- Add sequence bucketing
- Add adaptive batch size
- Add more datasets
- Add `tensor_parallel`
- Add auto find max batch size
  - Check max batch size
