# 🦖 X—LLM: Simple & Cutting Edge LLM Finetuning

# WORK IN PROGRESS

<div align="center">

[![Build](https://github.com/KompleteAI/xllm/actions/workflows/build.yaml/badge.svg?branch=main)](https://github.com/KompleteAI/xllm/actions/workflows/build.yaml)
[![Github: License](https://img.shields.io/github/license/KompleteAI/xllm.svg?color=blue)](https://github.com/KompleteAI/xllm/blob/main/LICENSE)
[![Github: Release](https://img.shields.io/github/v/release/kompleteai/xllm.svg)](https://github.com/KompleteAI/xllm/releases)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/modelfront/predictor/blob/master/.pre-commit-config.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/KompleteAI/xllm/graph/badge.svg?token=ZOBMDXVW4B)](https://codecov.io/gh/KompleteAI/xllm)
[![PyPI - Version](https://img.shields.io/pypi/v/xllm.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/xllm/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/xllmling.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/xllm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xllm.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/xllm/)

Easy & cutting edge LLM finetuning using the most advanced methods (QLoRA, DeepSpeed, GPTQ, Flash Attention 2, FSDP,
etc)

Developed by [@BobaZooba](https://huggingface.co/BobaZooba) | E-mail: [bobazooba@gmail.com](mailto:bobazooba@gmail.com)

I'm open to work &
partnership: [Advising](https://komplete.framer.ai) | [My CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing) | [My LinkedIn](https://www.linkedin.com/in/boriszubarev/)

</div>

# Why you should use X—LLM 🪄

Are you using LLM for your project or research and finding it hard to deal with all the technical details? Wish you
could focus on the data, the task, and the architectural changes, rather than spending so much time trying to figure out
how modern LLM training methods work?

X—LLM is your answer. This easy-to-use library takes care of all training optimization so you can focus on what matters
the most - improving your models and your data. Updated with the latest technology for training models, X—LLM is
designed by engineers, for engineers.

## Features

- Hassle-free training for Large Language Models
- Seamless integration of new data and data processing
- Effortless expansion of the library
- Speed up your training, while simultaneously reducing model sizes
- Each checkpoint is saved immediately to the 🤗 HuggingFace Hub
- Easy-to-use integration with your existing project
- Customize almost any part of your training with ease
- Track your training progress using `W&B`
- Supported many 🤗 Transformers models like `Mistal AI`, `Llama 2`, `Falcon`, `Phi`, `Qwen`, `MPT` and many more
- Benefit from cutting-edge advancements in LLM training optimization
  - QLoRA and fusing
  - Flash Attention 2
  - Gradient checkpointing
  - bitsandbytes
  - GPTQ (including post-training quantization)
  - DeepSpeed
  - FSDP
  - And many more

# Quickstart ⚡

Please check these useful materials:

- [Docs](https://github.com/KompleteAI/xllm/blob/main/DOCS.md): here, we go into detail about everything the library can
  do
- [Demo project](https://github.com/KompleteAI/xllm-demo): here's a step-by-step example of how to use X—LLM and fit it
  into your own project
- [Template project](https://github.com/KompleteAI/xllm-template): here's a template, a kickoff point you can use for
  your projects

## Installation

This repository is tested on Python 3.8+, PyTorch 2.0.1+ and CUDA 11.8.

We recommend [virtual environment](https://docs.python.org/3/library/venv.html) for X—LLM installation. If you're
unfamiliar with Python virtual environments, check out
the [user guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

To install default version run this in command line:

```bash
pip install xllm
```

To install version which include `deepspeed`, `flash-attn` and `auto-gptq` run this in command line:

```bash
pip install xllm[train]
```

Default `xllm` version recommended for local development, `xllm[train]` recommended for training.

## Run

```python
from xllm import HuggingFaceConfig
from xllm.datasets import GeneralDataset
from xllm.experiments import Experiment

config = HuggingFaceConfig(model_name_or_path="facebook/opt-350m")

train_data = ["Hello!"] * 100

train_dataset = GeneralDataset.from_list(data=train_data)
experiment = Experiment(config=config, train_dataset=train_dataset)

experiment.build()

experiment.run()

experiment.push_to_hub(repo_id="YOUR_NAME/MODEL_NAME")
```

### Recommended environment

You will avoid a lot of problems if you use the right environment.

CUDA version: `11.8`  
Docker: `huggingface/transformers-pytorch-gpu:latest`

# Using X—LLM 🦖

X—LLM is made in order to simplify the local development of the project and its launch on the GPU.

## Run the existing project

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

Config path: `src.xllm.core.config.HuggingFaceConfig`

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

Building something cool with [X—LLM](https://github.com/KompleteAI/xllm)? Kindly reach out to us
at [bobazooba@gmail.com](mailto:bobazooba@gmail.com). We'd love to hear from you.

## Hall of fame

Write to us so that we can add your project.

- [Shurale7B-v1](https://huggingface.co/KompleteAI/Shurale7b-v1)

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
- Add multipacking
- Fix caching in CI
- Add sequence bucketing
- Add adaptive batch size
- Add more datasets
- Add `tensor_parallel`
- Add auto find max batch size
  - Check max batch size

# Donuts 🍩

Please support me for more updates!

<a href="https://www.buymeacoffee.com/xllm" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-green.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

# Call to partnership 🛰

**Looking for an expert in modern LLMs?** I've got the experience you need. I'll guide you through every step,
fine-tuning everything from data collection to model training and improvement.

**Why me?** Well, with six years of experience in deep learning R&D projects, I've mastered a range of roles - from
leading a team to rolling up my sleeves as an engineer. I've built and improved products from scratch and I'm keen to do
the same for you.

**Worried about your team?** Don't be. With four years as a lecturer at Russia’s best university, I can equip them with
the skills they need to succeed.

**Want to know more?** Check
out [my CV](https://docs.google.com/document/d/1BhFvIHQ1mpm81P-n2A-lhNac-U2wOGc6F2uS9gKvk88/edit?usp=sharing), [LinkedIn](https://www.linkedin.com/in/boriszubarev/),
and [past projects](https://komplete.framer.ai/cases) for the full scoop.

**Ready to start?** Let's arrange a free intro meeting. I'll outline the resources we'll need to make your project a
success.
[Contact me form](https://komplete.framer.ai/#contact)

If you're an engineer, I'd appreciate it if you could pass
along [my LinkedIn](https://www.linkedin.com/in/boriszubarev/) or [website](https://komplete.framer.ai/) to your
manager.
