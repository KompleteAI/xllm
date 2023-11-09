# Copyright 2023 Komplete AI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type

from transformers import HfArgumentParser

from ..core.config import Config
from ..quantization.quantizer import Quantizer
from ..run.quantize import quantize
from ..utils.cli import setup_cli


def cli_run_quantize(
    config_cls: Type[Config] = Config,
) -> Quantizer:
    parser = HfArgumentParser(config_cls)
    config = parser.parse_args_into_dataclasses()[0]
    setup_cli(config=config, logger_path="./xllm_gptq_quantize.log")
    quantizer = quantize(config=config)
    return quantizer


if __name__ == "__main__":
    cli_run_quantize(config_cls=Config)
