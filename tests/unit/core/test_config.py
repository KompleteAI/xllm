import pytest
import torch

from src.xllm.core.config import HuggingFaceConfig
from src.xllm.core.deepspeed_configs import STAGE_2


def test_check_hub():
    config = HuggingFaceConfig(hub_model_id="llama2", push_to_hub=False)
    config.check_hub()


def test_check_none_deepspeed():
    config = HuggingFaceConfig(deepspeed_stage=None)
    config.check_deepspeed()


def test_check_not_use_flash_attention_2():
    config = HuggingFaceConfig(use_flash_attention_2=False)
    config.check_flash_attention()


def test_check_hub_push_without_repo():
    config = HuggingFaceConfig(hub_model_id=None, push_to_hub=True)
    with pytest.raises(ValueError):
        config.check_hub()


def test_tokenizer_name():
    config = HuggingFaceConfig(tokenizer_name_or_path="llama1000", model_name_or_path="llama2")
    assert config.correct_tokenizer_name_or_path == "llama1000"


def test_tokenizer_name_from_model():
    config = HuggingFaceConfig(tokenizer_name_or_path=None, model_name_or_path="llama2")
    assert config.correct_tokenizer_name_or_path == "llama2"


def test_lora_target_modules():
    config = HuggingFaceConfig(raw_lora_target_modules="q,w,e")
    assert config.lora_target_modules == ["q", "w", "e"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Running tests without GPU")
def test_dtype():
    config = HuggingFaceConfig()
    assert config.dtype == torch.float16


def test_deepspeed():
    config = HuggingFaceConfig(deepspeed_stage=2)
    assert config.deepspeed == STAGE_2


def test_check_deepspeed():
    config = HuggingFaceConfig(deepspeed_stage=0)
    config.check_deepspeed()


def test_check_deepspeed_exception():
    config = HuggingFaceConfig(deepspeed_stage=2)
    with pytest.raises(ImportError):
        config.check_deepspeed()


def test_check_flash_attention():
    config = HuggingFaceConfig(use_flash_attention_2=False)
    config.check_flash_attention()


def test_check_flash_attention_exception():
    config = HuggingFaceConfig(use_flash_attention_2=True)
    with pytest.raises(ImportError):
        config.check_flash_attention()


def test_check_auto_gptq_exception():
    config = HuggingFaceConfig()
    with pytest.raises(ImportError):
        config.check_auto_gptq()
