from src.xllm.core.config import HuggingFaceConfig
from src.xllm.utils.cli import setup_cli


def test_setup_cli(config: HuggingFaceConfig):
    setup_cli(config=config)
