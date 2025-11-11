"""
LLM Model Configuration and Utilities
"""
from enum import Enum
from dataclasses import dataclass


class ModelName(Enum):
    """Available LLM models"""
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"


@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class ModelFactory:
    """Factory to create model configurations"""

    CONFIGS = {
        "creative": ModelConfig(
            name=ModelName.GPT_4.value,
            temperature=0.9,
            max_tokens=2000
        ),
        "precise": ModelConfig(
            name=ModelName.GPT_35_TURBO.value,
            temperature=0.3,
            max_tokens=1000
        ),
        "balanced": ModelConfig(
            name=ModelName.GPT_4_TURBO.value,
            temperature=0.7,
            max_tokens=2000
        ),
    }

    @staticmethod
    def get_config(style: str) -> ModelConfig:
        """Get model configuration by style"""
        return ModelFactory.CONFIGS.get(
            style,
            ModelFactory.CONFIGS["balanced"]
        )

    @staticmethod
    def list_models() -> list:
        """List available model configurations"""
        return list(ModelFactory.CONFIGS.keys())


if __name__ == "__main__":
    # Test model configurations
    for model_style in ModelFactory.list_models():
        config = ModelFactory.get_config(model_style)
        print(f"{model_style}: {config}")
