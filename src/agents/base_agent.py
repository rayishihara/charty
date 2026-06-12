from abc import ABC, abstractmethod
from src.agents.config import BaseAgentConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor

class BaseAgent(ABC):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        config: BaseAgentConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.adapter_name = config.adapter_name

        self.tokenize_config = config.tokenize_config
        self.generate_config = config.generate_config

    def _tokenize(
        self,
        prompt: list[dict[str, str]]
    ) -> dict[str, Tensor]:
        return self.tokenizer.apply_chat_template(
            prompt,
            **self.tokenize_config
        )

    def _generate(
        self,
        tokenized_prompt: dict[str, Tensor]
    ) -> str:       
        input_length = tokenized_prompt["input_ids"].shape[-1]
        response = self.model.generate(
            **tokenized_prompt,
            **self.generate_config,
        )
        generated_tokens = response[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @abstractmethod
    def run(self) -> str:
        raise NotImplementedError("Override this method")