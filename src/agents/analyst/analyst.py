import os
import time
import shutil
import mlflow

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.agents.base_agent import BaseAgent
from src.agents.config import AnalystConfig


class Analyst(BaseAgent):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        config: AnalystConfig,
    ) -> None:
        super().__init__(tokenizer, model, config)

    def run(self, message: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": "You are a data visualisation specialist. You will look at raw data in a CSV format and produce a summary of key features."
            },
            {
                "role": "user",
                "content": message
            },
        ]

        if self.adapter_name is not None:
            self.model.set_adapter(self.adapter_name)
        elif getattr(self.model, "_hf_peft_config_loaded", False):
            self.model.disable_adapters()  # Clear adapters, if at all used

        with mlflow.start_run(run_name="analyst_inference", nested=True):
            run_id = mlflow.active_run().info.run_id

            mlflow.log_param("user_message_preview", message[:200])

            os.makedirs("/app/temp", exist_ok=True)
            prompt_filepath = f"/app/temp/prompt_{run_id}.txt"
            with open(prompt_filepath, "w") as f:
                f.write(message)
            mlflow.log_artifact(prompt_filepath)

            start = time.time()
            response = self._generate(self._tokenize(prompt))
            latency = time.time() - start

            mlflow.log_param("response_preview", response[:200])
            
            mlflow.log_metric("latency_seconds", latency)
            mlflow.log_metric("prompt_length_chars", len(message))
            mlflow.log_metric("response_length_chars", len(response))

            response_filepath = f"/app/temp/response_{run_id}.txt"
            with open(response_filepath, "w") as f:
                f.write(response)
            mlflow.log_artifact(response_filepath)

            # Clean up temp files
            shutil.rmtree("/app/temp")

        return response
        

    def fine_tune(self):
        ...