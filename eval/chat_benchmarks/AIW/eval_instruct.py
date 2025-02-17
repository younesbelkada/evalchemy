import json
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import re

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv

from eval.task import BaseBenchmark


class AIWBenchmark(BaseBenchmark):
    """
    AIW Benchmark for evaluating the math reasoning of LLMs.
    Link: https://github.com/LAION-AI/AIW

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AIW/data/aiw_data.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        n_trials: int = 10,
    ):
        """
        Initialize AIW benchmark.

        Args:
            data_file: File containing the AIW dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.data_file = data_file
        self.debug = debug
        self.max_new_tokens = 32768  # set higher to avoid truncation for reasoning models
        self.seed = seed
        self.n_trials = n_trials

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()
        # Prepare instances for model

        all_instances = []

        for idx, example in enumerate(examples):
            messages = [
                {"role": "user", "content": example["prompt"]},
            ]

            try:
                templated_messages = model.apply_chat_template(messages)
            except Exception as e:
                print(f"Error applying chat template: {e}")
                breakpoint()
                templated_messages = messages

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 1.0,
                            "seed": self.seed,
                        },
                    ),
                    idx,
                )
            )
        
        # Generate model responses
        self.logger.info("Generating responses for AIW...")
        outputs = self.compute(model, all_instances)
        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = sum(is_equiv(str(example["right_answer"]), example["model_answer"]) for example in examples)

        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": solved / total,
            }
        )

        return results


    def load_questions(self) -> List[Dict[str, str]]:
        """Load AIW questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = json.load(f)
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution, which is expected to be in the format of \boxed{answer}.

        Uses the same logic as hendrycks_math.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \boxed.
        """
        try:
            model_response = output.replace('\n', ' ')
            return re.findall(r'answer:.*?(\d+)', model_response.lower())[0]
        except:
            try:
                return re.findall(r'has.*?(\d+)', model_response.lower())[0]
            except:
                print(f'Error parsing model response for model')
                return ""
