import json
import logging
import os
from typing import Any, Dict, Generator, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

PROMPT = """Problem: {problem}\nAnswer:"""


class AMC23Benchmark(BaseBenchmark):
    """
    AIME24 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/aime24
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AIME24/data/aime24.json",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize AIME24 benchmark.

        Args:
            data_dir: Directory containing the AIME24 dataset (id, problem, reference_solution, expected_answer, source)
            max_tokens: Maximum number of tokens for generation
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.data_file = data_file
        self.debug = debug

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.read_test_examples(self.data_file)

        all_instances = []
        for idx, example in enumerate(examples):
            messages = [{"role": "user", "content": PROMPT.format(problem=example["question"])}]
            templated_messages = model.apply_chat_template(messages)
            all_instances.append(Instance("generate_until", example, (templated_messages, {"do_sample": False}), idx))

        self.logger.info("Generating responses for AIME24...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated solution completions.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = sum(is_equiv(str(example["answer"]), example["model_answer"]) for example in examples)

        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": solved / total,
            }
        )

        return results

    def read_test_examples(self, data_path: str) -> Generator[Dict[str, str], None, None]:
        """
        Read and format test examples from data file.

        Args:
            data_path: Path to the data file

        Yields:
            Dictionary containing task_id and formatted prompt
        """
        with open(data_path, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {data_path}")
        return questions

    def extract_answer(output: str) -> str:
        """Extract the final answer from a model-generated solution.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \boxed.
        """
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer
        except:
            return ""
