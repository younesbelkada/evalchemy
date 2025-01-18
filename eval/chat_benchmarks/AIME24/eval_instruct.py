import json
import os
import tempfile
import logging

from tqdm import tqdm
from typing import Dict, Any, List, Optional

from lm_eval.tasks.hendrycks_math.utils import remove_boxed, last_boxed_only_string, is_equiv
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark

PROMPT = """Problem: {problem}\nAnswer:"""


class AIME24Benchmark(BaseBenchmark):
    """
    AIME24 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/aime24
    """

    def __init__(
        self,
        data_dir: str = "eval/chat_benchmarks/AIME24/data",
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
        self.data_dir = data_dir
        self.debug = debug

    def read_test_examples(self, data_path: str) -> List[Dict[str, str]]:
        """Read the AIME24 dataset from the given data path."""
        with open(data_path, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {data_path}")
        return questions

    # Implements abstract method from BaseBenchmark
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        try:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

            problem_file = os.path.join(self.data_dir, "aime24.json")
            examples = list(self.read_test_examples(problem_file))
            self.logger.info(f"Processing {len(examples)} examples")

            all_instances = []
            for idx, example in enumerate(examples):
                try:
                    inputs = model.apply_chat_template(
                        [{"role": "user", "content": PROMPT.format(problem=example["problem"])}]
                    )

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "do_sample": False,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            self.logger.info("Generating responses for AIME24...")
            outputs = self.compute(model, all_instances)

            # Return None early for non-primary ranks
            if model.rank != 0:
                return None

            generated_examples = []
            for example, output in zip(examples, outputs):
                example_with_output = example.copy()
                example_with_output["output"] = output
                processed_output = extract_answer(output)
                example_with_output["processed_output"] = processed_output
                generated_examples.append(example_with_output)

            output_path = os.path.join(temp_dir, "aime24.jsonl")
            with open(output_path, "w", encoding="utf-8") as fw:
                for ex in generated_examples:
                    fw.write(json.dumps(ex) + "\n")

            self.logger.info(f"Saved {len(generated_examples)} examples to {output_path}")

            return {
                "temp_dir_obj": temp_dir_obj,
                "num_examples": len(generated_examples),
                "total_examples": len(examples),
            }

        except Exception as e:
            self.logger.error(f"Error in generate_responses: {str(e)}")
            raise

    # Implements abstract method from BaseBenchmark
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

        try:
            temp_dir_obj = results["temp_dir_obj"]
            temp_dir = temp_dir_obj.name
            result_path = os.path.join(temp_dir, "aime24.jsonl")

            with open(result_path, "r") as f:
                examples = [json.loads(x) for x in f]

            num_solved = 0
            total = 0
            for example in tqdm(examples):
                num_solved += int(is_equiv(str(example["expected_answer"]), example["processed_output"]))
                total += 1

            results.update(
                {
                    "num_examples_processed": total,
                    "num_solved": num_solved,
                }
            )

            temp_dir_obj.cleanup()
            return results

        except Exception as e:
            self.logger.error(f"Error in evaluate_responses: {str(e)}")
            if temp_dir_obj:
                temp_dir_obj.cleanup()
            raise
