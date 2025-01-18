from typing import Dict, List, Any, Generator, Optional
import json
import os
import re
import tempfile
import logging
from tqdm import tqdm
from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark
from eval.chat_benchmarks.AMC23.utils import *

PROMPT = """Problem: {problem}\nAnswer:"""


class AMC23Benchmark(BaseBenchmark):
    """
    AMC23 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/amc23
    """

    def __init__(
        self,
        data_dir: str = "eval/chat_benchmarks/AMC23/data",
        max_tokens: int = 1024,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize AMC23 benchmark.

        Args:
            data_dir: Directory containing MBPP datasets
            max_tokens: Maximum number of tokens for generation
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.data_dir = data_dir
        self.max_tokens = max_tokens
        self.debug = debug

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

            problem_file = os.path.join(self.data_dir, "amc23.json")
            examples = list(self.read_test_examples(problem_file))
            self.logger.info(f"Processing {len(examples)} examples")

            all_instances = []
            for idx, example in enumerate(examples):
                try:
                    inputs = model.apply_chat_template(
                        [{"role": "user", "content": PROMPT.format(problem=example["question"])}]
                    )

                    all_instances.append(
                        Instance(
                            "generate_until",
                            example,
                            (
                                inputs,
                                {
                                    "max_gen_toks": self.max_tokens,
                                    "do_sample": False,
                                },
                            ),
                            idx,
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Error preparing instance {idx}: {str(e)}")
                    continue

            self.logger.info("Generating responses for AMC23...")
            outputs = self.compute(model, all_instances)

            # Return None early for non-primary ranks
            if model.rank != 0:
                return None

            generated_examples = []
            for example, output in zip(examples, outputs):
                try:
                    example_with_output = example.copy()
                    example_with_output["output"] = output
                    processed_output = extract_answer(output)
                    example_with_output["processed_output"] = processed_output
                    print(f"example: {example}")
                    print(f"output: {output}")
                    print(f"processed output: {processed_output}")
                    generated_examples.append(example_with_output)
                except Exception as e:
                    self.logger.error(f"Error processing output for {example['id']}: {str(e)}")
                    continue

            output_path = os.path.join(temp_dir, "amc23.jsonl")
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
            result_path = os.path.join(temp_dir, "amc23.jsonl")

            with open(result_path, "r") as f:
                examples = [json.loads(x) for x in f]

            num_solved = 0
            total = 0
            for example in tqdm(examples):
                num_solved += process_result(str(example["answer"]), example["processed_output"])
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

    def run_benchmark(self, model: LM) -> Dict[str, float]:
        """
        Run the complete AMC23 benchmark evaluation pipeline.

        Args:
            model: Language model instance

        Returns:
            Dictionary containing evaluation metrics, or None for non-primary ranks
        """
        self.logger.info("Starting AMC23 benchmark evaluation")
        try:
            generation_results = self.generate_responses(model)

            # If not primary rank, return None early
            if generation_results is None:
                return None

            evaluation_results = self.evaluate_responses(generate_responses)

            return evaluation_results

        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
