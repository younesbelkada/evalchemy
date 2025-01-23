import json
import logging
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

import base64
import zlib
import pickle
import json
import copy
from .livecodebench_utils import lcb_run, map_to_example, has_test_type, post_process_code, translate_private_test_cases

from eval.task import BaseBenchmark
from datasets import load_dataset

import re
def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    return matches


class LiveCodeBenchBenchmark(BaseBenchmark):
    """
    LiveCodeBench Benchmark for evaluating the math reasoning of LLMs.

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize LiveCodeBench benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.debug = debug
        self.max_gen_toks = 16384  # set higher to avoid truncation for reasoning models


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
        if self.debug:
            examples = examples[:10]

        # Prepare instances for model
        all_instances = []
        for idx, example in enumerate(examples):
            if examples["is_stdin"]:
                prompt_text = "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition." + example["prompt"]
            else:
                prompt_text = "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution." + example["prompt"]
    
            messages = [
                {"role": "system", "content": "You are a helpful and harmless assistant. You are DeepSeek R1 developed by DeepSeek. You should think step-by-step."},
                {"role": "user", "content": prompt_text}
                ]

            templated_messages = model.apply_chat_template(messages)

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (templated_messages, {"do_sample": False, "max_gen_toks": self.max_gen_toks}),
                    idx,
                )
            )

        # Generate model responses
        self.logger.info("Generating responses for LiveCodeBench...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        examples_list = []
        
        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = has_code(output)
            examples_list.append(example)
                
        return {"examples": examples_list}
            

    def check_correctness(
            self,
            problem: Dict,
            completion: str,
            timeout: float,
            is_extracted: bool = False
        ) -> Dict:
            """
            Evaluates the functional correctness of a completion by running the test
            suite provided in the problem.

            :param completion_id: an optional completion ID so we can match
                the results later even if execution finishes asynchronously.
            """
            result_list = lcb_run(problem, completion, timeout, is_extracted)
            details = [r[0] for r in result_list]
            all_passed = all(details)
            
            result = ""
            if result_list and all_passed:
                result = "passed"

            return result == "passed"
    

    def evaluate_responses(self, responses: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        results = []
        total_correct = 0
        total_finish = 0
        for example in responses["examples"]:
            response_entry = {
                    "content": example["model_answer"],
                    "correctness": None,
                    "reason": None,
                }

            code_filter_result = example["model_answer"]

            if len(code_filter_result) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Does not contain code component."
            else:	
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(example)
                curr_res = self.check_correctness(problem=problem_to_check, completion=post_process_code(last_code), timeout=6, is_extracted=not problem_to_check["is_stdin"])
                if curr_res:
                    response_entry["correctness"] = True
                    response_entry["reason"] = ""
                else:
                    response_entry["correctness"] = False
                    response_entry["reason"] = "Code is incorrect."
            total_correct += response_entry["correctness"]
            total_finish += 1

            results.append(response_entry)

        return {
                    "total_correct": total_correct, 
                    "total_finish": total_finish,
                    "accuracy": total_correct / total_finish,
                    "examples": results
                }

    def load_questions(self) -> List[Dict[str, str]]:
        """Load LiveCodeBench questions from source."""
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", split='test[:500]', trust_remote_code=True)
        dataset = dataset.map(
            lambda example: {
                "private_test_cases": translate_private_test_cases(example["private_test_cases"])
            }
        )
        dataset = dataset.map(map_to_example, remove_columns=dataset.column_names)
        return dataset
