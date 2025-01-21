import json
import logging

from datasets import load_dataset
from eval.task import BaseBenchmark
from lm_eval.api.model import LM
from swebench.harness.run_evaluation import main as run_evaluation
from typing import Any, Dict, Optional

PREDS_PATH = "temp_swebench_preds.json"


class SWEBenchBenchmark(BaseBenchmark):
    """
    SWE-bench (Software Engineering) benchmark for evaluating
    Language Models' ability to resolve GitHub Issues.
    """
    def __init__(
        self,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        split: str = "test",
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)
        self.split = split
        self.debug = debug
        """
        Options for <dataset + split(s)>:
        - princeton-nlp/SWE-bench (dev, test)
        - princeton-nlp/SWE-bench_Lite (test)
        - princeton-nlp/SWE-bench_Verified (test)
        - princeton-nlp/SWE-bench_Multimodal (dev, test)
        """
        self.dataset = load_dataset(dataset, split=self.split)
    
    def generate_responses(self, model: LM) -> Dict[str, Any]:
        pass

    
    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        with open(PREDS_PATH, "w") as f:
            json.dump(results, f)

        run_evaluation(
            dataset_name=self.dataset_name,
            split=self.split,
            predictions_path=PREDS_PATH,
            instance_ids=None,
            max_workers=4,
            force_rebuild=False,
            cache_level="none",
            clean=False,
            open_file_limit=4096,
            run_id="swe-bench-evalchemy",
            timeout=1800,
            namespace="swebench",
            rewrite_reports=False,
            modal=False,
            instance_image_tag="v1",
            report_dir="."
        )

        return json.load(open(""))
    
    def run_benchmark(self, model: LM) -> Dict[str, float]:
        self.logger.info("Starting SWE-bench evaluation")
        try:
            results = self.generate_responses(model)
            return self.evaluate_responses(results)
        except Exception as e:
            self.logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}
