from typing import Dict, List

import datasets
from lm_eval.tasks.hendrycks_math.utils import is_equiv

def extract_answer(output: str) -> str:
    '''
        Input: model-generated solution
        Output: extracted final answer. Output "" if the final answer is not inside \\boxed
    '''
    try:
        answer = remove_boxed(last_boxed_only_string(output))
        return answer
    except:
        return ""
        
def process_result(answer: str, solution: str) -> Dict[str, int]:
    '''
        Input: answer - gold final answer, solution - predicted final answer
        Output: whether the gold answer and predicted final answers are equivalent
    '''
    retval = 0
    if is_equiv(answer, solution):
        retval = 1
    return retval