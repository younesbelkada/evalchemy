from typing import List, Dict, Any, Optional, Union, Tuple
import json

from bespokelabs import curator
from datasets import Dataset

from lm_eval.api.model import TemplateLM
from lm_eval.models.api_models import JsonChatStr
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from lm_eval.models.utils import handle_stop_sequences
import os


class prompter(curator.LLM):
    def prompt(self, row):
        return row["messages"]

    def parse(self, row, response):
        return {"response": response}


@register_model("curator")
class CuratorAPIModel(TemplateLM):
    def __init__(
        self,
        model: str = None,
        pretrained: str = None,
        max_length: Optional[int] = 2048,
        num_concurrent: int = 1,
        max_retries: int = 3,
        timeout: int = 300,
        tokenized_requests: bool = False,
        **kwargs,
    ):
        super().__init__()
        os.environ["CURATOR_DISABLE_CACHE"] = "true"
        if tokenized_requests:
            raise NotImplementedError("Tokenized requests not implemented for curator.")
        self.tokenized_requests = False
        self.model_name = model or pretrained
        self.max_length = max_length
        self.num_concurrent = num_concurrent
        self.max_retries = max_retries
        self.timeout = timeout
        self.llm = None
        self.gen_kwargs = {}
        self._max_gen_toks = 2048
        self.eos = None

        # Disable cache since it is not necessary
        os.environ["CURATOR_DISABLE_CACHE"] = "true"

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = False,
        gen_kwargs: Optional[dict] = None,
        eos=None,
        **kwargs,
    ) -> dict:
        assert generate, "Curator only supports generation."
        # Create the payload for the API request
        if self.llm is None:
            self.gen_kwargs = gen_kwargs.copy()
            self.eos = eos
            max_tokens = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.get("until", None), eos)
            gen_kwargs = {
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
            }
            self.llm = prompter(model_name=self.model_name, generation_params=gen_kwargs)
        else:
            assert self.gen_kwargs == gen_kwargs, "Generation parameters must be the same for all requests in curator"
            assert self.eos == eos, "EOS must be the same for all requests in curator"
        return messages

    def create_message(
        self, messages: Union[List[List[int]], List[str], List[JsonChatStr]], generate=False
    ) -> Union[List[List[int]], List[dict], List[str], str]:
        # Convert messages to the format expected by the API
        if isinstance(messages, list) and all(isinstance(m, JsonChatStr) for m in messages):
            return Dataset.from_dict({"messages": [json.loads(m.prompt) for m in messages]})
        else:
            raise ValueError("Messages must be a list of JsonChatStr objects")

    @staticmethod
    def parse_logprobs(
        outputs: Union[Any, List[Any]], tokens: List[List[int]] = None, ctxlen: List[int] = None, **kwargs
    ) -> List[Tuple[float, bool]]:
        # Implement log probability parsing logic
        raise NotImplementedError("Log probability parsing not implemented.")
        logprobs = []
        for output in outputs:
            # Assuming output has a structure that includes log probabilities
            logprob = output.get("logprob", 0.0)  # Replace with actual key
            is_greedy = output.get("is_greedy", False)  # Replace with actual key
            logprobs.append((logprob, is_greedy))
        return logprobs

    @staticmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        # Parse the generated outputs from the API
        return [output["response"] for output in outputs]

    @property
    def tokenizer_name(self) -> str:
        return self.model_name

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> Union[str, JsonChatStr]:
        # Convert chat history to the required format
        return JsonChatStr(json.dumps(chat_history))

    def model_call(self, messages: Union[List[List[int]], List[str], List[JsonChatStr]], **kwargs) -> Optional[dict]:
        payload = self._create_payload(self.create_message(messages), **kwargs)
        response = self.llm(payload)["response"]
        return response

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Log likelihood tokens not implemented for curator.")
        results = []
        for context, continuation in requests:
            # Assuming the model can compute log likelihoods
            response = self.model_call([context, continuation])
            logprob = response.get("logprob", 0.0)  # Replace with actual key
            is_greedy = response.get("is_greedy", False)  # Replace with actual key
            results.append((logprob, is_greedy))
        return results

    @property
    def eot_token_id(self) -> Optional[int]:
        # Assuming the model has a specific end-of-text token ID
        return self.llm.eot_token_id  # Replace with actual method to get EOT token ID

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        # Tokenize contexts if required
        if self.tokenized_requests:
            raise NotImplementedError("Tokenized requests not implemented for curator.")

        # Extract contexts and generation kwargs from the Instance objects
        contexts = [req.args[0] for req in requests]
        gen_kwargs = [req.args[1] for req in requests]

        # Assert all gen_kwargs are the same
        assert all(
            gen_kwargs[0] == gkw for gkw in gen_kwargs
        ), "Generation parameters must be the same for all requests in curator"

        contexts_dataset = self.create_message(contexts)
        payload = self._create_payload(contexts_dataset, generate=True, gen_kwargs=gen_kwargs[0])
        response = self.llm(payload)["response"]
        return response

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError("Log likelihood rolling not implemented for curator.")
        loglikelihoods = []
        for context in requests:
            response = self.model_call(context)
            loglikelihood = response.get("loglikelihood", 0.0)  # Replace with actual key
            loglikelihoods.append(loglikelihood)
        return loglikelihoods

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        raise NotImplementedError("Token encoding not implemented for curator.")
        return self.llm.tokenizer.encode(string)  # Replace with actual method to tokenize
