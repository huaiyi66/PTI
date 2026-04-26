import copy
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import PreTrainedTokenizer
from transformers.generation import LogitsProcessor
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import pdb
from qwen_vl.qwen_generation_utils import make_context

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def make_context_refined(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "You are a helpful assistant.",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []
    
    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            text = f"{role}\n{content}"

            role_tokenized = tokenizer.encode(role, allowed_special=set(tokenizer.IMAGE_ST))
            content_tokenized = tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))
            token = role_tokenized + nl_tokens + content_tokenized

            return text, token
        
        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"
            
            current_context_size = len(system_tokens) + len(next_context_tokens) + len(context_tokens)

            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break
        
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text

        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        input_ids = torch.tensor([context_tokens]).to("cuda")
        # 151857 is <img>, 151858 is </img>
        img_start_idx = torch.where(input_ids == 151857)[1][0].item() + 1  # 19 + 1 = 20
        img_end_idx = torch.where(input_ids == 151858)[1][0].item()  # 276  Subtraction: 276 - 20 = 256
    
    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text, return_tensors='pt', padding='longest')
        # pdb.set_trace()
        # input_ids = context_tokens.input_ids.to("cuda")
        input_ids = context_tokens.to("cuda")
        img_start_idx = None
        img_end_idx = None
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    
    return raw_text, input_ids, img_start_idx, img_end_idx
