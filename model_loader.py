import os
import myutils
from collections import namedtuple
import PIL.Image
import torch
import yaml
from anchor import (
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    SYSTEM_MESSAGE
)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
import pdb
from transformers import StoppingCriteria, StoppingCriteriaList
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

def load_model_args_from_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    ModelArgs = namedtuple("ModelArgs", data["ModelArgs"].keys())
    TrainingArgs = namedtuple("TrainingArgs", data["TrainingArgs"].keys())

    model_args = ModelArgs(**data["ModelArgs"])
    training_args = TrainingArgs(**data["TrainingArgs"])

    return model_args, training_args


def load_llava_model(model_path):
    model_name = get_model_name_from_path(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    return tokenizer, model, image_processor, model


def load_qwen_model(model_path):
    from qwen_vl.modeling_qwen import QWenLMHeadModel
    from qwen_vl.qwen_generation_utils import decode_tokens, get_stop_words_ids
    from transformers import AutoTokenizer

    model_path = os.path.expanduser(model_path)
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map='auto',
        trust_remote_code=True, 
        fp16=True,
    ).eval()


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id  # you can also set bos_id and eos_id to tokenizer.eod_id
    image_processor = model.transformer.visual.image_transform
    return tokenizer, model, image_processor, model


def load_pretrained_qwen_vl(model_path, device_map="cuda", bf16=False, fp16=True, load_in_4bit=False):
    from transformers import   AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from qwen_vl.modeling_qwen import QWenLMHeadModel
    # from qwen_vl_chat.modeling_qwen import QWenLMHeadModel
    # from qwen_vl.qwen_generation_utils import decode_tokens, get_stop_words_ids
    '''
    # special tokens
    IMSTART='<|im_start|>' # 151644
    IMEND='<|im_end|>' # 151645
    ENDOFTEXT='<|endoftext|>' # 151643
    '''
    # bf16 or fp16=True -> device_map="auto"
    
    quantization_config = None
    
    kwargs = {"device_map": device_map, 
              "bf16": bf16,
              "fp16": fp16, 
              "quantization_config": quantization_config}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)

    model = QWenLMHeadModel.from_pretrained(
        '/home/zhangcs/.cache/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8',
        device_map='auto', # cuda
        trust_remote_code=True, 
        fp16=True,
        low_cpu_mem_usage=True,
    ).eval()


    model.generation_config = GenerationConfig.from_pretrained(model_path, device_map="cuda", trust_remote_code=True)  # generation_config.json
    
    tokenizer.eos_token_id = model.generation_config.eos_token_id
    tokenizer.pad_token_id =  model.generation_config.pad_token_id
    # tokenizer.im_start_id: 151644 '<|im_start|>'
    # tokenizer.im_end_id: 151645 '<|im_end|>'
    # pdb.set_trace()

    # def image_processor(x): # image path
    #     return x
    image_processor = model.transformer.visual.image_transform

    return tokenizer, model, image_processor, model


def load_pretrained_deepseek_vl_chat(model_path):

    # model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor= VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token 
    image_processor = vl_chat_processor.image_processor

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = model.to(torch.bfloat16).cuda().eval()
    # model = model.half().cuda().eval()
    # model = model.to(torch.float16).cuda().eval()

    return tokenizer, model, image_processor, vl_chat_processor, model.language_model



def prepare_llava_inputs(template, query, image, tokenizer, answer=None):
    image_tensor = image["pixel_values"][0]
    if type(image_tensor) != torch.Tensor:
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).to("cuda")
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
    # qu = [template.replace("<question>", q + " Please answer this question with one word.") for q in query]
    qu = [template.replace("<question>", q ) for q in query]
    # qu = [template.replace("<question>", "Describe this image in detail." ) for q in query]
    batch_size = len(query)

    chunks = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]
    # pdb.set_trace()
    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    # pdb.set_trace()
    # tt = tokenizer(chunk_after, return_tensors="pt", padding="longest", add_special_tokens=False, )
    # tt.attention_mask.shape

    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * IMAGE_TOKEN_INDEX
    )

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1).to(torch.int64)
    kwargs = {}
    kwargs["images"] = image_tensor.half()
    kwargs["input_ids"] = input_ids
    return qu, img_start_idx, img_end_idx, kwargs

def prepare_qwenvlchat_inputs(template, query, image, tokenizer):
    '''
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    <img>/home/zhenghh8/hallucination/PAI/coco/val2014/COCO_val2014_000000000042.jpg</img>
    Please help me describe it.<|im_end|>
    <|im_start|>assistant

    '''

    # image_replace = IMAGE_PAD_TAG * IMG_TOKEN_SPAN # 256 * <imgpad>
    qu = [template.replace('<question>', q) for q in query]
    chunks = [p.split('<imagehere>') for p in qu]

    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]

    system_raw = '<|im_start|>system\n' + SYSTEM_MESSAGE + '<im_end>\n'
    token_system = tokenizer.encode(
            system_raw,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        ).to("cuda")

    token_before = tokenizer(
        chunk_before, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to("cuda").input_ids

    image = ['<img>' + img + '</img>' for img in image]  # 256 + 2
    token_img = tokenizer(
        image, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to("cuda").input_ids

    token_after = tokenizer(
        chunk_after, 
        return_tensors="pt", 
        padding="longest", 
        add_special_tokens=False
    ).to("cuda").input_ids

    input_ids = torch.cat([token_system, token_before, token_img, token_after], dim=1)
    # include special token 
    sys_start_idx = 0
    sys_end_idx = sys_start_idx + len(token_system[0])

    instruction_start_idx = sys_end_idx
    instruction_end_idx = instruction_start_idx + len(token_before[0]) + len(token_img[0]) + len(token_after[0])

    img_start_idx = instruction_start_idx + len(token_before[0])
    img_end_idx = instruction_start_idx + len(token_before[0]) + len(token_img[0])

    kwargs = {}
    kwargs["input_ids"] = input_ids
    # pdb.set_trace()

    return qu, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs


def prepare_deepseekvlchat_inputs(query, image_data, vl_gpt, vl_chat_processor):
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>{query}",
            "images": [f"{image_data}"],
        },
        {"role": "Assistant", "content": ""},
    ]
    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    kwargs = {}
    kwargs["inputs_embeds"] =  inputs_embeds
    kwargs["attention_mask"] =  prepare_inputs['attention_mask']
    img_start_idx, img_end_idx, = 41, 41+576
    # pdb.set_trace()

    # --- 步骤 1: 准备用于计算缓存的输入 (除最后一个token外的所有内容) ---
    cache_input_ids = prepare_inputs['input_ids'][:, :-1]
    cache_attention_mask = prepare_inputs['attention_mask'][:, :-1]
    cache_images_seq_mask = prepare_inputs['images_seq_mask'][:, :-1]
    # 图像 pixel_values 必须保留，因为前缀部分需要看到图像
    cache_prepare_inputs = {
        'sft_format': prepare_inputs['sft_format'],
        "input_ids": cache_input_ids,
        "attention_mask": cache_attention_mask,
        "pixel_values": prepare_inputs['pixel_values'],
        'images_seq_mask':cache_images_seq_mask,
        'images_emb_mask':prepare_inputs['images_emb_mask'],
    }
    prefix_embeds = vl_gpt.prepare_inputs_embeds(**cache_prepare_inputs)
    last_token_input_ids = prepare_inputs['input_ids'][:, -1:]
    # 新的 attention_mask 是旧的 mask 后面拼上一个 1
    # generation_attention_mask = torch.cat(
    #     [cache_attention_mask, torch.ones_like(last_token_input_ids)], dim=-1
    # )
    kwargs["prefix_embeds"] =  prefix_embeds
    kwargs["input_ids"] =  last_token_input_ids
    # kwargs["attention_mask"] =  generation_attention_mask

    return query, img_start_idx, img_end_idx, kwargs

 
def prepare_deepseekvlchat_inputs_v2(template, query, image, tokenizer, vl_gpt, ):
    # image_tensor = image
    image_tensor = image["pixel_values"].to(vl_gpt.dtype).to(vl_gpt.device)
    if type(image_tensor) != torch.Tensor:
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).to("cuda")
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
    # pil_img = PIL.Image.open(image_path)
    # pil_img = pil_img.convert("RGB")
    
    qu = [template.replace("<question>", q ) for q in query]
    batch_size = len(query)

    chunks = [q.split("<image_placeholder>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after = [chunk[1] for chunk in chunks]
    # pdb.set_trace()
    token_before = (
        tokenizer(
            chunk_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    token_after = (
        tokenizer(
            chunk_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
        )
        .to("cuda")
        .input_ids
    )
    # pdb.set_trace()
    # tt = tokenizer(chunk_after, return_tensors="pt", padding="longest", add_special_tokens=False, )
    # tt.attention_mask.shape

    bos = (
        torch.ones([batch_size, 1], dtype=torch.int64, device="cuda")
        * tokenizer.bos_token_id
    )

    img_start_idx = len(token_before[0]) + 1
    img_end_idx = img_start_idx + IMAGE_TOKEN_LENGTH
    image_token = ( torch.ones([batch_size, 1], dtype=torch.int64, device="cuda") * 100015 )
    image_token = image_token.repeat(1, 576)

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1).to(torch.int64).to(vl_gpt.device)
    
    seq_len = input_ids.shape[-1]
    # batched_images_seq_mask = torch.zeros((batch_size, seq_len)).bool()
    batched_images_seq_mask =  input_ids == 100015
    batched_images_emb_mask = torch.ones((batch_size, 1, 576)).bool()
    batched_attention_mask = torch.ones((batch_size, seq_len)).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(pixel_values=image_tensor,
                                                 input_ids=input_ids,
                                                 images_seq_mask=batched_images_seq_mask.to(vl_gpt.device),
                                                 images_emb_mask=batched_images_emb_mask.to(vl_gpt.device),
                                                 )
    
    prefix_embeds = vl_gpt.prepare_inputs_embeds(pixel_values=image_tensor,
                                                 input_ids=input_ids[:, :-1],
                                                 images_seq_mask=batched_images_seq_mask[:, :-1].to(vl_gpt.device),
                                                 images_emb_mask=batched_images_emb_mask.to(vl_gpt.device),
                                                 )

    last_token_input_ids = input_ids[:, -1:]

    kwargs  =  {}
    kwargs["inputs_embeds"] =  inputs_embeds
    kwargs["prefix_embeds"] =  prefix_embeds
    kwargs["input_ids"] =  last_token_input_ids
    kwargs["attention_mask"] =  batched_attention_mask

    return qu, img_start_idx, img_end_idx, kwargs
 


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all(input_ids[:, -len(stop):] == stop).item():
                return True
        return False
    
def stop_word_to_criteria(stop_word_ids):
    stop_words_ids = [torch.tensor(ids).to(device='cuda') for ids in stop_word_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.vlm_model = None
        self.llm_model = None
        self.image_processor = None
        self.load_model()


    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser("/home/zhangcs/zhangcs/code/VISTA/llava/llava-v1.5-7b")
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )

        elif self.model_name == "qwen-vl-chat":
            model_path = "Qwen/Qwen-VL-Chat"
            # model_path = os.path.expanduser("/home/zhangcs/zhangcs/model_weight/Qwen-VL")

            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_pretrained_qwen_vl(model_path)
            )
            self.stopping_criteria = stop_word_to_criteria(stop_word_ids=[[151643, 151644, 151645]])
            # self.stopping_criteria = stop_word_to_criteria(stop_word_ids=[[151643, 151644, 151645]])
            # IMSTART='<|im_start|>' # 151644
            # IMEND='<|im_end|>' # 151645
            # ENDOFTEXT='<|endoftext|>' # 151643
        elif self.model_name == "deepseek-vl-chat":
            model_path = "deepseek-ai/deepseek-vl-7b-chat"
            self.tokenizer, self.vlm_model, self.image_processor, self.processor, self.llm_model = (
                load_pretrained_deepseek_vl_chat(model_path)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    

    def prepare_inputs_for_model(self, template, query, image):
        if self.model_name == "llava-1.5":
            questions, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs(
                template, query, image, self.tokenizer
            )
   
        elif self.model_name == 'qwen-vl-chat':
            questions, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs = prepare_qwenvlchat_inputs(
                template, query, image, self.tokenizer
            )
        elif self.model_name == 'deepseek-vl-chat':
            # questions, img_start_idx, img_end_idx, kwargs = prepare_deepseekvlchat_inputs(query[0], image[0], self.vlm_model, self.processor)
            questions, img_start_idx, img_end_idx, kwargs = prepare_deepseekvlchat_inputs_v2(template, query, image, self.tokenizer, self.vlm_model)
            
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.img_start_idx = img_start_idx
        self.img_end_idx = img_end_idx
        return questions, kwargs

    def decode(self, output_ids, inputs_len=None):
        # get outputs
        if self.model_name == "llava-1.5":
            # replace image token by pad token
            # pdb.set_trace()
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            output_text = [text.split("ASSISTANT:")[-1].strip() for text in output_text]

        elif self.model_name == "qwen-vl-chat":
            output_text = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            output_text = [text.split('assistant\n')[-1].strip() for text in output_text]
        
        elif self.model_name == 'deepseek-vl-chat':
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        return output_text