
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
import math
import pdb
from transformers import set_seed
import re
import random
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from transformers import DynamicCache, BatchEncoding, PreTrainedModel
from cache_utils.steering.config import SteeringConfig
from cache_utils.utils.constants import AggregationMethods
from cache_utils.utils.logging_setup import logger
from anchor import SYSTEM_MESSAGE
from model_loader import prepare_llava_inputs, prepare_deepseekvlchat_inputs, prepare_qwenvlchat_inputs
from pycocotools.coco import COCO
from myutils import PCA 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from cache_utils.cache_util import _random_mask_patchgrid, remove_words_from_desc

from cache_utils.cache_util import repeat_past_for_beams
def process_image_with_mask(image_processor, image_raw, coco=None, seg_ann_info=None, random=False, mask_ratio=0.9):
    image_tensor = np.array(image_raw).transpose(2, 0, 1)  # 转为[C, H, W]
    image_tensor = torch.from_numpy(image_tensor).float()
    image = image_processor(image_raw)
    
    if seg_ann_info:
        all_mask = torch.zeros(image_tensor[0].shape)
        for i, ann in enumerate(seg_ann_info):
            mask = coco.annToMask(ann)
            mask_tensor = torch.from_numpy(mask).float()
            all_mask = torch.max(all_mask, mask_tensor) 
        
        mask_tensor = all_mask.repeat(image_tensor.shape[0], 1, 1)
        image_object = image_tensor * mask_tensor
        reverse_mask = 1 - mask_tensor
        image_background = image_tensor * reverse_mask


        image_object = image_processor(image_object)
        image_background = image_processor(image_background)

        return image, image_object, image_background
    
    if random:
        C, H, W = image_tensor.shape
        device = image_tensor.device

        keep_ratio = 1 - mask_ratio
        rand_mask = _random_mask_patchgrid(H, W, keep_ratio, device=device)
        mask_tensor = rand_mask.repeat(C, 1, 1)
        image_neg = image_tensor * mask_tensor
        image_neg = image_processor(image_neg)

        return image, image_neg, image

    return image


def process_image_with_mask_qwen(image_raw, raw_path, coco=None, seg_ann_info=None, random=False, mask_ratio=0.9):
    """
    处理图像时应用mask，保留指定区域
    mask_tensor: 二值mask张量（1表示保留区域，0表示去除区域），若为None则返回原图
    """

    # 先将原始图像转为张量（如RGB格式，shape: [3, H, W]）
    image_tensor = np.array(image_raw).transpose(2, 0, 1)  # 转为[C, H, W]
    image_tensor = torch.from_numpy(image_tensor).float() # torch.Size([3, 480, 640])
    
    image_file = raw_path.replace('train2014', 'train2014_qwen')
    image_object_file = image_file.replace('.jpg', '_object.jpg')
    image_background_file = image_file.replace('.jpg', '_background.jpg')

    return image_file, image_object_file, image_background_file
    
    if seg_ann_info:
        all_mask = torch.zeros(image_tensor[0].shape)
        for i, ann in enumerate(seg_ann_info):
            mask = coco.annToMask(ann)
            mask_tensor = torch.from_numpy(mask).float()
            all_mask = torch.max(all_mask, mask_tensor) 
        
        mask_tensor = all_mask.repeat(image_tensor.shape[0], 1, 1)
        image_object = image_tensor * mask_tensor
        reverse_mask = 1 - mask_tensor
        image_background = image_tensor * reverse_mask

        image_object_img = Image.fromarray(image_object.detach().cpu().numpy().transpose(1, 2, 0).astype('uint8'))
        image_background_img = Image.fromarray(image_background.detach().cpu().numpy().transpose(1, 2, 0).astype('uint8'))

        image_raw.save(image_file)
        image_object_img.save(image_object_file)
        image_background_img.save(image_background_file)

        return image_file, image_object_file, image_background_file
    
   

def get_prompts_v2(data_demos, template, question, image_processor, tokenizer=None, vlm_model=None, coco=None, evaluator=None, args=None):

    # llava
    images = []
    input_ids_org, input_ids_positive, input_ids_negative = [], [], []
    images_original, images_positive, images_negative  =  [], [], []

    # minigpt4
    input_iems_positive_txt, input_iems_negative_txt = [], []
    attnmask_positive_txt, attnmask_negative_txt = [], []
    input_iems_positive_img, input_iems_negative_img = [], []
    attnmask_positive_img, attnmask_negative_img =  [], []

    # qwen
    input_ids_positive_txt, input_ids_negative_txt = [], []
    input_ids_positive_img, input_ids_negative_img = [], []
    
    for data in tqdm(data_demos, desc="Construct positives and negatives"):

        file_name = data['image_file']
        image_path = os.path.join(args.data_file, 'train2014', file_name + '.jpg')
        seg_ann_info = data['anns']

        joined = ", ".join(data['Object']) + '.'

        positive_answer = joined
        negative_answer = remove_words_from_desc(data['Object'], data['desc']) 

        raw_image = Image.open(image_path).convert("RGB")
        # print(joined, negative_answer)
        
        if args.model== "llava-1.5": 
            org_image, pos_image, neg_image  = process_image_with_mask(image_processor, raw_image, coco, seg_ann_info)
        elif args.model == "qwen-vl-chat" or args.model == "deepseek-vl-chat":
            org_image, pos_image, neg_image  = process_image_with_mask_qwen(raw_image, image_path, coco, seg_ann_info)

        query_positive = [( positive_answer)] 
        query_negative = [( negative_answer)] 

        template_pos = template 
        template_neg = template
        
        
        if args.model== "llava-1.5":
            qu_pos, img_start_idx, img_end_idx, kwargs_positive = prepare_llava_inputs(template_pos,  query_positive , pos_image, tokenizer)
            qu_neg, _, _, kwargs_negative = prepare_llava_inputs(template_neg,  query_negative , neg_image, tokenizer)

            _, _, _, kwargs_ = prepare_llava_inputs(template, [question] , org_image, tokenizer)

            input_ids_positive.append(kwargs_positive["input_ids"])
            input_ids_negative.append(kwargs_negative["input_ids"])
            images_positive.append(kwargs_positive["images"]) # (1, 3, 336, 336)
            images_negative.append(kwargs_negative["images"]) # (1, 3, 336, 336)
            images_original.append(kwargs_["images"]) # (1, 3, 336, 336)
            input_ids_org.append(kwargs_["input_ids"])

            
        elif args.model == "qwen-vl-chat":
      
            qu_pos, img_start_idx, img_end_idx, sys_start_idx, sys_end_idx, instruction_start_idx, instruction_end_idx, kwargs_positive_txt = prepare_qwenvlchat_inputs(template_pos,  query_positive, [org_image], tokenizer)
            qu_neg, _, _, _, _, _, _, kwargs_negative_txt = prepare_qwenvlchat_inputs(template_neg,  query_negative , [org_image], tokenizer)
            qu_org, _, _, _, _, _, _, kwargs_positive_img = prepare_qwenvlchat_inputs(template,   [question] , [pos_image], tokenizer)
            _, _, _, _, _, _, _, kwargs_negative_img = prepare_qwenvlchat_inputs(template,   [question] , [neg_image], tokenizer)


            input_ids_positive_txt.append(kwargs_positive_txt["input_ids"])
            input_ids_negative_txt.append(kwargs_negative_txt["input_ids"])
            input_ids_positive_img.append(kwargs_positive_img["input_ids"])
            input_ids_negative_img.append(kwargs_negative_img["input_ids"])

        elif args.model == "deepseek-vl-chat":

            qu_pos, img_start_idx, img_end_idx, kwargs_positive_txt = prepare_deepseekvlchat_inputs(query_positive[0],  org_image, vlm_model, image_processor)
            qu_neg, _, _, kwargs_negative_txt = prepare_deepseekvlchat_inputs(query_negative[0],  org_image , vlm_model, image_processor)
            qu_org,  _, _, kwargs_positive_img = prepare_deepseekvlchat_inputs(question[0],  pos_image , vlm_model, image_processor)
            _,  _, _, kwargs_negative_img = prepare_deepseekvlchat_inputs(question[0],  neg_image, vlm_model, image_processor)

            input_iems_positive_txt.append(kwargs_positive_txt["inputs_embeds"].detach().cpu())
            input_iems_negative_txt.append(kwargs_negative_txt["inputs_embeds"].detach().cpu())
            attnmask_positive_txt.append(kwargs_positive_txt["attention_mask"].detach().cpu())
            attnmask_negative_txt.append(kwargs_negative_txt["attention_mask"].detach().cpu())


            input_iems_positive_img.append(kwargs_positive_img["inputs_embeds"].detach().cpu())
            input_iems_negative_img.append(kwargs_negative_img["inputs_embeds"].detach().cpu())
            attnmask_positive_img.append(kwargs_positive_img["attention_mask"].detach().cpu())
            attnmask_negative_img.append(kwargs_negative_img["attention_mask"].detach().cpu())

        else:
            raise ValueError(f"Unknown model: {args.model_name}")
    
    args.img_start_idx =  img_start_idx
    args.img_end_idx = img_end_idx
    
    if args.model== "llava-1.5":
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)

        images = [(images_negative[demo_id], images_positive[demo_id]) for demo_id in range(len(images_negative))]
        images = tuple(images)
        
        return input_ids_org, images_original, images, inputs
    
    elif args.model == "minigpt4" or args.model == "deepseek-vl-chat":
        inputs_txt = [(input_iems_negative_txt[demo_id], input_iems_positive_txt[demo_id]) for demo_id in range(len(input_iems_negative_txt))]
        inputs_txt = tuple(inputs_txt)

        inputs_img = [(input_iems_negative_img[demo_id], input_iems_positive_img[demo_id]) for demo_id in range(len(input_iems_positive_img))]
        inputs_img = tuple(inputs_img)

        mask_txt = [(attnmask_negative_txt[demo_id], attnmask_positive_txt[demo_id]) for demo_id in range(len(attnmask_positive_txt))]
        mask_txt = tuple(mask_txt)

        mask_img = [(attnmask_negative_img[demo_id], attnmask_positive_img[demo_id]) for demo_id in range(len(attnmask_negative_img))]
        mask_img = tuple(mask_img)

        return inputs_txt, inputs_img, mask_txt, mask_img
    
    elif args.model == "qwen-vl-chat":
        inputs_txt = [(input_ids_negative_txt[demo_id], input_ids_positive_txt[demo_id]) for demo_id in range(len(input_ids_positive_txt))]
        inputs_txt = tuple(inputs_txt)

        inputs_img = [(input_ids_negative_img[demo_id], input_ids_positive_img[demo_id]) for demo_id in range(len(input_ids_negative_img))]
        inputs_img = tuple(inputs_img)
        
        return inputs_txt, inputs_img


def get_demos(args, image_processor, model, tokenizer, vlm_model, template=None,  file_path='your/path', category='Object',  seg_dict=None): 
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)

    random.seed(args.seed)
    
    data_demos = random.sample(data, args.n_contrastive_samples)
    question = "Describe this image in detail."
    annFile = '/home/zhangcs/zhangcs/dataset/coco/annotations/instances_train2014.json'
    coco = COCO(annFile)

    for i in range(len(data_demos)):
        data_demos[i]['anns'] = seg_dict[data_demos[i]['image_file']] 

    if args.model== "llava-1.5" :
        input_ids, inputs_images, contrast_images, contrast_ids = get_prompts_v2(data_demos, template, question, image_processor, tokenizer=tokenizer,  coco=coco,  evaluator=None, args=args)
        return input_ids, inputs_images, contrast_images, contrast_ids 
    elif args.model == "qwen-vl-chat":
        contrast_txt, contrast_img  = get_prompts_v2(data_demos, template, question, image_processor, tokenizer=tokenizer, coco=coco,  evaluator=None, args=args)
        return contrast_txt, contrast_img
    elif args.model == "deepseek-vl-chat":
        inputs_txt, inputs_img, mask_txt, mask_img   = get_prompts_v2(data_demos, template, question, image_processor, vlm_model=vlm_model, coco=coco,  evaluator=None, args=args)
        return inputs_txt, inputs_img, mask_txt, mask_img
    

def extract_steering_kv_text(
    model,
    tokenizer,
    input_images,
    contrast_images,
    input_ids,
    steering_config: SteeringConfig,
    device="cpu",
    model_name='llava-1.5',
):

    steering_values = defaultdict(lambda: torch.tensor([]).to(device))
    steering_keys = defaultdict(lambda: torch.tensor([]).to(device))
    activations = defaultdict(lambda: torch.tensor([]))
    output_dict = {}
    pos_attn_demo = []
    neg_attn_demo = []
    # for example in tqdm(data.iter(batch_size=batch_size)):
    for example_id in tqdm(range(len(input_images)), desc='Obtaining textual direction'):
        # example.keys() = dict_keys(['steps', 'id', 'question', 'answer', 'positive', 'negative'])

        neg_tokens, pos_tokens = input_ids[example_id]
        images = input_images[example_id]
        # neg_images, pos_images = contrast_images[example_id]

        # Select the indices of the last token in the sequence
        if steering_config.extraction_method == "last_token": # True
            subtract_index = 1
        elif steering_config.extraction_method == "penultimate_token":
            subtract_index = 2
        elif isinstance(steering_config.extraction_method, int):
            subtract_index = steering_config.extraction_method
        else:
            raise ValueError(f"Invalid value provided for extraction_method: {steering_config.extraction_method}. Valid values are ['last_token', 'penultimate_token'] or integers.")

        # Find indices of the tokens to extract the steering vectors from
        pos_indices = torch.tensor(pos_tokens.shape[1] - subtract_index).cuda() # tensor([1312], device='cuda:0') 取最后一个token
        neg_indices = torch.tensor(neg_tokens.shape[1] - subtract_index).cuda() # tensor([477], device='cuda:0')
        batch_indices = torch.arange(pos_tokens.size(0), device=pos_tokens.device)
        # Log the tokens that are used to extract the steering vectors
        if steering_config.verbose:
            extraction_token_pos = tokenizer.batch_decode(pos_tokens[batch_indices, pos_indices], skip_special_tokens=True)
            extraction_token_neg = tokenizer.batch_decode(neg_tokens[batch_indices, neg_indices], skip_special_tokens=True)
            print(f"Extracting steering vectors for the tokens '{extraction_token_pos}' and '{extraction_token_neg}'")
            
        # Run the model with the hook to cache activations
        cache_positive, cache_negative = DynamicCache(), DynamicCache()
        with torch.no_grad():
            pos_kwargs = {'input_ids': pos_tokens, 'images': images.unsqueeze(0).half().to(pos_tokens.device)}
            neg_kwargs = {'input_ids': neg_tokens, 'images': images.unsqueeze(0).half().to(neg_tokens.device)}

            pos_out = model(**pos_kwargs, output_hidden_states=True,  past_key_values=cache_positive)
            neg_out = model(**neg_kwargs, output_hidden_states=True,  past_key_values=cache_negative)


        for layer_id in range(len(cache_positive.value_cache)): # 0-32
            pos_values = cache_positive.value_cache[layer_id][batch_indices, :, pos_indices, :] # torch.Size([1, 32, 128])
            neg_values = cache_negative.value_cache[layer_id][batch_indices, :, neg_indices, :] # torch.Size([1, 32, 128])

            pos_keys = cache_positive.key_cache[layer_id][batch_indices, :, pos_indices, :]
            neg_keys = cache_negative.key_cache[layer_id][batch_indices, :, neg_indices, :]

            # Take the differnece between the vectors
            if steering_config.take_difference: # True
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values - neg_values]) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys - neg_keys])
                # activations[layer_id] = torch.cat([activations[layer_id], pos_activations - neg_activations])
            else:
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values], dim=0) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys], dim=0)
    
    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values:
            steering_values[layer_id] = torch.mean(steering_values[layer_id], dim=0) # [n_heads, head_dim]
            steering_keys[layer_id] = torch.mean(steering_keys[layer_id], dim=0)

        output_dict["values"] = dict(steering_values)
        output_dict["keys"] = dict(steering_keys)

    elif steering_config.aggregation_method == AggregationMethods.pca:
        for layer_id in steering_values:
            values_data = steering_values[layer_id] # [num_demos, H, Dh]
            num_demos, n_heads, head_dim = values_data.shape
            values_reshaped = values_data.view(num_demos, -1)
            
            pca_values = PCA(n_components=1).to(values_reshaped.device).fit(values_reshaped.float())
            direction_values = (pca_values.components_.sum(dim=0,keepdim=True) + pca_values.mean_).mean(0).view(1, n_heads, head_dim)

            steering_values[layer_id] = direction_values
            
            keys_data = steering_keys[layer_id]
            keys_reshaped = keys_data.view(num_demos,-1)
            
            pca_keys = PCA(n_components=1).to(keys_reshaped.device).fit(keys_reshaped.float())
            direction_keys = (pca_keys.components_.sum(dim=0,keepdim=True) + pca_keys.mean_).mean(0).view(1, n_heads, head_dim)
            steering_keys[layer_id] = direction_keys

        output_dict["values"] = dict(steering_values)
        output_dict["keys"] = dict(steering_keys)

 
    return output_dict


def extract_steering_kv_text_deepseek(
    model,
    inputs_txt,
    mask_txt,
    steering_config: SteeringConfig,
    device="cpu",
):

    steering_values = defaultdict(lambda: torch.tensor([]).to(device))
    steering_keys = defaultdict(lambda: torch.tensor([]).to(device))
    output_dict = {}
    for example_id in tqdm(range(len(inputs_txt)), desc='Obtaining textual direction'):

        neg_tokens, pos_tokens = inputs_txt[example_id]
        neg_masks, pos_masks = mask_txt[example_id]

        if steering_config.extraction_method == "last_token": # True
            subtract_index = 1
        elif steering_config.extraction_method == "penultimate_token":
            subtract_index = 2
        elif isinstance(steering_config.extraction_method, int):
            subtract_index = steering_config.extraction_method
        else:
            raise ValueError(f"Invalid value provided for extraction_method: {steering_config.extraction_method}. Valid values are ['last_token', 'penultimate_token'] or integers.")

        pos_indices = torch.tensor(pos_tokens.shape[1] - subtract_index).cuda() 
        neg_indices = torch.tensor(neg_tokens.shape[1] - subtract_index).cuda() 
        batch_indices = torch.arange(pos_tokens.size(0), device=pos_tokens.device)
            
        cache_positive, cache_negative = DynamicCache(), DynamicCache()
        with torch.no_grad():
            pos_kwargs = {'inputs_embeds': pos_tokens.to('cuda'), 'attention_mask': pos_masks.to('cuda')}
            neg_kwargs = {'inputs_embeds': neg_tokens.to('cuda'), 'attention_mask': neg_masks.to('cuda')}

            pos_out = model(**pos_kwargs, output_hidden_states=True,  past_key_values=cache_positive)
            neg_out = model(**neg_kwargs, output_hidden_states=True,  past_key_values=cache_negative)
            
        for layer_id in range(len(cache_positive.value_cache)): # 0-32
            pos_values = cache_positive.value_cache[layer_id][batch_indices, :, pos_indices, :] 
            neg_values = cache_negative.value_cache[layer_id][batch_indices, :, neg_indices, :] 

            pos_keys = cache_positive.key_cache[layer_id][batch_indices, :, pos_indices, :]
            neg_keys = cache_negative.key_cache[layer_id][batch_indices, :, neg_indices, :]

            # Take the differnece between the vectors
            if steering_config.take_difference: # True
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values - neg_values]) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys - neg_keys])
            else:
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values], dim=0) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys], dim=0)
                # activations[layer_id] = torch.cat([activations[layer_id], pos_activations], dim=0)
   
    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values:
            steering_values[layer_id] = torch.mean(steering_values[layer_id], dim=0, keepdim=True) # [n_heads, head_dim]
            steering_keys[layer_id] = torch.mean(steering_keys[layer_id], dim=0, keepdim=True)

        output_dict["values"] = dict(steering_values)
        output_dict["keys"] = dict(steering_keys)

    elif steering_config.aggregation_method == AggregationMethods.pca:
        for layer_id in steering_values:
            values_data = steering_values[layer_id] # [num_demos, H, Dh]
            num_demos, n_heads, head_dim = values_data.shape
            values_reshaped = values_data.view(num_demos, -1)
            pca_values = PCA(n_components=10).to(values_reshaped.device).fit(values_reshaped.float())
            direction_values = (pca_values.components_.sum(dim=0,keepdim=True) + pca_values.mean_).mean(0).view(1, n_heads, head_dim)
            steering_values[layer_id] = direction_values
            keys_data = steering_keys[layer_id]
            keys_reshaped = keys_data.view(num_demos,-1)
            pca_keys = PCA(n_components=10).to(keys_reshaped.device).fit(keys_reshaped.float())
            direction_keys = (pca_keys.components_.sum(dim=0,keepdim=True) + pca_keys.mean_).mean(0).view(1, n_heads, head_dim)
            steering_keys[layer_id] = direction_keys


        output_dict["values"] = dict(steering_values)
        output_dict["keys"] = dict(steering_keys)
    
    return output_dict


# @torch.no_grad()
def extract_steering_kv_text_qwenVLchat(
    model,
    tokenizer,
    contrast_txt,
    steering_config: SteeringConfig,
    device="cpu",
):

    steering_values = defaultdict(lambda: torch.tensor([]).to(device))
    steering_keys = defaultdict(lambda: torch.tensor([]).to(device))
    output_dict = {}
    for example_id in tqdm(range(len(contrast_txt)), desc='Obtaining textual direction'):

        neg_tokens, pos_tokens = contrast_txt[example_id]

        # Select the indices of the last token in the sequence
        if steering_config.extraction_method == "last_token": # True
            subtract_index = 1
        elif steering_config.extraction_method == "penultimate_token":
            subtract_index = 2
        elif isinstance(steering_config.extraction_method, int):
            subtract_index = steering_config.extraction_method
        else:
            raise ValueError(f"Invalid value provided for extraction_method: {steering_config.extraction_method}. Valid values are ['last_token', 'penultimate_token'] or integers.")

        pos_indices = torch.tensor(pos_tokens.shape[1] - subtract_index).cuda() 
        neg_indices = torch.tensor(neg_tokens.shape[1] - subtract_index).cuda() 
        batch_indices = torch.arange(pos_tokens.size(0), device=pos_tokens.device)
        # Log the tokens that are used to extract the steering vectors
        if steering_config.verbose:
            extraction_token_pos = tokenizer.batch_decode(pos_tokens[batch_indices, pos_indices], skip_special_tokens=True)
            extraction_token_neg = tokenizer.batch_decode(neg_tokens[batch_indices, neg_indices], skip_special_tokens=True)
            print(f"Extracting steering vectors for the tokens '{extraction_token_pos}' and '{extraction_token_neg}'")
            
        # Run the model with the hook to cache activations
        cache_positive, cache_negative = DynamicCache(), DynamicCache()
        with torch.no_grad():
            pos_kwargs = {'input_ids': pos_tokens}
            neg_kwargs = {'input_ids': neg_tokens}
            
            pos_out = model(**pos_kwargs, output_hidden_states=True,  use_cache=True)
            cache_positive =  pos_out['past_key_values']
            neg_out = model(**neg_kwargs, output_hidden_states=True,  use_cache=True)
            cache_negative =  neg_out['past_key_values']



        for layer_id in range(len(cache_positive)): # 0-32
            pos_values = cache_positive[layer_id][1][batch_indices, pos_indices, :, :] 
            neg_values = cache_negative[layer_id][1][batch_indices, neg_indices, :, :] 

            pos_keys = cache_positive[layer_id][0][batch_indices, pos_indices, :, :]
            neg_keys = cache_negative[layer_id][0][batch_indices, neg_indices, :, :]

            # Take the differnece between the vectors
            if steering_config.take_difference: # True
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values - neg_values]) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys - neg_keys])
                # activations[layer_id] = torch.cat([activations[layer_id], pos_activations - neg_activations])
            else:
                steering_values[layer_id] = torch.cat([steering_values[layer_id], pos_values], dim=0) # [batch_size, n_heads, head_dim]
                steering_keys[layer_id] = torch.cat([steering_keys[layer_id], pos_keys], dim=0)
                # activations[layer_id] = torch.cat([activations[layer_id], pos_activations], dim=0)
    
    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values:
            steering_values[layer_id] = torch.mean(steering_values[layer_id], dim=0) # [n_heads, head_dim]
            steering_keys[layer_id] = torch.mean(steering_keys[layer_id], dim=0)

        output_dict["values"] = dict(steering_values)
        output_dict["keys"] = dict(steering_keys)

    elif steering_config.aggregation_method == AggregationMethods.pca:
        for layer_id in steering_values:
            values_data = steering_values[layer_id] # [num_demos, H, Dh]
            num_demos, n_heads, head_dim = values_data.shape
            values_reshaped = values_data.view(num_demos, -1)
            
            pca_values = PCA(n_components=1).to(values_reshaped.device).fit(values_reshaped.float())
            direction_values = (pca_values.components_.sum(dim=0,keepdim=True) + pca_values.mean_).mean(0).view(1, n_heads, head_dim)

            steering_values[layer_id] = direction_values
            
            keys_data = steering_keys[layer_id]
            keys_reshaped = keys_data.view(num_demos,-1)
            
            pca_keys = PCA(n_components=1).to(keys_reshaped.device).fit(keys_reshaped.float())
            direction_keys = (pca_keys.components_.sum(dim=0,keepdim=True) + pca_keys.mean_).mean(0).view(1, n_heads, head_dim)
            steering_keys[layer_id] = direction_keys

        output_dict["values"] = dict(steering_values)
        output_dict["keys"] = dict(steering_keys)

    return output_dict

def extract_steering_kv_img(
    model,
    tokenizer,
    # data,
    input_images,
    contrast_images,
    input_ids,
    contrast_ids,
    steering_config: SteeringConfig,
    mask_ratio=0.9,
    # batch_size=1,
    device="cpu",
):

    steering_values_img = defaultdict(lambda: torch.tensor([]).to(device))
    steering_keys_img = defaultdict(lambda: torch.tensor([]).to(device))
    activations_img = defaultdict(lambda: torch.tensor([]))
    output_dict_img = {}
    img_token_start = 34
    img_token_len = 576
    img_token_slice = slice(img_token_start, img_token_start + img_token_len) 
    top_k=100
    pos_attn_demo = []
    neg_attn_demo = []

    with torch.no_grad():
        for example_id in tqdm(range(len(contrast_images)), desc='Obtaining visual direction'):
            # example.keys() = dict_keys(['steps', 'id', 'question', 'answer', 'positive', 'negative'])
            images = input_images[example_id]

            neg_tokens, pos_tokens = contrast_ids[example_id]
            input_tokens = input_ids[example_id]
            neg_images, pos_images = contrast_images[example_id]
            # Find indices of the tokens to extract the steering vectors from
        
            batch_indices = torch.arange(neg_images.size(0), device=neg_images.device)
    
            # cache_positive_text, cache_negative_text = DynamicCache(), DynamicCache()
            cache_positive_img, cache_negative_img = DynamicCache(), DynamicCache()
            pos_kwargs_img = {'input_ids': input_tokens, 'images': pos_images.unsqueeze(0).half().to(input_tokens.device)}
            neg_kwargs_img = {'input_ids': input_tokens, 'images': neg_images.unsqueeze(0).half().to(input_tokens.device)}

            pos_out_img = model(**pos_kwargs_img, output_hidden_states=True,  past_key_values=cache_positive_img)
            neg_out_img = model(**neg_kwargs_img, output_hidden_states=True,  past_key_values=cache_negative_img)

 
            for layer_id in range(len(cache_positive_img.value_cache)): # 0-32
                pos_values_img = cache_positive_img.value_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2) # torch.Size([1, 32,128])
                neg_values_img = cache_negative_img.value_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2)  # torch.Size([1, 32, 128])
                pos_keys_img = cache_positive_img.key_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2) 
                neg_keys_img = cache_negative_img.key_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2) 

                # Take the differnece between the vectors
                if steering_config.take_difference: # True
                    steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], pos_values_img - neg_values_img]) # [batch_size, n_heads, head_dim]
                    # steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], neg_values_img - pos_values_img]) # [batch_size, n_heads, head_dim]
                    steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], pos_keys_img - neg_keys_img])
                    # activations_img[layer_id] = torch.cat([activations_img[layer_id], pos_activations_img - neg_activations_img])
                else:
                    steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], pos_values_img], dim=0) # [batch_size, n_heads, img_tokens, head_dim]
                    steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], pos_keys_img], dim=0)
                    # activations_img[layer_id] = torch.cat([activations_img[layer_id], pos_activations_img], dim=0)  
                    #    torch.Size([100, 2, 25, 577, 1024])     100, 2, 32, 576, 32*128


    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values_img:
            steering_values_img[layer_id] = torch.mean(steering_values_img[layer_id], dim=0) # [n_heads, head_dim]
            steering_keys_img[layer_id] = torch.mean(steering_keys_img[layer_id], dim=0)

        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    elif steering_config.aggregation_method == AggregationMethods.pca:
        evr_values_dict = {}   # {layer_id: EVR1}
        evr_keys_dict   = {}   # {layer_id: EVR1}
        headimp_values_cols = []  # list of (layer_id, tensor[H])
        headimp_keys_cols   = []  # list of (layer_id, tensor[H])

        for layer_id in steering_values_img:
            values_data_img = steering_values_img[layer_id] # [num_demos, H, Dh]
            num_demos,  n_heads, head_dim = values_data_img.shape
            values_reshaped = values_data_img.view(num_demos, -1)
            pca_values_img = PCA(n_components=1).to(values_reshaped.device).fit(values_reshaped.float())
            direction_values = (pca_values_img.components_.sum(dim=0,keepdim=True) + pca_values_img.mean_).mean(0).view(1, n_heads, head_dim)
            steering_values_img[layer_id] = direction_values


            keys_data_img = steering_keys_img[layer_id]
            keys_reshaped = keys_data_img.view(num_demos,-1)
            pca_keys_img = PCA(n_components=1).to(keys_reshaped.device).fit(keys_reshaped.float())
            direction_keys = (pca_keys_img.components_.sum(dim=0,keepdim=True) + pca_keys_img.mean_).mean(0).view(1, n_heads, head_dim)
            steering_keys_img[layer_id] = direction_keys


        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    
    return output_dict_img


def extract_steering_kv_img_deepseek(
    model, 
    inputs_img, 
    mask_img, 
    steering_config: SteeringConfig,
    img_start_idx = 8,
    img_end_idx = 40,
    device="cpu",
):

    steering_values_img = defaultdict(lambda: torch.tensor([]).to(device))
    steering_keys_img = defaultdict(lambda: torch.tensor([]).to(device))
    output_dict_img = {}
    
    img_token_slice = slice(img_start_idx, img_end_idx) 

    # for example in tqdm(data.iter(batch_size=batch_size)):
    with torch.no_grad():
        for example_id in tqdm(range(len(inputs_img)), desc='Obtaining visual direction'):
            # example.keys() = dict_keys(['steps', 'id', 'question', 'answer', 'positive', 'negative'])

            neg_tokens, pos_tokens = inputs_img[example_id]
            neg_masks, pos_masks = mask_img[example_id]
        
            batch_indices = torch.arange(neg_tokens.size(0), device=neg_tokens.device)
    
            # cache_positive_text, cache_negative_text = DynamicCache(), DynamicCache()
            cache_positive_img, cache_negative_img = DynamicCache(), DynamicCache()
            pos_kwargs = {'inputs_embeds': pos_tokens.to('cuda'), 'attention_mask': pos_masks.to('cuda')}
            neg_kwargs = {'inputs_embeds': neg_tokens.to('cuda'), 'attention_mask': neg_masks.to('cuda')}


            pos_out_img = model(**pos_kwargs, output_hidden_states=True,  past_key_values=cache_positive_img)
            neg_out_img = model(**neg_kwargs, output_hidden_states=True,  past_key_values=cache_negative_img)

           
            for layer_id in range(len(cache_positive_img.value_cache)): # 0-32
                pos_values_img = cache_positive_img.value_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2) # torch.Size([1, 32,128])
                neg_values_img = cache_negative_img.value_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2)  # torch.Size([1, 32, 128])
                pos_keys_img = cache_positive_img.key_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2) 
                neg_keys_img = cache_negative_img.key_cache[layer_id][batch_indices, :, img_token_slice, :].mean(2) 

                # Take the differnece between the vectors
                if steering_config.take_difference: # True
                    steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], pos_values_img - neg_values_img]) # [batch_size, n_heads, head_dim]
                    steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], pos_keys_img - neg_keys_img])
                else:
                    steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], pos_values_img], dim=0) # [batch_size, n_heads, img_tokens, head_dim]
                    steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], pos_keys_img], dim=0)

            
    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values_img:
            steering_values_img[layer_id] = torch.mean(steering_values_img[layer_id], dim=0, keepdim=True) # [n_heads, head_dim]
            steering_keys_img[layer_id] = torch.mean(steering_keys_img[layer_id], dim=0, keepdim=True)

        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    elif steering_config.aggregation_method == AggregationMethods.pca:
        for layer_id in steering_values_img:
            values_data_img = steering_values_img[layer_id] # [num_demos, H, Dh]
            num_demos,  n_heads, head_dim = values_data_img.shape
            values_reshaped = values_data_img.view(num_demos, -1)
            pca_values_img = PCA(n_components=1).to(values_reshaped.device).fit(values_reshaped.float())
            # pdb.set_trace()
            direction_values = (pca_values_img.components_.sum(dim=0,keepdim=True) + pca_values_img.mean_).mean(0).view(1, n_heads, head_dim)
            steering_values_img[layer_id] = direction_values


            keys_data_img = steering_keys_img[layer_id]
            keys_reshaped = keys_data_img.view(num_demos,-1)
            pca_keys_img = PCA(n_components=1).to(keys_reshaped.device).fit(keys_reshaped.float())
            # direction_keys = pca_keys_img.components_[0].view(1, n_heads, head_dim)        
            direction_keys = (pca_keys_img.components_.sum(dim=0,keepdim=True) + pca_keys_img.mean_).mean(0).view(1, n_heads, head_dim)
            steering_keys_img[layer_id] = direction_keys

        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    return output_dict_img

def extract_steering_kv_img_qwenVLchat(
    model,
    contrast_images,
    steering_config: SteeringConfig,
    img_token_start = 37,
    img_token_end = 295,
    device="cpu",
):

    steering_values_img = defaultdict(lambda: torch.tensor([]).to(device))
    steering_keys_img = defaultdict(lambda: torch.tensor([]).to(device))
    output_dict_img = {}
    img_token_slice = slice(img_token_start+1, img_token_end-1) 

    # for example in tqdm(data.iter(batch_size=batch_size)):
    with torch.no_grad():
        for example_id in tqdm(range(len(contrast_images)), desc='Obtaining visual direction'):
            # example.keys() = dict_keys(['steps', 'id', 'question', 'answer', 'positive', 'negative'])
            neg_tokens, pos_tokens = contrast_images[example_id]
            batch_indices = torch.arange(neg_tokens.size(0), device=neg_tokens.device)
    
            cache_positive_img, cache_negative_img = DynamicCache(), DynamicCache()
            pos_kwargs_img = {'input_ids': pos_tokens}
            neg_kwargs_img = {'input_ids': neg_tokens}

            pos_out_img = model(**pos_kwargs_img, output_hidden_states=True,  use_cache=True)
            cache_positive_img =  pos_out_img['past_key_values']
            neg_out_img = model(**neg_kwargs_img, output_hidden_states=True,  use_cache=True)
            cache_negative_img =  neg_out_img['past_key_values']

            for layer_id in range(len(cache_positive_img)): # 0-32
                
                pos_values_img = cache_positive_img[layer_id][1][batch_indices, img_token_slice, :, :].mean(1)  # torch.Size([1, 32, 128])
                neg_values_img = cache_negative_img[layer_id][1][batch_indices, img_token_slice, :, :].mean(1)  # torch.Size([1, 32, 128])

                pos_keys_img = cache_positive_img[layer_id][0][batch_indices, img_token_slice, :, :].mean(1) 
                neg_keys_img = cache_negative_img[layer_id][0][batch_indices, img_token_slice, :, :].mean(1) 
             
                # Take the differnece between the vectors
                if steering_config.take_difference: # True
                    steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], pos_values_img - neg_values_img]) # [batch_size, n_heads, head_dim]
                    steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], pos_keys_img - neg_keys_img])
                else:
                    steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], pos_values_img], dim=0) # [batch_size, n_heads, img_tokens, head_dim]
                    steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], pos_keys_img], dim=0)
          
    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values_img:
            steering_values_img[layer_id] = torch.mean(steering_values_img[layer_id], dim=0) # [n_heads, head_dim]
            steering_keys_img[layer_id] = torch.mean(steering_keys_img[layer_id], dim=0)

        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    elif steering_config.aggregation_method == AggregationMethods.pca:
        for layer_id in steering_values_img:
            values_data_img = steering_values_img[layer_id] # [num_demos, H, Dh]
            num_demos,  n_heads, head_dim = values_data_img.shape
            values_reshaped = values_data_img.view(num_demos, -1)
            pca_values_img = PCA(n_components=1).to(values_reshaped.device).fit(values_reshaped.float())
            # pdb.set_trace()
            direction_values = (pca_values_img.components_.sum(dim=0,keepdim=True) + pca_values_img.mean_).mean(0).view(1, n_heads, head_dim)
            steering_values_img[layer_id] = direction_values



            keys_data_img = steering_keys_img[layer_id]
            keys_reshaped = keys_data_img.view(num_demos,-1)
            pca_keys_img = PCA(n_components=1).to(keys_reshaped.device).fit(keys_reshaped.float())
            # direction_keys = pca_keys_img.componets_[0].view(1, n_heads, head_dim)        
            direction_keys = (pca_keys_img.components_.sum(dim=0,keepdim=True) + pca_keys_img.mean_).mean(0).view(1, n_heads, head_dim)
            steering_keys_img[layer_id] = direction_keys

        
        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    return output_dict_img

def generate_with_cache_steering(
    model: PreTrainedModel,
    # tokens,
    steering_kv,
    task,
    steering_config: SteeringConfig,
    args,
    output_full_dict=False,
    stopping_criteria=None,
    mask_id=None,
    mask_shape=None,
    seg_dict=None, 
    coco=None,
    **kwargs,
):
    # pdb.set_trace()

    if 'llava' in args.model:
        tokens = kwargs["input_ids"]
    elif 'minigpt' in args.model:
        tokens = kwargs["inputs_embeds"]


    if steering_config.how == "last":
        application_token_idx = -1
    elif isinstance(steering_config.how, int):
        application_token_idx = steering_config.how
    else:
        raise ValueError(f"Invalid value provided for how: {steering_config.how}. Valid values are ['last'] or integers.")

    # Append a special token to the input tokens if needed to be able to steer the cache of last token
    if steering_config.append_special_token and application_token_idx == -1:
        # token_to_append = get_token_to_append(steering_config, tokens, task=task) # torch.Size([1, 1])
        token_to_append = tokens[:, -1:] # torch.Size([1, 1])
        # print(f"Appending special token '{steering_config.tokenizer.decode(token_to_append[0].item())}' to the input tokens.")
        tokens = torch.cat([tokens, token_to_append], dim=-2)
        # kwargs["input_ids"] = tokens
        kwargs["inputs_embeds"] = tokens
        if "attention_mask" in kwargs:
            token_to_append = kwargs['attention_mask'][:, -1:]
            kwargs['attention_mask'] = torch.cat([kwargs['attention_mask'], torch.ones_like(token_to_append)], dim=-1)
        

    # Log the tokens that the steering is applied to
    if steering_config.verbose:
        decoded_last_tokens = steering_config.tokenizer.decode(tokens[:, application_token_idx-1]) 
        logger.debug(f"Applying steering to the cache of the following tokens '{decoded_last_tokens}'")

        print(f"Applying steering to the cache of the following tokens '{decoded_last_tokens}'")

    # Create the initial cache
    if 'llava' in args.model:
        cache_input = {
            "input_ids":  kwargs["input_ids"],
            'images': kwargs['images'],
        }
        last_input_before = kwargs
        last_input = kwargs

    elif 'qwen-vl-chat' in args.model :
        cache_input = {
            "input_ids":  kwargs["input_ids"],
        }

    elif 'deepseek-vl-chat' in args.model :
        cache_input = {
            "inputs_embeds":   kwargs["prefix_embeds"],
            'attention_mask': kwargs['attention_mask'],
        }
        last_input = {
            "input_ids":   kwargs["input_ids"],
            'attention_mask': kwargs['attention_mask'],
            'pad_token_id': kwargs['pad_token_id'],
            'bos_token_id': kwargs['bos_token_id'],
            'eos_token_id': kwargs['eos_token_id'],
        }
        last_input_before = {
            'inputs_embeds': kwargs['inputs_embeds'],
            'attention_mask': kwargs['attention_mask'],
            'pad_token_id': kwargs['pad_token_id'],
            'bos_token_id': kwargs['bos_token_id'],
            'eos_token_id': kwargs['eos_token_id'],
        }
        try:
            last_input['top_p'] = kwargs['top_p']
            last_input['top_k'] = kwargs['top_k']
            last_input_before['top_p'] = kwargs['top_p']
            last_input_before['top_k'] = kwargs['top_k']
        except:
            pass
        
        try:
            last_input['logits_processor'] = kwargs['logits_processor']
        except:
            pass

        kwargs = last_input


    if 'llava' in args.model or 'minigpt4' in args.model or 'deepseek' in args.model:
        cache_type = 'class'
    # elif 'qwen' in args.model or 'minigpt' in args.model:
    else:
        cache_type = 'tuple'
        
    past_key_values = precompute_kv_cache(model, cache_input, cache_type)

    # # # Steer the cache
    past_key_values = steer_kv_cache(
        past_key_values,
        steering_kv,
        steering_config,
        application_token_idx=application_token_idx,
        steer_type='both',
        cache_type=cache_type,
        args=args,
        # steer_type='image',
        # steer_type='text',
    )
    
    # repeat_past_for_beams
    num_beams = args.num_beams  # = 5
    if num_beams > 1:
        past_key_values = repeat_past_for_beams(past_key_values, num_beams)

    try:
        # Generate
        # output.keys() odict_keys(['sequences', 'past_key_values'])
        output = model.generate(
                        do_sample=args.do_sample,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        temperature=args.temperature,
                        return_dict=True,
                        stopping_criteria=stopping_criteria, 
                        # output_attentions= True,
                        # output_scores=True,
                        # return_dict_in_generate=True,
                        past_key_values=past_key_values,
                        # output_hidden_states=False,
                        output_hidden_states=True if args.logits_aug else False,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        repetition_penalty=args.repetition_penalty,
                        # **last_input, 
                        **kwargs, 
                    )
        
        if output_full_dict:
            return output

        output_tokens = output.sequences

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise e

    return output_tokens


def precompute_kv_cache(model, cache_input, cache_type = 'class'):
    """
    Precompute the key and value caches for the input tokens except the last one.
    tokens: ["input_ids", "images"]
    """

    tokens = cache_input.copy()
    past_key_values = DynamicCache()
    
    if isinstance(tokens, BatchEncoding) or isinstance(tokens, dict):    
        for k, v in tokens.items():
            if k in ["input_ids",]: 
                tokens[k] = v[:, :-1]
    
    with torch.no_grad():
        if cache_type == 'class':
            model(**tokens, output_hidden_states=True, past_key_values=past_key_values, use_cache=True)
        else:
            out = model(**tokens, output_hidden_states=True, use_cache=True)
            # past_key_values = DynamicCache.from_legacy_cache(out['past_key_values'])
            past_key_values = out['past_key_values']
        # past_key_values.key_cache[0].shape  out['past_key_values'][0][0].shape  past_cache.key_cache[0].shape
        # past_cache = DynamicCache.from_legacy_cache(out['past_key_values'])
    return past_key_values


def steer_kv_cache(cache, steering_kv, steering_config, application_token_idx=-1, steer_type='text', cache_type = 'class', args=None):

    steering_kv_text, steering_kv_img = steering_kv


    if steer_type == 'text':
        if "values" in steering_kv_text:
            # Steer the values cache
            for layer_idx, past_values in steering_kv_text["values"].items():
                steer_kv_cache_layer(cache, past_values, steering_config, layer_idx, type='values', application_token_idx=application_token_idx)
        if "keys" in steering_kv_text:
            # Steer the keys cache
            for layer_idx, past_keys in steering_kv_text["keys"].items():
                steer_kv_cache_layer(cache, past_keys, steering_config, layer_idx, type='keys', application_token_idx=application_token_idx)
    elif steer_type == 'image':
        if "values" in steering_kv_img:
            # Steer the values cache
            for layer_idx, past_values in steering_kv_img["values"].items():
                steer_kv_cache_layer_img(cache, past_values, steering_config, layer_idx, type='values', application_token_idx=args.img_start_idx, img_tokens=args.img_end_idx - args.img_start_idx)


        if "keys" in steering_kv_img:
            # Steer the keys cache
            for layer_idx, past_keys in steering_kv_img["keys"].items():
                steer_kv_cache_layer_img(cache, past_keys, steering_config, layer_idx, type='keys', application_token_idx=args.img_start_idx, img_tokens=args.img_end_idx - args.img_start_idx)

    else:
        if "values" in steering_kv_text:
            # Steer the values cache
            for layer_idx, past_values in steering_kv_text["values"].items():
                steer_kv_cache_layer(cache, past_values, steering_config, layer_idx, type='values', application_token_idx=application_token_idx, cache_type=cache_type)
        if "keys" in steering_kv_text:
            # Steer the keys cache
            for layer_idx, past_keys in steering_kv_text["keys"].items():
                steer_kv_cache_layer(cache, past_keys, steering_config, layer_idx, type='keys', application_token_idx=application_token_idx, cache_type=cache_type)

        # pdb.set_trace()

        if "values" in steering_kv_img:
            # Steer the values cache
            for layer_idx, past_values in steering_kv_img["values"].items():
                steer_kv_cache_layer_img(cache, past_values, steering_config, layer_idx, type='values', application_token_idx=args.img_start_idx, img_tokens=int(args.img_end_idx - args.img_start_idx), cache_type=cache_type)
                
        if "keys" in steering_kv_img:
            # Steer the keys cache
            for layer_idx, past_keys in steering_kv_img["keys"].items():
                steer_kv_cache_layer_img(cache, past_keys, steering_config, layer_idx, type='keys', application_token_idx=args.img_start_idx, img_tokens=int(args.img_end_idx - args.img_start_idx), cache_type=cache_type)

    return cache


def steer_kv_cache_layer(cache, steering_vector, steering_config, layer_idx, type='values', application_token_idx=-1, cache_type = 'class'):
    """
    Steer the key and value cache of a specific layer.
    """
    sv = steering_vector.clone() # [n_heads, head_dim]

    if  cache_type == 'class':
        if type == 'values':
            cav = cache.value_cache[layer_idx][:, :, application_token_idx, :].clone() 
            cache.value_cache[layer_idx][:, :, application_token_idx, :] = steer_kv_cache_add(cav, sv, steering_config.txt_values, layer_idx)

        elif type == 'keys':
            cak = cache.key_cache[layer_idx][:, :, application_token_idx, :].clone() 
            cache.key_cache[layer_idx][:, :, application_token_idx, :] = steer_kv_cache_add(cak, sv, steering_config.txt_keys, layer_idx)
    
    elif  cache_type == 'tuple':
        if type == 'values':
            cav = cache[layer_idx][1][:, application_token_idx, :, :].clone() 
            cache[layer_idx][1][:, application_token_idx, :, :] = steer_kv_cache_add(cav, sv, steering_config.txt_values, layer_idx)

        elif type == 'keys':
            cak = cache[layer_idx][0][:, application_token_idx, :, :].clone() 
            cache[layer_idx][0][:, application_token_idx, :, :] = steer_kv_cache_add(cak, sv, steering_config.txt_keys, layer_idx)
    


def steer_kv_cache_layer_img(cache, steering_vector, steering_config, layer_idx, type='values', application_token_idx=34, img_tokens=576, cache_type = 'class'):
    """
    Steer the key and value cache of a specific layer.
    """
    sv = steering_vector.clone() 
    img_token_slice = slice(application_token_idx, application_token_idx + img_tokens)  


    # Apply the vector to the cache
    if  cache_type == 'class':
        if type == 'values':
            cav = cache.value_cache[layer_idx][:, :, img_token_slice , :].clone()  
            cache.value_cache[layer_idx][:, :, img_token_slice, :] = steer_kv_cache_add(cav, sv, steering_config.img_values)

        elif type == 'keys':
            cak = cache.key_cache[layer_idx][:, :, img_token_slice, :].clone() 
            cache.key_cache[layer_idx][:, :, img_token_slice, :] = steer_kv_cache_add(cak, sv, steering_config.img_keys)

    elif  cache_type == 'tuple':

        if type == 'values':
            cav = cache[layer_idx][1][:, img_token_slice, :, :].clone() 
            cache[layer_idx][1][:, img_token_slice, :, :] = steer_kv_cache_add(cav, sv, steering_config.img_values, token_dim=1)

        elif type == 'keys':
            cak = cache[layer_idx][0][:, img_token_slice, :, :].clone() 
            cache[layer_idx][0][:, img_token_slice, :, :] = steer_kv_cache_add(cak, sv, steering_config.img_keys, token_dim=1)
    


def steer_kv_cache_add(cache, steering_vector, lam, layer_idx=0, token_dim=2):

    original_norm = torch.norm(cache.float(), p=2, dim=-1, keepdim=True)  
    if len(original_norm.shape) == 4 and len(steering_vector.shape) != 4:
        if token_dim == 2:
            sv_broadcast = steering_vector.unsqueeze(2).repeat(1, 1, cache.shape[2], 1)  
        elif token_dim == 1:
            sv_broadcast = steering_vector.unsqueeze(1).repeat(1, cache.shape[1], 1, 1)  

    else:
        sv_broadcast = steering_vector
    
    cos_sim = F.cosine_similarity(cache.float(), -sv_broadcast, dim=-1)
    lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(cache.device), cos_sim).unsqueeze(-1)

    y = lam * lambda_sim * F.normalize(sv_broadcast, dim=-1) 
    
    cache_norm = F.normalize(cache.float(), p=2, dim=-1) 

    x = F.normalize(cache_norm + y, p=2, dim=-1) * original_norm  

    return x.half()  

