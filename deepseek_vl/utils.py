
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
# import kornia
from transformers import set_seed
import re
import random
from .pca import PCA
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import List, Tuple

# from cache_utils.cache_steer import remove_words_from_desc

def process_image(image_processor, image_raw):
    answer = image_processor(image_raw)

    # Check if the result is a dictionary and contains 'pixel_values' key

    try:
        if 'pixel_values' in answer:
            answer = answer['pixel_values'][0]
    except:
        pass
    
    # Convert numpy array to torch tensor if necessary
    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    
    # If it's already a tensor, return it directly
    elif isinstance(answer, torch.Tensor):
        return answer
    
    else:
        raise ValueError("Unexpected output format from image_processor.")
    
    return answer

def remove_words_from_desc(object_list, desc):
    # 按短语长度（单词数量）排序，确保长短语先被处理，避免被拆分
    sorted_objects = sorted(object_list, key=lambda x: len(x.split()), reverse=True)
    
    processed_desc = desc
    for item in sorted_objects:
        # 构建正则表达式，确保匹配整个单词/短语
        # 使用\b作为单词边界，re.escape处理特殊字符
        pattern = r'\b' + re.escape(item) + r'\b'
        processed_desc = re.sub(pattern, '', processed_desc)

    # 处理多余的空格：多个空格合并为一个，去除首尾空格
    processed_desc = re.sub(r'\s+', ' ', processed_desc).strip()
    
    # 处理标点符号前的空格（如 " ." 变为 "."）
    processed_desc = re.sub(r' (\.)', r'\1', processed_desc)
    
    return processed_desc

def mask_patches(tensor, indices, patch_size=14):
    """
    Creates a new tensor where specified patches are set to the mean of the original tensor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    indices (list of int): Indices of the patches to modify
    patch_size (int): Size of one side of the square patch
    
    Returns:
    torch.Tensor: New tensor with modified patches
    """
    # Clone the original tensor to avoid modifying it
    new_tensor = tensor.clone()

    # Calculate the mean across the spatial dimensions
    mean_values = tensor.mean(dim=(1, 2), keepdim=True)
    
    # Number of patches along the width
    patches_per_row = tensor.shape[2] // patch_size
    total_patches = (tensor.shape[1] // patch_size) * (tensor.shape[2] // patch_size)


    for index in indices:
        # Calculate row and column position of the patch
        row = index // patches_per_row
        col = index % patches_per_row

        # Calculate the starting pixel positions
        start_x = col * patch_size
        start_y = row * patch_size

        # Replace the patch with the mean values
        new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = mean_values.expand(-1, patch_size, patch_size)#new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size].mean(dim=(1, 2), keepdim=True).expand(-1, patch_size, patch_size)# mean_values.expand(-1, patch_size, patch_size)

    return new_tensor


def get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=True, image_processor=None):
    if model_is_llaval:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        qs_pos = question 
        qs_neg = question

        if hasattr(model.config, 'mm_use_im_start_end'):

            if model.config.mm_use_im_start_end:
                qs_pos = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_pos
            else:
                qs_pos = DEFAULT_IMAGE_TOKEN + '\n' + qs_pos

            if model.config.mm_use_im_start_end:
                qs_neg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_neg
            else:
                qs_neg = DEFAULT_IMAGE_TOKEN + '\n' + qs_neg

            conv_pos = conv_templates[args.conv_mode].copy()
            conv_pos.append_message(conv_pos.roles[0], qs_pos)
            conv_pos.append_message(conv_pos.roles[1], None)
            conv_neg = conv_templates[args.conv_mode].copy()
            conv_neg.append_message(conv_neg.roles[0], qs_neg)
            conv_neg.append_message(conv_neg.roles[1], None)


            prompts_positive  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            prompts_negative  = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]

            # prompts_negative  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            # prompts_positive  = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]

            # prompts_positive  = [conv_pos.get_prompt() + ", ".join(k['co_objects'] + k['uncertain_objects']) + '.' for k in data_demos]
            # prompts_positive  = [conv_pos.get_prompt() + remove_words_from_desc(k['co_objects'] + k['uncertain_objects'], k['value'])  for k in data_demos]
            # prompts_negative  = [conv_neg.get_prompt() + k['value'] for k in data_demos]

            # prompts_positive  = [conv_pos.get_prompt() + ", ".join(k['Object']) + '.' for k in data_demos]
            # prompts_negative  = [conv_pos.get_prompt() + remove_words_from_desc(k['Object'], k['paragraph'])  for k in data_demos]
            # prompts_positive  = [conv_pos.get_prompt() + remove_words_from_desc(k['Object'], k['paragraph'])  for k in data_demos]
            # prompts_negative  = [conv_neg.get_prompt() + k['paragraph'] for k in data_demos]

            # pdb.set_trace()
           
            # prompts_positive, prompts_negative = [], []
            # for data in tqdm(data_demos, desc="Process postives and negatives:"):
            #     positives = data['Object']
            #     desc = data['desc']
            #     # 执行删除操作
            #     negatives = remove_words_from_desc(positives, desc) 
            #     prompts_positive.append(conv_pos.get_prompt() + ", ".join(positives) + '.')
            #     prompts_negative.append(conv_neg.get_prompt() + negatives)
                # pdb.set_trace()

            input_ids_positive = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_positive]
            input_ids_negative = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_negative]

            inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
            inputs = tuple(inputs)

        else:
            # from transformers import InstructBlipProcessor
            # processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

            # input_ids_positive = []
            # input_ids_negative = []

            # for k in data_demos:
            #     image_path = os.path.join(args.data_file, 'train2014', k['image'])

            #     image_raw = Image.open(image_path).convert("RGB")
            #     input_ids_positive.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt").to(model.device))
            #     input_ids_negative.append(processor(images=image_raw, text=question + k['h_value'], return_tensors="pt").to(model.device))

            # deepsesk
            from deepseek_vl.utils.io import load_pil_images
            input_embeds_positive = []
            input_embeds_negative = []
            for k in data_demos:
                image_path = os.path.join(args.data_file, 'train2014', k['image'])
                conversation_pos = [
                    {
                        "role": "User",
                        "content": f"<image_placeholder>{question}"+k['value'],
                        "images": [f"{image_path}"],
                    },
                    {"role": "Assistant", "content": ""},
                ]
                conversation_neg = [
                    {
                        "role": "User",
                        "content": f"<image_placeholder>{question}"+k['value'],
                        "images": [f"{image_path}"],
                    },
                    {"role": "Assistant", "content": ""},
                ]

                pil_images = load_pil_images(conversation_pos)
                prepare_inputs_pos = image_processor(conversations=conversation_pos, images=pil_images, force_batchify=True).to('cuda')
                inputs_embeds_pos = model.prepare_inputs_embeds(**prepare_inputs_pos)

                prepare_inputs_neg = image_processor(conversations=conversation_neg, images=pil_images, force_batchify=True).to('cuda')
                inputs_embeds_neg = model.prepare_inputs_embeds(**prepare_inputs_neg)

                input_embeds_positive.append({"inputs_embeds":inputs_embeds_pos, "attention_mask":prepare_inputs_pos['attention_mask']})
                input_embeds_negative.append({"inputs_embeds":inputs_embeds_neg, "attention_mask":prepare_inputs_neg['attention_mask']})

            inputs = [(input_embeds_negative[demo_id], input_embeds_positive[demo_id]) for demo_id in range(len(input_embeds_positive))]
            inputs = tuple(inputs)
    
    else:
        prompts_positive = []
        prompts_negative = []

        for k in data_demos:
            image_path = os.path.join(args.data_file, 'train2014', k['image'])    
            prompts_positive.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['value']}]))
            prompts_negative.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['h_value']}]))

        input_ids_positive = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_positive]
        input_ids_negative = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_negative]
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    return inputs

def get_demos(args, image_processor, model, tokenizer, patch_size = 14, file_path = '/home/zhangcs/zhangcs/code/VISTA/vti_utils/hallucination_vti_demos.jsonl', model_is_llaval=True): 
    # Initialize a list to store the JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)
    question = "Describe this image in detail." 

    inputs_images = []

    # from pycocotools.coco import COCO
    # annFile = '/home/zhangcs/zhangcs/dataset/coco/annotations/instances_train2014.json'
    # coco = COCO(annFile)
    # with open("/home/zhangcs/zhangcs/code/VISTA/cache_utils/coco2014_train_instances_outputs.jsonl", "r", encoding="utf-8") as f:
    #     seg_dict = json.load(f)

    coco_file = '/home/zhangcs/zhangcs/dataset/coco/'

    for i in tqdm(range(len(data_demos)), desc="Process data demos:"):
        # question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        # image_path = os.path.join(coco_file, 'train2014', data_demos[i]['image_file'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_processor, image_raw)
        image_tensor_cd_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
            image_tensor_cd_all_trials.append(image_tensor_cd)
        inputs_images.append([image_tensor_cd_all_trials, image_tensor])
        
        # if seg_dict:
        #     image_ = np.array(image_raw).transpose(2, 0, 1)  # 转为[C, H, W]
        #     image_ = torch.from_numpy(image_).float()
        #     seg_ann_info = seg_dict[data_demos[i]['image'].split(".")[0]]    
        #     all_mask = torch.zeros(image_[0].shape)
        #     for i, ann in enumerate(seg_ann_info):
        #         mask = coco.annToMask(ann)
        #         mask_tensor = torch.from_numpy(mask).float()
        #         all_mask = torch.max(all_mask, mask_tensor) 
            
        #     mask_tensor = all_mask.repeat(image_.shape[0], 1, 1)
        #     image_object = image_ * mask_tensor
        #     reverse_mask = 1 - mask_tensor
        #     image_background = image_ * reverse_mask
        #     image_object = process_image(image_processor, image_object)
        #     image_background = process_image(image_processor, image_background)

        # inputs_images.append([image_background, image_object])

    
    input_ids = get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=model_is_llaval, image_processor=image_processor)
    
    return inputs_images, input_ids


def get_hiddenstates(model, inputs, image_tensor, model_is_llaval = True):
        h_all = []
        with torch.no_grad():
            for example_id in tqdm(range(len(inputs)), desc="Extract hidden states:"):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    if image_tensor is None:
                        # pdb.set_trace()
                        h = model(**inputs[example_id][style_id], output_hidden_states=True, return_dict=True).hidden_states
                    else:
                        h = model(
                                inputs[example_id][style_id],
                                images=image_tensor[example_id][-1].unsqueeze(0).half(),
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                    
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all


def get_attentions(model, inputs, image_tensor, token_idx = -1):
        h_all = []
        with torch.no_grad():
            for example_id in tqdm(range(len(inputs)), desc="Extract hidden states:"):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    h = model(
                                inputs[example_id][style_id],
                                images=image_tensor[example_id][-1].unsqueeze(0).half(),
                                use_cache=False,
                                output_attentions=True,
                                return_dict=True).attentions
                    embedding_token = []
                    # pdb.set_trace()
                    for layer in range(len(h)):
                        embedding_token.append(torch.sum(h[layer][0,:,token_idx, 34:34+576], dim=-1).detach().cpu())
                    
                    embedding_token = torch.stack(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all

def obtain_textual_vti_attn(model, inputs, image_tensor, rank=1):
    from cache_utils.cache_util import repeat_past_for_beams, visualize_avg_head_attention, vis_key_value_diff, visualize_pos_neg_kv, visualize_pca_for_layer, save_kv_diff, visualize_last_token_attention, vis_attn_change_rate
    for i in [630,  -1]:
        attentions = get_attentions(model, inputs, image_tensor, i)
    
        num_demonstration = len(attentions)  
        neg_all = []
        pos_all = []
        for demonstration_id in range(num_demonstration):
            neg_all.append(attentions[demonstration_id][0])
            pos_all.append(attentions[demonstration_id][1])

        # pdb.set_trace()
        pos_attn_sum = torch.sum(torch.stack(pos_all, dim=0), dim=0)
        neg_attn_sum = torch.sum(torch.stack(neg_all, dim=0), dim=0)
        
        vis_attn_change_rate(pos_attn_sum.detach().cpu(), neg_attn_sum.detach().cpu(), save_path= f"/home/zhangcs/zhangcs/code/VISTA/cache_utils/vis/test_VTI_attn_rate_token_{i}.jpg")
    pdb.set_trace()
    #  torch.Size([1, 1, 135168])   torch.Size([1,  135168])
    return direction, reading_direction


def obtain_textual_vti(model, inputs, image_tensor, rank=1, model_is_llaval=True):
    hidden_states = get_hiddenstates(model, inputs, image_tensor, model_is_llaval = model_is_llaval)
    # hidden_states = get_hiddenstates(model, inputs, image_tensor=None, model_is_llaval = model_is_llaval)

    hidden_states_all = []
    num_demonstration = len(hidden_states)  # torch.Size([33, 4096])
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)  # torch.Size([70, 135168])
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data) 
    # pdb.set_trace()

    #  torch.Size([1, 1, 135168])   torch.Size([1,  135168])
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def average_tuples(tuples: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # Check that the input list is not empty
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    # Check that all tuples have the same length
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the indices of the tuples
    for i in range(n):
        # Stack the tensors at the current index and compute the average
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)

    # Convert the list of averaged tensors to a tuple
    averaged_tuple = tuple(averaged_tensors)

    return averaged_tuple

def get_visual_hiddenstates(model, image_tensor, model_is_llaval=True):
    h_all = []
    with torch.no_grad():
        if model_is_llaval:
            try:
                vision_model = model.model.vision_tower.vision_tower.vision_model
            except:
                vision_model = model.vision_model
        else:
            vision_model = model.transformer.visual
            model.transformer.visual.output_hidden_states = True

            model.low_res_size
            
        for example_id in tqdm(range(len(image_tensor)), desc='Extract visual hidden states:'):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                if isinstance(image_tensor[example_id][style_id], list):  
                    h = []
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        if model_is_llaval:
                            h_ = vision_model(
                                image_tensor_.unsqueeze(0).half().cuda(),
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                        else:
                            # pdb.set_trace()
                            _, h_ = vision_model(image_tensor_.unsqueeze(0).cuda())
                            # h_ = vision_model(image_tensor_.unsqueeze(0).cuda())
                        h.append(h_)
                    # pdb.set_trace()
                    h = average_tuples(h)
                else:
                    if model_is_llaval:
                        h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).half().cuda(),
                            output_hidden_states=True,
                            return_dict=True).hidden_states

                    else:
                        _, h = vision_model(image_tensor[example_id][style_id].unsqueeze(0).cuda())
                        # h = vision_model(image_tensor[example_id][style_id].unsqueeze(0).cuda())
                
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu()) # torch.Size([1, 577, 1024])
                embedding_token = torch.cat(embedding_token, dim=0) # torch.Size([25, 577, 1024])
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
        if not model_is_llaval:
            model.transformer.visual.output_hidden_states = False
    # pdb.set_trace()
    del h, embedding_token

    return h_all

def obtain_visual_vti(model, image_tensor, rank=1, model_is_llaval=True):

    hidden_states = get_visual_hiddenstates(model, image_tensor, model_is_llaval = model_is_llaval)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape # torch.Size([25, 577, 1024])
    num_demonstration = len(hidden_states)
    
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1) # torch.Size([577, 25600])
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:] # n_token (no CLS token) x n_demos x D     torch.Size([577, 6, 25600])
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1) # torch.Size([577, 1, 25600])  torch.Size([577,  25600])  torch.Size([25, 577, 1024])
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)

    # pdb.set_trace()
    # torch.save(direction, 'qwen_vti_visual.pt')
    return direction, reading_direction