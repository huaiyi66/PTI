import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
import pdb
import torch
from torch.utils.data import DataLoader, Subset

import myutils
from eval_data_loader import COCODataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader

from cache_utils.steering.config import SteeringConfig
from cache_utils.utils.parsers import (
    prompt_construction_args,
    pairs_construction_args,
    steering_extraction_args,
    applying_steering_args,
    cache_steering_args,
)
from cache_utils.cache_steer import get_demos, extract_steering_kv_text, generate_with_cache_steering, extract_steering_kv_img, extract_steering_kv_text_deepseek, extract_steering_kv_img_deepseek
from cache_utils.cache_steer import extract_steering_kv_text_qwenVLchat,extract_steering_kv_img_qwenVLchat 
from pycocotools.coco import COCO
import pickle
from cache_utils.cache_util import save_steering_data, load_steering_data

def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR evaluation on MLLMs.")
    # General arguments
    parser.add_argument("--exp_folder", type=str, default="chair_eval", help="save folder name")
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--data-path", type=str, default="../download_datasets/COCO_2014/val2014", help="data path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--subset-size", type=int, default=500)

    # Visual steering vector arguments
    parser.add_argument("--vsv", action="store_true", help='Use visual steering vector')
    parser.add_argument("--vsv-lambda", type=float, default=0.1)
    parser.add_argument("--layers", default=None)

    # penultimate logits augmentation
    parser.add_argument("--logits-aug", action="store_true", help='Use penultimate logits augmentation')
    parser.add_argument("--logits-layers", type=str, default='25,30', help='Layer for penultimate logits augmentation')
    parser.add_argument("--logits-alpha", type=float, default=0.3, help='Alpha for penultimate logits augmentation')

    # Decoding arguments
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--num-workers", type=int, default=1)
    
    #cache
    parser.add_argument("--method", type=str, default="vista")
    parser.add_argument("--just_test", action='store_true', default=False)
    parser.add_argument("--category", type=str, default='Object')
    parser.add_argument("--mask_ratio", type=float, default=0.9)
    parser = pairs_construction_args(parser)
    parser = steering_extraction_args(parser)
    parser = applying_steering_args(parser)
    parser = cache_steering_args(parser)
    parser = prompt_construction_args(parser)

    return parser.parse_args()

    

def get_file_name(args):
    file_name = "_".join(myutils.prepare_common_fileparts(args))
    return file_name


def main(args):
    # bath size should be 1 as we are generating image specific steering vectors
    assert args.batch_size == 1, "Batch size should be 1"
    # seed everything
    myutils.seed_everything(args.seed)
    # disable torch init
    disable_torch_init()
    # init_folder_structure
    args.save_dir = myutils.init_folder_structure(args)
    # prepare file name
    args.file_name = get_file_name(args)
    args.data_file = '/home/zhangcs/zhangcs/dataset/coco/'
    cnt = 0
    # prepare template
    template = myutils.prepare_template(args)
    print(args.file_name)
    # get model loader
    model_loader = ModelLoader(args.model)

    if 'qwen' in args.model:
        model_loader.llm_model.generation_config.do_sample = args.do_sample
        model_loader.llm_model.generation_config.top_k  = args.top_k
        model_loader.llm_model.generation_config.top_p  = args.top_p
        model_loader.llm_model.generation_config.num_beams  = args.num_beams 
        model_loader.llm_model.generation_config.max_new_tokens  = args.max_new_tokens 


    # get dataloader
    coco_dataset = COCODataSet(data_path=args.data_path, trans=model_loader.image_processor,  model=args.model)
    # get a randomly sample subdataset without replacement and fixed seed
    if args.subset_size > 0 and args.subset_size < len(coco_dataset):
        np.random.seed(args.seed )
        subset_indices = np.random.choice(len(coco_dataset), args.subset_size, replace=False)
        coco_dataset = Subset(coco_dataset, subset_indices)

    coco_loader = DataLoader(coco_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    evaluator = pickle.load(open('/home/zhangcs/zhangcs/code/VISTA/chair.pkl', 'rb'))

    
    if 'cache' in args.method:
        # args.file_name += f'_cache_steering_{args.category}_ck_{args.c_keys}_cv_{args.c_values}_num_{args.n_contrastive_samples}_shot_{args.num_fewshot_examples}'
        args.file_name += f'_cache_steering_{args.category}_img_key_{args.img_keys}_img_val_{args.img_values}_txt_key_{args.txt_keys}_txt_val_{args.txt_values}_num_{args.n_contrastive_samples}'
        print("using cache steer:", args.file_name)

        with open("/home/zhangcs/zhangcs/code/PTI/cache_utils/coco2014_train_instances_outputs.jsonl", "r", encoding="utf-8") as f:
            seg_dict = json.load(f)
        annFile = '/home/zhangcs/zhangcs/dataset/coco/annotations/instances_train2014.json'
        coco = COCO(annFile)

        if args.model== "llava-1.5":
            input_ids, inputs_images, contrast_images, contrast_ids  = get_demos(args, model_loader.image_processor, model_loader.llm_model, model_loader.tokenizer, vlm_model=None, template=template,
                                            file_path='./cache_utils/coco2014_train_captions_outputs_object.jsonl', 
                                            category=args.category, seg_dict=seg_dict)

            # Define the config for dataset creation
            config = SteeringConfig(
                n_contrastive_samples=args.n_contrastive_samples,
                num_fewshot_examples=args.num_fewshot_examples,
                tokenizer=model_loader.tokenizer,
                add_generation_prompt=args.add_generation_prompt,
                aggregation_method=args.aggregation_method
            )

            steering_kv_text, steering_kv_img = None, None

            steering_kv_text = extract_steering_kv_text(
                    model=model_loader.llm_model,
                    tokenizer=model_loader.tokenizer,
                    input_images = inputs_images, 
                    contrast_images = contrast_images, 
                    input_ids = contrast_ids, 
                    steering_config=config,
                    device="cuda"
                )

            steering_kv_img  = extract_steering_kv_img(
                    model=model_loader.llm_model,
                    tokenizer=model_loader.tokenizer,
                    input_images = inputs_images, 
                    contrast_images = contrast_images, 
                    input_ids = input_ids, 
                    contrast_ids = contrast_ids, 
                    steering_config=config,
                    mask_ratio=args.mask_ratio,
                    device="cuda"
                )
            # pdb.set_trace()
            
            # img_values, img_keys = load_steering_data("steering_img_100.pt")
            # steering_kv_img = {'keys': img_keys, 'values': img_values } 
            # txt_values, txt_keys = load_steering_data("steering_txt_100.pt")
            # steering_kv_text = {'keys': txt_keys, 'values': txt_values } 
            # args.img_end_idx = 34+576
            # args.img_start_idx = 34

        elif args.model == 'qwen-vl-chat':
            # steering_kv_text, steering_kv_img = None, None

            # contrast_txt, contrast_img  = get_demos(args, model_loader.image_processor, model_loader.llm_model, model_loader.tokenizer, vlm_model=None, template=template,
            #                                 file_path='./cache_utils/coco2014_train_captions_outputs_object.jsonl', 
            #                                 category=args.category, seg_dict=seg_dict)
            
            # config = SteeringConfig(
            #     n_contrastive_samples=args.n_contrastive_samples,
            #     num_fewshot_examples=args.num_fewshot_examples,
            #     tokenizer=model_loader.tokenizer,
            #     add_generation_prompt=args.add_generation_prompt,
            #     aggregation_method=args.aggregation_method
            # )
        
            # steering_kv_text = extract_steering_kv_text_qwenVLchat(
            #         model=model_loader.llm_model,
            #         tokenizer=model_loader.tokenizer,
            #         contrast_txt = contrast_txt, 
            #         steering_config=config,
            #         device="cuda",
            #     )

            # steering_kv_img  = extract_steering_kv_img_qwenVLchat(
            #         model=model_loader.llm_model,
            #         contrast_images = contrast_img, 
            #         steering_config=config,
            #         img_token_start = args.img_start_idx,
            #         img_token_end = args.img_end_idx,
            #         device="cuda"
            #     )
            # pdb.set_trace()
            img_values, img_keys = load_steering_data("steering_img_100_qwen.pt")
            steering_kv_img = {'keys': img_keys, 'values': img_values } 
            txt_values, txt_keys = load_steering_data("steering_txt_100_qwen.pt")
            steering_kv_text = {'keys': txt_keys, 'values': txt_values } 
            args.img_start_idx  = 37 + 1
            args.img_end_idx = 295 - 1
        elif args.model == 'deepseek-vl-chat':
            # inputs_txt, inputs_img, mask_txt, mask_img = get_demos(args, model_loader.processor, model=None, tokenizer=None, vlm_model=model_loader.vlm_model, template=template,
            #                                 file_path='./cache_utils/coco2014_train_captions_outputs_object.jsonl', 
            #                                 category=args.category, seg_dict=seg_dict)
            
            # config = SteeringConfig(
            #     n_contrastive_samples=args.n_contrastive_samples,
            #     num_fewshot_examples=args.num_fewshot_examples,
            #     tokenizer=model_loader.tokenizer,
            #     add_generation_prompt=args.add_generation_prompt,
            #     aggregation_method=args.aggregation_method
            # )
        
            # steering_kv_text = extract_steering_kv_text_deepseek(
            #     model=model_loader.llm_model,
            #     inputs_txt = inputs_txt, 
            #     mask_txt = mask_txt, 
            #     steering_config=config,
            #     device="cuda"
            # )

            # steering_kv_img  = extract_steering_kv_img_deepseek(
            #     model=model_loader.llm_model,
            #     inputs_img = inputs_img, 
            #     mask_img = mask_img, 
            #     steering_config=config,
            #     img_start_idx = 41,
            #     img_end_idx = 41+576,
            #     device="cuda"
            # )
            # pdb.set_trace()
            img_values, img_keys = load_steering_data("steering_img_100_deepseek.pt")
            steering_kv_img = {'keys': img_keys, 'values': img_values } 
            txt_values, txt_keys = load_steering_data("steering_txt_100_deepseek.pt")
            steering_kv_text = {'keys': txt_keys, 'values': txt_values } 
            args.img_start_idx  = 41
            args.img_end_idx = 41+576


        steering_config = SteeringConfig(
            tokenizer=model_loader.tokenizer,
            img_keys=args.img_keys,                     # The steering coefficient for the keys        
            img_values=args.img_values,                    # The steering coefficient for the values
            txt_keys=args.txt_keys,                     # The steering coefficient for the keys        
            txt_values=args.txt_values,                    # The steering coefficient for the values
            append_special_token=args.append_special_token,      # Whether to append a special token to the input to offset the position of the steering token
        )
    
    if args.just_test:
        args.file_name = 'test'

    # prepare save file
    result_file = os.path.join(args.save_dir, args.file_name + ".jsonl")
    if os.path.exists(result_file) and not args.just_test:
        exit(f"Result file {result_file} already exists. Exiting.")
    f = open(result_file, "w", encoding="utf-8")


    cnt = 0

    # inference
    for _, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
        with torch.inference_mode():
            img_id = data["img_id"]
            if args.model== "llava-1.5" or args.model== "minigpt4"  or args.model== "instructblip":
                image = data["image"]
            elif args.model == 'qwen-vl-chat':
                image = data["raw_image"]
            elif args.model == 'deepseek-vl-chat':
                # image = data["raw_image"]
                image = data["image"]
            
            img_file = data["img_file"][0]

            

            batch_size = img_id.shape[0]
            query = ["Please help me describe the image in detail."] * batch_size

            cnt += 1
            if cnt > 50 and args.just_test:
                break

            from cache_utils.cache_util import visualize_avg_head_attention, analyze_image_token_attention
            token_regions = {"img": (34, 34+576)}  # 图像token从索引10到585
            from PIL import Image
            raw_image = Image.open(data["raw_image"][0]).convert("RGB")
            # pdb.set_trace()
            # raw_image.save('/home/zhangcs/zhangcs/code/VISTA/cache_utils/org.jpg')
            raw_img_np = np.array(raw_image)  # 形状：(raw_h, raw_w, 3)，值范围[0,255]
            raw_img_norm = raw_img_np / 255.0  # 归一化到[0,1]，用于matplotlib显示
            raw_shape = np.array(raw_image).shape[:-1]

            with myutils.maybe_autocast(args.model, model_loader.vlm_model.device):
                # prepare inputs
                questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)

                _, kwargs = model_loader.prepare_inputs_for_model(template, query, image)


                # generate
                if args.do_sample:
                    kwargs['top_p'] = args.top_p
                    kwargs['top_k'] = args.top_k
                
                if args.model == 'deepseek-vl-chat':
                    kwargs['pad_token_id'] = model_loader.tokenizer.eos_token_id
                    kwargs['bos_token_id'] = model_loader.tokenizer.bos_token_id
                    kwargs['eos_token_id'] = model_loader.tokenizer.eos_token_id

                    if 'cache' not in args.method:
                        del kwargs["prefix_embeds"]
                        del kwargs["input_ids"]
            
                if 'cache' in args.method:
                    outputs = generate_with_cache_steering(
                        model_loader.llm_model,
                        steering_kv = (steering_kv_text, steering_kv_img), 
                        task='POPE',
                        steering_config=steering_config,
                        output_full_dict=True,
                        args=args,
                        stopping_criteria=model_loader.stopping_criteria  if 'qwen' in args.model or 'gpt' in args.model else None,
                        mask_id=img_file,
                        mask_shape=raw_shape,
                        seg_dict=seg_dict,
                        coco=coco,
                        **kwargs,
                    )
                   

                else:
                    outputs = model_loader.llm_model.generate(
                        do_sample=args.do_sample,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        num_beams=args.num_beams,
                        # output_attentions=True,
                        stopping_criteria=model_loader.stopping_criteria if 'qwen' in args.model or 'gpt' in args.model else None,
                        output_hidden_states=True if args.logits_aug else False,
                        # output_scores=True,
                        # output_hidden_states=True ,
                        # output_attentions= True,
                        # return_dict_in_generate=True,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        # return_dict=True,
                        # return_dict_in_generate=True,
                        **kwargs
                        )
                    
                
            output_text = model_loader.decode(outputs)

        # write to file
        for i in range(len(output_text)):
            f.write(json.dumps({"image_id": int(img_id[i]), "caption": output_text[i]}) + "\n")
        f.flush()
    f.close()




if __name__ == "__main__":
    from chair_ans import CHAIR
    args = parse_args()
    main(args)