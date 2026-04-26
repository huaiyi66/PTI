python chair_eval_cache.py --model "llava-1.5" --data-path '/home/zhangcs/zhangcs/dataset/coco/val2014' --exp_folder 'chair_ablation' \
--just_test \
--method 'cache'  --add_generation_prompt  --img_keys 0.1 --img_values 0.6 --txt_keys 0.1  --txt_values 0.6  --n_contrastive_samples 100 --category 'Object' --aggregation_method 'pca' 



python chair_ans.py  --cap_file '/home/zhangcs/zhangcs/code/PTI/exp_chair/chair_ablation/llava-1.5/test.jsonl' --image_id_key image_id --caption_key caption --coco_path /home/zhangcs/zhangcs/dataset/coco/annotations/ --save_path /home/zhangcs/zhangcs/code/PTI/exp_chair/chair_ablation/llava-1.5/eval_test.jsonl

