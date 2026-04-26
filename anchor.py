IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_LENGTH = 576
MINIGPT4_IMAGE_TOKEN_LENGTH = 32
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
SHIKRA_IMAGE_TOKEN_LENGTH = 256
SHIKRA_IMG_START_TOKEN = 32001
SHIKRA_IMG_END_TOKEN = 32002

IMAGE_PAD_TAG = '<imgpad>' # Qwen-VL-Chat
IMG_TOKEN_SPAN = 256
QWEN_IMG_START_TOKEN = 151857
QWEN_IMG_END_TOKEN = 151858

INSTRUCTION_TEMPLATE = {
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "qwen-vl-chat": "<|im_start|>user\n<imagehere>\n<question><|im_end|>\n<|im_start|>assistant\n",
    "deepseek-vl-chat": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n\nUser: <image_placeholder><question>\n\nAssistant:",
}

INSTRUCTION_TEMPLATE_NO_IMG = {
    "llava-1.5": "USER: <question> ASSISTANT:",
}

IMAGE_PLACEHOLDER = {
    "llava-1.5": "<ImageHere>",
}


SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


POPE_PATH = {
    "random": "./pope_coco/coco_pope_random.json",
    "popular": "./pope_coco/coco_pope_popular.json",
    "adversarial": "./pope_coco/coco_pope_adversarial.json",
}

MME_PATH = {
    "existence": "/home/zhangcs/zhangcs/code/PTI/MME_Benchmark/MME_Benchmark/existence.json",
    "count": "/home/zhangcs/zhangcs/code/PTI/MME_Benchmark/MME_Benchmark/count.json",
    "position": "/home/zhangcs/zhangcs/code/PTI/MME_Benchmark/MME_Benchmark/position.json",
    "color": "/home/zhangcs/zhangcs/code/PTI/MME_Benchmark/MME_Benchmark/color.json",
}

AMBER_PATH = {
    "d": "/home/zhangcs/zhangcs/code/PTI/AMBER/data/query/query_discriminative.json",
    "g": "/home/zhangcs/zhangcs/code/PTI/AMBER/data/query/query_generative.json",
}


GPT_EVAL_PROMPT = '''
You are a vision-language evaluator. Given an image and an AI-generated description, perform the following tasks:

1. List clearly visible contents in the image that are not mentioned in the description.
2. List hallucinated contents in the description that are not present in the image.
3. List contents accurately described in the description that match the image.

For each task, include objects, object properties (e.g., color, count, position), and relationships between objects. You must answer each content with a single word, separating different contents by commas. If no contents apply, write "None". Make sure there is no overlapping words between three tasks.

Answer 1: [Missing contents]
Answer 2: [Hallucinated contents]
Answer 3: [Accurate contents]
'''


# OpenAI API Key
API_KEY = "Your OpenAI API Key"