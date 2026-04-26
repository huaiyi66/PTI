# PTI

## Overview

`PTI` 是一个面向多模态大模型图像描述任务的实验仓库，核心目标是对模型生成过程注入基于 KV cache 的 steering 信号，并在 COCO 等基准上观察其对 hallucination 的影响。

从当前代码看，这个仓库主要围绕 `chair_eval_cache.py` 展开：
- 加载视觉语言模型
- 构造图像描述提示词
- 提取或加载 text / image steering key-value
- 在生成阶段对 cache 进行干预
- 输出描述结果到 `jsonl`
- 可进一步用 `chair_ans.py` 计算 CHAIR 相关指标

当前代码明确支持的模型分支有：
- `llava-1.5`
- `qwen-vl-chat`
- `deepseek-vl-chat`

## What This Repository Does

这个仓库主要做两类事情：

1. 生成图像描述
   以 COCO 验证集图像为输入，调用多模态模型生成 caption。

2. 研究 cache steering 对 hallucination 的影响
   代码会构造或读取 steering 向量，并在推理时对 key/value cache 进行调整，再将结果写入实验目录，用于后续评估。

从脚本命名和实现看，当前主实验重点在：
- CHAIR 评测流程
- 基于 object/background 或 contrastive samples 的 steering 提取
- 多模型统一实验入口
- 对 COCO、POPE、MME、AMBER 等评测数据格式的支持

## Directory Structure

下面是当前仓库中最重要的目录和文件：

- `chair_eval_cache.py`: 主实验入口。负责参数解析、模型加载、数据迭代、cache steering 推理和结果写盘。
- `chair_ans.py`: CHAIR 指标评估脚本，基于 COCO annotation 和生成的 caption 结果进行打分。
- `model_loader.py`: 模型加载与输入构造逻辑，封装了 LLaVA、Qwen-VL、DeepSeek-VL 的差异。
- `eval_data_loader.py`: 数据集封装，包含 `COCODataSet`、`POPEDataSet`、`MMEDataSet`、`AMBERDataSet`。
- `myutils.py`: 公共工具函数，包括随机种子、输出目录初始化、prompt 模板准备、PCA 等。
- `anchor.py`: 常量定义，包含不同模型的 prompt 模板、图像 token 长度、数据路径常量等。
- `test.sh`: 示例运行命令，展示了如何启动一次 CHAIR 相关实验。
- `cache_utils/`: cache steering 的核心实现目录。
- `cache_utils/cache_steer.py`: steering 提取与应用相关主逻辑。
- `cache_utils/cache_util.py`: cache 操作、steering 保存/加载、注意力分析等底层工具。
- `cache_utils/steering/`: steering 配置类。
- `cache_utils/utils/`: 命令行参数、日志和常量工具。
- `cache_utils/coco2014_train_captions_outputs_object.jsonl`: 用于构建 steering 或 demo 的训练集对象描述缓存。
- `cache_utils/coco2014_train_instances_outputs.jsonl`: 训练集实例信息缓存。
- `llava/`: 仓库内置的 LLaVA 相关代码。
- `qwen_vl/`: 仓库内置的 Qwen-VL 适配代码。
- `deepseek_vl/`: 仓库内置的 DeepSeek-VL 适配代码。
- `steering_img_100.pt`, `steering_txt_100.pt`: 预存的 steering 张量文件。

## Key Files and Modules

### 1. 主入口：`chair_eval_cache.py`

这是最值得优先阅读的文件。主流程大致如下：

1. 解析实验参数，例如模型类型、数据路径、生成参数、steering 参数。
2. 初始化输出目录 `./exp_chair/{exp_folder}/{model}/`。
3. 通过 `ModelLoader` 加载模型和图像预处理器。
4. 构造 `COCODataSet` 并按需抽样子集。
5. 如果启用 `cache` 方法，则提取或读取 steering key/value。
6. 对每张图像执行生成，输出 caption。
7. 将结果保存为 `jsonl` 文件。

输出文件格式类似：

```json
{"image_id": 42, "caption": "a man riding a bicycle on the beach"}
```

### 2. 模型加载：`model_loader.py`

该文件统一封装了不同模型的：
- 权重加载
- tokenizer 初始化
- image processor 初始化
- 输入 prompt 与 image token 的拼接方式
- 生成结果解码

当前代码中已经写死的模型分支包括：
- `llava-1.5`
- `qwen-vl-chat`
- `deepseek-vl-chat`

需要注意：部分模型路径是硬编码的，例如：
- LLaVA 权重路径指向 `/home/zhangcs/zhangcs/code/VISTA/llava/llava-v1.5-7b`
- Qwen-VL 的 Hugging Face cache 路径也带有本地环境假设

这意味着仓库并不是一个“开箱即用”的通用项目，更像是绑定具体实验环境的研究代码。

### 3. 数据集定义：`eval_data_loader.py`

这里封装了多种评测数据格式：
- `COCODataSet`: 用于图像描述生成
- `POPEDataSet`: 处理 POPE 问答样本
- `MMEDataSet`: 处理 MME benchmark 样本
- `AMBERDataSet`: 处理 AMBER 数据

从当前主脚本看，`chair_eval_cache.py` 直接使用的是 `COCODataSet`，但底层已经为其他基准预留了适配。

### 4. 评估：`chair_ans.py`

该文件主要实现 CHAIR 评估逻辑，核心能力包括：
- 载入 COCO annotation
- 用同义词表归一化 object 表达
- 比较 caption 中对象词与标注对象的匹配情况
- 计算 hallucination 相关统计指标

从文件头注释看，它基于公开 CHAIR 实现修改而来，并做了 Python 3 适配和缓存优化。

### 5. Steering 逻辑：`cache_utils/`

这是整个项目最核心的研究实现区域。

可以从以下文件开始看：
- `cache_utils/cache_steer.py`: steering 向量提取、聚合和生成期注入
- `cache_utils/cache_util.py`: cache 读写、token 处理、可视化与相似度分析
- `cache_utils/steering/config.py`: steering 参数配置
- `cache_utils/utils/parsers.py`: 命令行参数定义

如果你是想理解“这个项目的创新点在哪”，这里是最需要深入阅读的部分。

## How It Works

以 `chair_eval_cache.py` 为例，当前实验链路大致可以概括为：

1. 读取 COCO 图像目录。
2. 对每张图像构造统一 query，例如 `Please help me describe the image in detail.`。
3. 根据模型类型把文本和图像整理成对应输入格式。
4. 若启用了 `cache` 方法：
   - 从缓存数据集中构造 contrastive demos
   - 提取文本和图像的 steering key/value
   - 在生成时调用 `generate_with_cache_steering(...)`
5. 若未启用 `cache` 方法，则直接走模型原生 `generate(...)`。
6. 将生成 caption 保存到 `exp_chair/.../*.jsonl`。
7. 可选地再运行 `chair_ans.py` 对生成结果做 CHAIR 评估。

## Environment and Dependencies

仓库里没有提供 `requirements.txt` 或环境配置文件，因此以下依赖是根据代码 import 保守归纳的。

高概率需要：
- Python 3
- PyTorch
- torchvision
- transformers
- Pillow
- numpy
- tqdm
- PyYAML
- pycocotools
- nltk

代码中还出现了以下依赖或环境假设，是否全部必需需要进一步确认：
- matplotlib
- seaborn
- pandas
- scikit-learn
- OpenAI SDK
- gradio / fastapi / uvicorn
- flash-attn / deepspeed / apex
- peft
- timm
- spacy

另外，仓库包含 `llava/`、`qwen_vl/`、`deepseek_vl/` 的本地代码副本或适配实现，因此实际可运行性不仅取决于 Python 包，还取决于本地模型权重、CUDA 环境以及若干硬编码路径是否有效。

## How to Run

### 1. 生成图像描述

仓库中唯一明确给出的运行示例在 `test.sh`：

```bash
python3 chair_eval_cache.py \
  --model "llava-1.5" \
  --data-path '/home/zhangcs/zhangcs/dataset/coco/val2014' \
  --exp_folder 'chair_ablation' \
  --just_test \
  --method 'cache' \
  --add_generation_prompt \
  --img_keys 0.1 \
  --img_values 0.6 \
  --txt_keys 0.1 \
  --txt_values 0.6 \
  --n_contrastive_samples 100 \
  --category 'Object' \
  --aggregation_method 'pca'
```

如果环境匹配，结果会被写到：

```text
./exp_chair/{exp_folder}/{model}/test.jsonl
```

### 2. 运行 CHAIR 评估

`test.sh` 中还给出了后处理示例，当前是注释状态：

```bash
python3 chair_ans.py \
  --cap_file '/home/zhangcs/zhangcs/code/PTI/exp_chair/chair_ablation/llava-1.5/test.jsonl' \
  --image_id_key image_id \
  --caption_key caption \
  --coco_path /home/zhangcs/zhangcs/dataset/coco/annotations/ \
  --save_path /home/zhangcs/zhangcs/code/PTI/exp_chair/chair_ablation/llava-1.5/eval_test.jsonl
```

### 3. 运行前需要确认的事项

在真正执行前，建议先检查：
- 模型权重路径是否存在
- COCO 数据路径是否存在
- COCO annotation 路径是否存在
- `chair.pkl`、steering `.pt` 文件、训练缓存 `.jsonl` 文件是否存在
- 当前 Python 环境是否包含 PyTorch、Transformers 和 `pycocotools`

## Output

当前代码可以明确产出的文件包括：

- `./exp_chair/{exp_folder}/{model}/*.jsonl`
  保存模型对图像生成的 caption 结果。

- CHAIR 评估结果文件
  由 `chair_ans.py` 写出，具体格式取决于命令行参数指定的 `save_path`。

- steering 缓存文件
  仓库顶层已经存在一些 `.pt` 文件，例如：
  - `steering_img_100.pt`
  - `steering_txt_100.pt`

## Known Gaps or Notes

- 这是一个研究实验仓库，不是完整产品化工程，代码中包含较多硬编码绝对路径。
- 仓库未提供标准环境描述文件，部署成本较高。
- 主入口当前直接使用 `COCODataSet`，虽然仓库中还定义了 POPE、MME、AMBER 数据集，但这些路径在当前 README 中未逐一验证完整运行链路。
- `anchor.py` 中包含 OpenAI API Key 占位符，但从当前主流程看，主实验并不直接依赖它。
- 一些依赖来自仓库内置代码或特定训练/推理环境，是否能在全新机器上复现，需进一步确认。

## Recommended Reading Order

如果你刚接手这个项目，建议按下面顺序阅读：

1. `test.sh`
2. `chair_eval_cache.py`
3. `model_loader.py`
4. `eval_data_loader.py`
5. `cache_utils/cache_steer.py`
6. `cache_utils/cache_util.py`
7. `chair_ans.py`

这样可以先理解实验入口，再逐步进入模型适配、数据加载、steering 机制和评估细节。
