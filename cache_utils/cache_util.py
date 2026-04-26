
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
# from .pca import PCA
import torch.nn.functional as F
import numpy as np
# from torchvision import transforms
# from typing import List, Tuple
from collections import defaultdict
# from tqdm.auto import tqdm
from cache_utils.steering.config import SteeringConfig
# from cache_utils.utils.helpers import pad_tokens, get_token_to_append
from cache_utils.utils.logging_setup import logger
from anchor import SYSTEM_MESSAGE
from pycocotools.coco import COCO
from myutils import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap  # 自定义颜色映射（可选）
import seaborn as sns
from transformers import DynamicCache, BatchEncoding, PreTrainedModel
from matplotlib.colors import LogNorm  # 对数刻度适配颜色范围（0.001~1）
from matplotlib.gridspec import GridSpec
import cv2  # 用于resize
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA as SklearnPCA  # 用于2D可视化（可选，PyTorch PCA用于1D）
from sklearn.manifold import TSNE
from typing import List, Optional


def _random_mask_patchgrid(
    H, W,
    keep_ratio: float,
    patch: int = 14,
    device=None,
    object_mask_2d: Optional[torch.Tensor] = None,  # [H,W], 0/1；若提供且仅在目标区域内采样，则以其像素计权
):
    """
    生成由 14x14（默认）小块组成的随机“保留”mask（二值），总保留面积≈ keep_ratio * 全图/目标面积
    - 自动处理边界 patch（不足14的边缘块）
    - 使用“按面积/目标像素加权”的无放回采样，直到累计面积达到目标值
    """
    keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
    ys = list(range(0, H, patch))
    xs = list(range(0, W, patch))

    # 逐块统计面积，以及（可选）与目标区域的重叠像素数
    rects, areas, weights = [], [], []
    total_area_ref = 0  # 用于定义目标面积：全图或目标区域像素
    if object_mask_2d is not None:
        total_area_ref = int(object_mask_2d.sum().item())
    else:
        total_area_ref = H * W

    for y in ys:
        h_ = min(patch, H - y)
        for x in xs:
            w_ = min(patch, W - x)
            a = h_ * w_
            rects.append((y, y + h_, x, x + w_))
            areas.append(a)
            if object_mask_2d is not None:
                # 权重=与目标区域的重叠像素（避免挑到目标外“空块”）
                w_overlap = int(object_mask_2d[y:y + h_, x:x + w_].sum().item())
                weights.append(w_overlap)
            else:
                weights.append(a)

    areas   = torch.tensor(areas,   dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    # 若目标区域过小或权重全零，退化为全零/全一的简单情况
    if total_area_ref == 0 or keep_ratio == 0.0:
        return torch.zeros((H, W), dtype=torch.float32, device=device)
    if keep_ratio == 1.0:
        return torch.ones((H, W), dtype=torch.float32, device=device)
    if weights.sum() == 0:
        # 目标掩码存在但没有任何可采样像素，返回全零
        return torch.zeros((H, W), dtype=torch.float32, device=device)

    target_area = keep_ratio * total_area_ref

    # 归一化到概率分布，做无放回采样，直到累计面积达到目标
    probs = (weights / weights.sum()).clamp_min(1e-12)
    selected = torch.zeros_like(areas, dtype=torch.bool, device=device)
    rem_idx = torch.arange(len(areas), device=device)
    cur_area = 0.0

    # 为避免极端情况下循环过长，设置上限（全部块）
    while cur_area < target_area and rem_idx.numel() > 0:
        # 从剩余集合中按权重采1个
        p = probs[rem_idx]
        p = p / p.sum()
        i = torch.multinomial(p, num_samples=1).item()
        idx = rem_idx[i].item()
        # 若这个块确实包含目标像素（或没有object_mask时总可用），加入
        if weights[idx] > 0:
            selected[idx] = True
            cur_area += (weights[idx].item() if object_mask_2d is not None else areas[idx].item())
        # 移出该块
        rem_idx = torch.cat([rem_idx[:i], rem_idx[i+1:]], dim=0)

    # 组装mask
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    for k, sel in enumerate(selected.tolist()):
        if sel:
            y0, y1, x0, x1 = rects[k]
            mask[y0:y1, x0:x1] = 1.0
    return mask


def remove_words_from_desc(object_list, desc):
    # 按短语长度（单词数量）排序，确保长短语先被处理，避免被拆分
    sorted_objects = sorted(object_list, key=lambda x: len(x.split()), reverse=True)
    
    processed_desc = desc
    for item in sorted_objects:
        # 构建正则表达式，确保匹配整个单词/短语
        # 使用\b作为单词边界，re.escape处理特殊字符
        pattern = r'\b' + re.escape(item) + r'\b'
        processed_desc = re.sub(pattern, '', processed_desc)

        # pattern = r'\b' + re.escape(item+'s') + r'\b'
        # processed_desc = re.sub(pattern, '', processed_desc)
    
    # 处理多余的空格：多个空格合并为一个，去除首尾空格
    processed_desc = re.sub(r'\s+', ' ', processed_desc).strip()
    
    # 处理标点符号前的空格（如 " ." 变为 "."）
    processed_desc = re.sub(r' (\.)', r'\1', processed_desc)
    
    return processed_desc


# def visualize_pca_for_layer(
#     data,  # [num_demos, n_heads, head_dim]，差异向量
#     pca,   # 已fit的PCA对象（你的自定义PCA）
#     direction,  # [1, n_heads, head_dim]，引导方向
#     layer_id,
#     title_prefix="Values",
#     show_projection_hist=True,
#     save_path="pca_visualization.jpg"
# ):
#     num_demos, n_heads, head_dim = data.shape
#     data_reshaped = data.view(num_demos, -1).cpu().numpy()  # [num_demos, features]，移到CPU for plotting
    
#     # 1. 拟合2D PCA用于散点图可视化（使用sklearn for simplicity）
#     pca_2d = SklearnPCA(n_components=2)
#     data_2d = pca_2d.fit_transform(data_reshaped)
    
#     # 2. 计算在1D主成分上的投影（使用你的PCA）
#     projections = pca.transform(torch.tensor(data_reshaped).to(pca.components_.device)).cpu().numpy().flatten()  # [num_demos]
    
#     # 3. 引导方向在2D中的投影（approximate）
#     direction_reshaped = direction.view(1, -1).cpu().numpy()  # [1, features]
#     direction_2d = pca_2d.transform(direction_reshaped)[0]  # [2]
    
#     # 创建多子图
#     fig, axs = plt.subplots(1, 3 if show_projection_hist else 2, figsize=(15, 5))
    
#     # 子图1: 2D散点图（差异向量分布）
#     axs[0].scatter(data_2d[:, 0], data_2d[:, 1], c='blue', label='Difference Vectors')
#     axs[0].arrow(0, 0, direction_2d[0]*np.max(np.abs(data_2d[:,0])), direction_2d[1]*np.max(np.abs(data_2d[:,1])), 
#                  head_width=0.05, head_length=0.1, fc='red', ec='red', label='Steering Direction')
#     axs[0].set_title(f'{title_prefix} Layer {layer_id}: 2D PCA Projection')
#     axs[0].set_xlabel('PC1')
#     axs[0].set_ylabel('PC2')
#     axs[0].legend()
#     axs[0].grid(True)
    
#     # 子图2: 解释方差
#     explained_var = pca.explained_variance_ratio_.cpu().numpy() if hasattr(pca, 'explained_variance_ratio_') else [pca_2d.explained_variance_ratio_[0]]
#     axs[1].bar(['PC1'], explained_var, color='green')
#     axs[1].set_title('Explained Variance Ratio')
#     axs[1].set_ylim(0, 1)
    
#     # 子图3: 投影直方图（可选）
#     if show_projection_hist:
#         axs[2].hist(projections, bins=20, color='purple', alpha=0.7)
#         axs[2].set_title('Histogram of Projections on PC1')
#         axs[2].set_xlabel('Projection Value')
#         axs[2].set_ylabel('Frequency')
    
#     plt.suptitle(f'PCA Visualization for {title_prefix} Layer {layer_id}')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     # plt.show()  # 或 plt.close() 如果只保存

def visualize_pca_for_layer(
    data_3d,         # [num_demos, H, Dh]，正-负差分
    pca_obj,         # 上面 fit 得到的 PCA 对象
    direction_3d,    # [1, H, Dh]，用于 steer 的方向（通常取 PC1 reshape）
    layer_id=0,
    title_prefix="Values",
    show_projection_hist=True,  # 这里保留参数，不想画第1个子图时可设为 False
    save_path=None,             # 若提供，则保存整张图
):
    assert data_3d.dim() == 3
    num_demos, H, Dh = data_3d.shape
    D = H * Dh

    # 展平
    X = data_3d.reshape(num_demos, D)               # [N, D]
    # pc1 = pca_obj.components_[0]                    # [D]
    pc1 = (pca_obj.components_.sum(dim=0,keepdim=True) + pca_obj.mean_).mean(0)
    proj = torch.matmul(X - pca_obj.mean_, pc1)     # [N]

    pc1_2d = pc1.view(H, Dh)                        # [H, Dh]
    head_importance = torch.linalg.norm(pc1_2d, dim=1)  # [H]

    dir2d = direction_3d.view(H, Dh)

    # 为两个热图设一个一致的颜色范围（可选，但利于横向比较）
    vlim = float(torch.max(torch.abs(torch.stack([pc1_2d, dir2d]))).item())
    vmin, vmax = -vlim, vlim if vlim > 0 else (None, None)

    # 创建一行四列的子图
    # figsize 可按需要调整；constrained_layout=True 避免文字重叠
    ncols = 4
    fig, axs = plt.subplots(1, ncols, figsize=(20, 4), constrained_layout=True)

    col = 0
    # 子图1：PC1 投影直方图
    if show_projection_hist:
        ax = axs[col]
        ax.hist(proj.detach().cpu().numpy(), bins=30)
        ttl = f"{title_prefix} L{layer_id}: PC1 projection"
        if hasattr(pca_obj, "explained_variance_ratio_"):
            evr = pca_obj.explained_variance_ratio_[0].item()
            ttl += f" (EVR≈{evr:.3f})"
        ax.set_title(ttl)
        ax.set_xlabel("Projection onto PC1")
        ax.set_ylabel("Count")
        col += 1
    else:
        # 如果不画直方图，放个占位空图使得后续三图仍然对齐
        axs[col].axis("off")
        col += 1

    # 子图2：头重要性柱状图
    ax = axs[col]
    ax.bar(np.arange(H), head_importance.detach().cpu().numpy())
    ax.set_title(f"{title_prefix} L{layer_id}: Head importance (||PC1||)")
    ax.set_xlabel("Head index")
    ax.set_ylabel("L2 norm")
    col += 1

    # 子图3：PC1 热图 [H x Dh]
    ax = axs[col]
    im1 = ax.imshow(pc1_2d.detach().cpu().numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title_prefix} L{layer_id}: PC1 heatmap")
    ax.set_xlabel("head_dim")
    ax.set_ylabel("head")
    # 独立的 colorbar：与子图共享高度
    cbar1 = fig.colorbar(im1, ax=ax)
    col += 1

    # 子图4：Steering direction 热图 [H x Dh]
    ax = axs[col]
    im2 = ax.imshow(dir2d.detach().cpu().numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(f"{title_prefix} L{layer_id}: Steering direction")
    ax.set_xlabel("head_dim")
    ax.set_ylabel("head")
    cbar2 = fig.colorbar(im2, ax=ax)

    if save_path is not None:
        fig.savefig(save_path, dpi=200)

    return evr, head_importance.detach().cpu()

def save_kv_diff(evr_values_dict, evr_keys_dict, headimp_values_cols, headimp_keys_cols, savepath):
    # 排序后的层索引
    layer_ids = sorted(evr_values_dict.keys())
    L = len(layer_ids)

    # EVR 折线数据
    evr_vals = np.array([evr_values_dict[i] for i in layer_ids])
    evr_keys = np.array([evr_keys_dict[i]   for i in layer_ids])


    # 组装“每头重要性”矩阵：行=Head，列=Layer
    headimp_values_mat = torch.stack(
        [v for (lid, v) in sorted(headimp_values_cols, key=lambda x: x[0])], dim=1
    ).numpy()  # [H, L]
    headimp_keys_mat = torch.stack(
        [v for (lid, v) in sorted(headimp_keys_cols,   key=lambda x: x[0])], dim=1
    ).numpy()  # [H, L]

    # 可选：按列（每层）做归一化，便于跨层比较（不看绝对量级）
    # norm_cols = True
    # if norm_cols:
    #     headimp_values_mat = headimp_values_mat / (headimp_values_mat.sum(axis=0, keepdims=True) + 1e-12)
    #     headimp_keys_mat   = headimp_keys_mat   / (headimp_keys_mat.sum(axis=0, keepdims=True)   + 1e-12)

    # 一行三列子图：左(EVR折线) + 中(Keys热图) + 右(Values热图)
    fig, axs = plt.subplots(
        1, 3, figsize=(21, 5), constrained_layout=True,
        gridspec_kw={'width_ratios': [1.2, 1.6, 1.6]}
    )

    # 子图1：EVR折线（Keys & Values）
    ax = axs[0]
    ax.plot(layer_ids, evr_keys, marker='o', label='Keys EVR (PC1)')
    ax.plot(layer_ids, evr_vals, marker='o', label='Values EVR (PC1)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('EVR of PC1')
    ax.set_title('EVR across layers')
    ax.set_xticks(layer_ids)
    # 如果层数很多，可稀疏刻度：
    # ax.set_xticks(layer_ids[::2])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 子图2：Keys 每头重要性热图（行=head，列=layer）
    ax = axs[1]
    im_k = ax.imshow(headimp_keys_mat, aspect='auto')
    ax.set_title('Keys head importance (||PC1||)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(layer_ids, rotation=0)
    fig.colorbar(im_k, ax=ax, fraction=0.046, pad=0.04)

    # 子图3：Values 每头重要性热图（行=head，列=layer）
    ax = axs[2]
    im_v = ax.imshow(headimp_values_mat, aspect='auto')
    ax.set_title('Values head importance (||PC1||)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(layer_ids, rotation=0)
    fig.colorbar(im_v, ax=ax, fraction=0.046, pad=0.04)

    # 保存或展示
    fig.savefig(savepath, dpi=200)
    
    return evr_keys, evr_vals, headimp_keys_mat, headimp_values_mat


def analyze_image_token_attention(
    outputs,
    token_regions,
    raw_img_norm,
    tokenizer,
    inital_seq_len,
    img_patch_size=(24, 24),
    save_path=None,
    num_steps=[0,10,20,30,40],
    max_time_steps=8,
    col=5,
    figsize=(12, 30)
):
    """
    可视化生成过程中的图像token注意力。
    
    参数:
    ----------
    outputs: dict
        模型生成输出，必须包含 'attentions'。
        outputs['attentions'] 是一个 list，每个元素对应生成步，
        每步是 list (len=num_layers)，每个元素的 shape: (batch_size, num_heads, seq_len, seq_len)
    token_regions: dict
        图像token在序列中的索引范围，例如 {"img": (34, 34+576)}
    raw_img_norm: np.ndarray
        原始图像数组，形状(H, W, 3)，像素值归一化到[0,1]
    img_patch_size: tuple
        单个图像patch的大小(H_patch, W_patch)
    save_path: str
        保存可视化图像路径
    max_time_steps: int
        最大生成步数
    figsize: tuple
        可视化图尺寸
    """
    
    # 获取图像token的索引
    img_start, img_end = token_regions["img"]
    img_token_len = img_end - img_start
    sequences = outputs['sequences']
    attentions = outputs['attentions']  # list[time_step] -> list[layer] -> tensor
    # 确定绘制步数
    # num_steps = min(max_time_steps, len(attentions))
    
    # 原始图像大小
    H_patch, W_patch = img_patch_size
    H_img = int(np.sqrt(img_token_len)) * H_patch
    W_img = int(np.sqrt(img_token_len)) * W_patch
    
    # 创建画布
    # fig, axes = plt.subplots(len(num_steps), 1, figsize=figsize)
    # if len(num_steps) == 1:
    #     axes = [axes]
    # 网格行列计算
    n_rows = math.ceil(len(num_steps) / col)
    fig, axes = plt.subplots(n_rows, col, figsize=(figsize[0]*col, figsize[1]*n_rows))
    axes = axes.flatten()  # 展平，方便索引

    for i, t in enumerate(num_steps):
        # 当前生成步的注意力
        step_attention = attentions[t]  # list[layer]
        # last_layer_attention = step_attention[-1]  # 最后一层
        last_layer_attention = torch.stack(step_attention).mean(dim=0)  # 对所有层求平均, shape: (batch, num_heads, seq_len, seq_len)
        
        # last_layer_attention shape: (batch, heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = last_layer_attention.shape
        
        # 平均所有head
        avg_attention = last_layer_attention.mean(dim=1)  # shape: (batch, seq_len, seq_len)

        # 计算所有head在所有token上的注意力总和
        # head_attention_sums = last_layer_attention.sum(dim=-1).sum(dim=-1)  # shape: (batch, num_heads)
        # max_attention_head_idx = head_attention_sums.argmax(dim=1)[0].item() # shape: (batch_size,)
        # avg_attention = last_layer_attention[:, max_attention_head_idx, :, :]  # batch=0, 选中的head
        

        # head_sum = last_layer_attention.sum(dim=1)  # 每个头的注意力总和：[num_heads]
        # best_head = torch.argmax(head_sum).item()
        # pdb.set_trace()
        # avg_attention = last_layer_attention[best_head]
        
        # 取batch=0, 最后一个生成token
        if t == 0:
            token_idx = -1  # 初始步，取最后一个token
        else:
            token_idx = -1  # 后续步同样取最后生成token
        
        token_attention = avg_attention[0, token_idx, :]  # shape: (seq_len,)
        
        # 只保留对图像token的注意力
        img_attention = token_attention[img_start:img_end]  # shape: (img_token_len,)
        img_attention_map = img_attention.reshape(int(np.sqrt(img_token_len)), int(np.sqrt(img_token_len))).cpu().numpy()
        img_attention_map = img_attention_map / (img_attention_map.max() + 1e-6)  # 归一化
        
        # 将注意力resize到原始图像大小
        img_attention_map_resized = np.kron(img_attention_map, np.ones((H_patch, W_patch)))

        raw_img_resized = cv2.resize(raw_img_norm, (H_img, W_img))

        gen_token_id = sequences[0, inital_seq_len+t+1]  # batch=0, 当前最后token
        gen_token_text = tokenizer.decode(gen_token_id)
            
        # 可视化
        ax = axes[i]
        ax.imshow(raw_img_resized)
        ax.imshow(img_attention_map_resized, cmap='jet', alpha=0.5)
        # ax.set_title(f'Time step {t}')
        # ax.set_title(f'Time step {t} Head {max_attention_head_idx}: {gen_token_text}')
        ax.set_title(f'Time step {t}: {gen_token_text}')
        ax.axis('off')
    
    # for t in range(num_steps, n_rows*n_cols):
    #     axes[t].axis('off')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()

def visualize_pos_neg_kv(
    cache_positive,
    cache_negative,
    batch_indices=0,
    img_token_slice=slice(34, 34+576),
    img_patch_size=(24, 24),
    max_heads=None,
    max_layers=None,
    save_path=None
):
    """
    可视化pos/neg的key和value
    横轴是head, 纵轴是layer
    """

    num_layers = len(cache_positive.value_cache)
    num_heads = cache_positive.value_cache[0].shape[1]

    if max_layers is not None:
        num_layers = min(num_layers, max_layers)
    if max_heads is not None:
        num_heads = min(num_heads, max_heads)

    # 先初始化矩阵存储平均值
    pos_values_mat = torch.zeros((num_layers, num_heads))
    neg_values_mat = torch.zeros((num_layers, num_heads))
    pos_keys_mat = torch.zeros((num_layers, num_heads))
    neg_keys_mat = torch.zeros((num_layers, num_heads))
    diff_keys_mat = torch.zeros((num_layers, num_heads))
    diff_value_mat = torch.zeros((num_layers, num_heads))

    for layer_id in range(num_layers):
        # pos_value
        # pdb.set_trace()

        pos_values = cache_positive.value_cache[layer_id][batch_indices, :num_heads, img_token_slice, :].mean(dim=-1)  # [batch, heads, tokens]
        pos_values_mat[layer_id, :] = pos_values.mean(dim=-1).cpu()  # 对batch和token平均
        # neg_value
        neg_values = cache_negative.value_cache[layer_id][batch_indices, :num_heads, img_token_slice, :].mean(dim=-1)
        neg_values_mat[layer_id, :] = neg_values.mean(dim=-1).cpu()

        # pos_key
        pos_keys = cache_positive.key_cache[layer_id][batch_indices, :num_heads, img_token_slice, :].mean(dim=-1)
        pos_keys_mat[layer_id, :] = pos_keys.mean(dim=-1).cpu()
        # neg_key
        neg_keys = cache_negative.key_cache[layer_id][batch_indices, :num_heads, img_token_slice, :].mean(dim=-1)
        neg_keys_mat[layer_id, :] = neg_keys.mean(dim=-1).cpu()

        diff_keys_mat[layer_id, :] = pos_keys.mean(dim=-1).cpu() - neg_keys.mean(dim=-1).cpu()
        diff_value_mat[layer_id, :] = pos_values.mean(dim=-1).cpu() - neg_values.mean(dim=-1).cpu()


    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im0 = axes[0].imshow(diff_keys_mat, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title("Diff Keys")
    axes[0].set_xlabel("Heads")
    axes[0].set_ylabel("Layers")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(diff_value_mat, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title("Diff Values")
    axes[1].set_xlabel("Heads")
    axes[1].set_ylabel("Layers")
    fig.colorbar(im1, ax=axes[1])

    # im0 = axes[0].imshow(pos_keys_mat, aspect='auto', origin='lower', cmap='viridis')
    # axes[0].set_title("Pos Keys")
    # axes[0].set_xlabel("Heads")
    # axes[0].set_ylabel("Layers")
    # fig.colorbar(im0, ax=axes[0])

    # im1 = axes[1].imshow(pos_values_mat, aspect='auto', origin='lower', cmap='viridis')
    # axes[1].set_title("Pos Values")
    # axes[1].set_xlabel("Heads")
    # axes[1].set_ylabel("Layers")
    # fig.colorbar(im1, ax=axes[1])

    # im2 = axes[2].imshow(neg_keys_mat, aspect='auto', origin='lower', cmap='viridis')
    # axes[2].set_title("Neg Keys")
    # axes[2].set_xlabel("Heads")
    # axes[2].set_ylabel("Layers")
    # fig.colorbar(im2, ax=axes[2])

    # im3 = axes[3].imshow(neg_values_mat, aspect='auto', origin='lower', cmap='viridis')
    # axes[3].set_title("Neg Values")
    # axes[3].set_xlabel("Heads")
    # axes[3].set_ylabel("Layers")
    # fig.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def vis_attn_change_rate(pos, neg, save_path ):
    # ---------------------- 2. 计算变化率 ----------------------
    epsilon = 1e-8  # 防止分母为0的极小值
    rate = (pos - neg) / (pos + epsilon)  # 逐元素计算变化率
    rate_percent = rate.numpy() * 100  # 转换为百分比（更直观）

    data_min = rate_percent.min()
    data_max = rate_percent.max()

    # pdb.set_trace()
    # ---------------------- 3. 可视化热力图 ----------------------
    plt.figure(figsize=(10, 8))

    # 自定义颜色映射（类似参考图的“绿→黄→红”渐变，对应变化率从负到正）
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [(0, "darkgreen"),   # 低变化率（负）
        (0.5, "yellow"),    # 变化率接近0
        (1, "darkred")]     # 高变化率（正）
        # [(0, "#9FC183"),   # 低变化率（负）
        # (0.5, "#91bfff"),    # 变化率接近0
        # (1, "#F08D87")]     # 高变化率（正）
    )

    # 绘制热力图
    im = plt.imshow(
        rate_percent, 
        # cmap=cmap, 
        aspect="auto", 
        # vmin=data_min,  # 颜色范围最小值（可根据实际数据调整）
        # vmax=data_max    # 颜色范围最大值（可根据实际数据调整）
        vmin=-50,
        vmax=50,
    )

    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label("Change Rate (%)", rotation=270, labelpad=20)  # 垂直显示颜色条标签

    # 设置坐标轴与标题
    plt.xlabel("Head")
    plt.ylabel("Layer")
    # plt.title("Caption vs. Non-caption Attention Change Rate")
    # plt.title("Clean Caption vs. Hallucinatory Caption")
    # plt.title("Object words vs. Clean Caption")
    # plt.title("Object words vs. remain words")
    plt.title("Remain words vs. Clean Caption")
    # plt.title("Hallucinatory Caption vs. Clean Caption")

    # 调整刻度（按需修改，示例为每5个刻度标一次）
    plt.xticks(np.arange(0, 32, 5))
    plt.yticks(np.arange(0, 32, 5))

    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()
    plt.close()


def vis_key_value_diff(cache_positive, cache_negative, save_path):
    # Initialize lists to store norms
    key_differences = []
    value_differences = []
    batch_indices = 0
    pos_indices = -1
    neg_indices = -1

    for layer_id in range(len(cache_positive.value_cache)):
        # Get the positive and negative key and value caches
        pos_values = cache_positive.value_cache[layer_id][batch_indices, :, pos_indices, :]
        neg_values = cache_negative.value_cache[layer_id][batch_indices, :, neg_indices, :]
        pos_keys = cache_positive.key_cache[layer_id][batch_indices, :, pos_indices, :]
        neg_keys = cache_negative.key_cache[layer_id][batch_indices, :, neg_indices, :]
        
        # Calculate the L2 norm of the differences (you can also try other norms or measures)
        value_diff_norm = torch.norm(pos_values - neg_values, p=2)
        key_diff_norm = torch.norm(pos_keys - neg_keys, p=2)
        
        value_differences.append(value_diff_norm.item())
        key_differences.append(key_diff_norm.item())

    # Plot the difference norms across layers
    layers = list(range(len(cache_positive.value_cache)))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(layers, value_differences, label='Value Difference (L2 Norm)')
    plt.xlabel('Layer ID')
    plt.ylabel('L2 Norm of Value Difference')
    plt.title('Key-Value Cache: Value Difference')

    plt.subplot(1, 2, 2)
    plt.plot(layers, key_differences, label='Key Difference (L2 Norm)', color='orange')
    plt.xlabel('Layer ID')
    plt.ylabel('L2 Norm of Key Difference')
    plt.title('Key-Value Cache: Key Difference')

    plt.tight_layout()
    # plt.show()
    # 调整布局（避免标签重叠）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"平均注意力热力图已保存至：{save_path}")
    


def vis_kv_cache(tensor1, tensor2, save_path, token_regions, img_patch_size=(24, 24), img_token_len=576):

    H_patch, W_patch = img_patch_size
    H_img = int(np.sqrt(img_token_len)) * H_patch
    W_img = int(np.sqrt(img_token_len)) * W_patch
    img_start, img_end = token_regions["img"]

    tensor1 = tensor1[img_start:img_end]  # shape: (img_token_len,)
    tensor1 = tensor1.reshape(int(np.sqrt(img_token_len)), int(np.sqrt(img_token_len))).cpu().numpy()
    tensor1 = tensor1 / (tensor1.max() + 1e-6)  # 归一化

    tensor2 = tensor2[img_start:img_end]  # shape: (img_token_len,)
    tensor2 = tensor2.reshape(int(np.sqrt(img_token_len)), int(np.sqrt(img_token_len))).cpu().numpy()
    tensor2 = tensor2 / (tensor1.max() + 1e-6)  # 归一化
    
    # 将注意力resize到原始图像大小
    data1 = np.kron(tensor1, np.ones((H_patch, W_patch)))
    data2 = np.kron(tensor2, np.ones((H_patch, W_patch)))
  

    # 创建画布和子图（1行2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('32x624', fontsize=16)

    # 绘制第一个热力图
    im1 = ax1.imshow(data1, aspect='auto', cmap='viridis')
    ax1.set_title('key')
    # ax1.set_xlabel('624')  # 横坐标
    # ax1.set_ylabel('32')  # 纵坐标
    fig.colorbar(im1, ax=ax1, shrink=0.8)  # 添加颜色条

    # 绘制第二个热力图
    im2 = ax2.imshow(data2, aspect='auto', cmap='viridis')
    ax2.set_title('val')
    # ax2.set_xlabel('624')  # 横坐标
    # ax2.set_ylabel('3')  # 纵坐标
    fig.colorbar(im2, ax=ax2, shrink=0.8)  # 添加颜色条

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 预留标题空间
    # plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"平均注意力热力图已保存至：{save_path}")

def visualize_avg_head_attention(
    attentions,        # 模型输出的attentions，形状：(num_layers, batch_size, num_heads, seq_len, seq_len)
    token_regions,     # 字典，标记token区域（如{"sys": (0, 10), "img": (10, 586), "ins_out": (586, 600)}）
    tokenizer,         # 用于解码token的tokenizer
    input_ids, 
    layer_indices,     # 要可视化的层索引列表（如[1, 16, 32]）
    save_path="./avg_attention_layers.png",  # 图像保存路径
    batch_idx=0,        # 批次索引（单样本时设为0）
    tick_interval=5,   # 刻度间隔，控制显示多少个token下标
    specific_indices=None,
    max_label_len=10,   # 标签最大长度，过长会被截断
):
    """
    对每个层的所有注意力头取平均，绘制热力图并标记token区域
    """
    num_layers = len(attentions)
    # num_layers = attentions.shape[0]
    seq_len = attentions[0].shape[3]    
     # 获取当前批次的token ids
    batch_input_ids = input_ids[batch_idx].cpu().numpy()

    # 创建子图（数量与layer_indices一致）
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(16 * len(layer_indices), 16))
    axes = axes if isinstance(axes, np.ndarray) else [axes]  # 处理单子图情况

    # 生成刻度位置和标签（避免过于密集）
     # 确定要显示的刻度位置
    if specific_indices is not None:
        # 使用用户指定的索引，过滤掉超出范围的索引
        # ticks = [idx for idx in specific_indices if 0 <= idx < seq_len]
        # # 确保索引按顺序排列
        # ticks = sorted(ticks)
        ticks = specific_indices
    else:
        # 回退到使用固定间隔
        ticks = np.arange(0, seq_len, tick_interval)

    tick_labels = [str(i) for i in ticks]
    
    # pdb.set_trace()
    tick_texts = []
    for idx in ticks:
        # 解码单个token
        if 34 <= idx and idx <= 576+34:
            tick_texts.append(idx)
            continue
        if 576+34 < idx:
            idx -= 576
        token_text = tokenizer.decode(batch_input_ids[idx], skip_special_tokens=False)
        # 处理空格和特殊字符
        token_text = token_text.replace(' ', '').replace('\u2581', '▁')  # 处理字节对编码的空格
        # 截断过长文本
        if len(token_text) > max_label_len:
            token_text = token_text[:max_label_len] + '...'
        tick_texts.append(token_text)

    for i, layer_idx in enumerate(layer_indices):
        # 1. 聚合当前层所有head的注意力：在head维度取平均
        # avg_attention = attentions[layer_idx, batch_idx].mean(dim=0).cpu().detach().numpy()  
        avg_attention = attentions[layer_idx][batch_idx].mean(dim=0).cpu().detach().numpy()  
        # last_token_attention = avg_attention[-1: ]  # 提取最后一个Query → (1, seq_len)
        last_query_label = "Last Token"  # 明确标注为“最后一个token”
        # 形状变为：(seq_len, seq_len)

        # 2. 绘制热力图（对数刻度，匹配颜色范围0.001~1）
        ax = axes[i]
        sns.heatmap(
            avg_attention,
            # last_token_attention,
            ax=ax,
            cmap="viridis",        # 颜色映射与示例风格一致
            norm=LogNorm(vmin=0.001, vmax=1.0),  # 对数刻度，突出低权重与高权重差异
            cbar=(i == len(layer_indices)-1),  # 仅最后一个子图显示颜色条
            cbar_kws={"shrink": 0.8} if (i == len(layer_indices)-1) else None,
            xticklabels=False,     # 隐藏默认x轴token标签（后续手动标记区域）
            yticklabels=False      # 隐藏默认y轴token标签
        )

        # # 3. 设置x轴和y轴的刻度与标签
        # ax.set_xticks(ticks + 0.5)  # +0.5是为了对齐热力图单元格中心
        # ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=2)
        # ax.set_yticks(ticks + 0.5)
        # ax.set_yticklabels(tick_labels, fontsize=8)

        # 3. 设置x轴和y轴的刻度与文本标签
        ax.set_xticks(ticks + 0.5)  # +0.5是为了对齐热力图单元格中心
        ax.set_xticklabels(tick_texts, rotation=45, ha='right', fontsize=3)
        # ax.set_yticks(ticks + 0.5)
        # ax.set_yticklabels(tick_texts, fontsize=3)
        # y轴（仅最后一个Query）：固定1个刻度，标注“Last Token”
        ax.set_yticks([0.5])  # 仅中间位置1个刻度
        ax.set_yticklabels([last_query_label], fontsize=10)

        
        # 添加轴标签
        ax.set_xlabel("Token Text", fontsize=10)
        if i == 0:  # 仅第一个子图显示y轴标签
            ax.set_ylabel("Token Text", fontsize=10)
        
        # 添加轴标签
        ax.set_xlabel("Token Index", fontsize=10)
        if i == 0:  # 仅第一个子图显示y轴标签
            ax.set_ylabel("Token Index", fontsize=10)

        # # 3. 标记token区域的分隔线与标签
        # for region_name, (start, end) in token_regions.items():
        #     if start < seq_len and end <= seq_len:  # 确保区域在序列范围内
        #         # 垂直线（分隔不同token区域）
        #         ax.axvline(x=start, color='white', linewidth=1)
        #         ax.axvline(x=end, color='white', linewidth=1)
        #         # 水平线（分隔不同token区域）
        #         ax.axhline(y=start, color='white', linewidth=1)
        #         ax.axhline(y=end, color='white', linewidth=1)
                
        #         # 区域标签（y轴左侧 + x轴底部）
        #         if i == 0:  # 仅在第一个子图标记y轴区域
        #             ax.text(-0.05, (start + end) / 2 / seq_len, 
        #                     region_name, va='center', ha='right', 
        #                     transform=ax.transAxes, color='white')
        #         ax.text((start + end) / 2 / seq_len, -0.05, 
        #                 region_name, va='bottom', ha='center', 
        #                 transform=ax.transAxes, color='white')

        ax.set_title(f"Layer {layer_idx}")  # 层标题

    # 调整布局（避免标签重叠）
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"平均注意力热力图已保存至：{save_path}")

def visualize_masked_image(image_tensor, save_path):
    """
    将处理后的图像张量（已应用mask）保存为JPG文件
    
    参数:
        image_tensor: 处理后的图像张量，shape为 [C, H, W]，值范围通常为 [0,1] 或 [0,255]
        save_path: 保存路径（如 'masked_image.jpg'）
    """
    # 1. 若张量在GPU上，先转移到CPU并转为NumPy数组
    if image_tensor.device.type == 'cuda':
        image_np = image_tensor.cpu().detach().numpy()
    else:
        image_np = image_tensor.detach().numpy()
    
    # 2. 调整维度顺序：从 [C, H, W] 转为 [H, W, C]（图像库通常需要此格式）
    image_np = np.transpose(image_np, (1, 2, 0))  # 转换后 shape: [H, W, C]
    
    # 3. 调整数值范围：若为 [0,1] 则转为 [0,255] 并转为 uint8 类型
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)  # 若已为 [0,255] 直接转类型
    
    # 4. 处理单通道/多通道：若为单通道（灰度图），转为RGB以便保存为JPG
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)  # 单通道转三通道
    
    # 5. 保存为JPG
    Image.fromarray(image_np).save(save_path)
    print(f"已保存可视化图像到: {save_path}")

def get_prompts(args, model, tokenizer, data_demos, question, category='Object', model_is_llaval=True):
    if model_is_llaval:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        qs_pos = question
        qs_neg = question
        if hasattr(model.config, 'mm_use_im_start_end'): # True

            if model.config.mm_use_im_start_end: # False
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
            
            # conv_sys = conv_templates[args.conv_mode].copy()
            # conv_sys.get_prompt() "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "

            # prompts_positive  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            # # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this image in detail. ASSISTANT:The image depicts a girl wearing a swimsuit running on the beach, flying a kite joyfully. The kite can be seen in the air above her, giving a sense of motion to the scene. The beach includes a residential area or hotels nearby, as various chairs and benches are scattered around the sands, evoking a comfortable and relaxing atmosphere. Some chairs are located closer to the water, while others are set up further back in the scene, possibly to provide more privacy for beachgoers. The girl's activity and the presence of chairs and benches convey the perfect day for enjoying outdoor activities by the ocean."
            # prompts_negative  = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]
            # # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this image in detail. ASSISTANT:The image depicts a girl wearing a swimsuit running on the beach, flying a kite joyfully. The kite can be seen in the air above her, giving a sense of motion to the scene. The beach includes a residential area or hotels nearby, as various chairs and benches are scattered around the sands, evoking a comfortable and relaxing atmosphere. Some chairs are located closer to the water, while others are set up further back in the scene, possibly to provide more privacy for beachgoers. The girl's activity and the presence of beach umbrella convey the perfect day for enjoying outdoor activities by the ocean."

            # prompts_positive, prompts_negative = [], []
            # for data in data_demos:
            #     positives = data[category]
            #     desc = data['desc']
            #     # 执行删除操作
            #     negatives = remove_words_from_desc(positives, desc) 
            #     prompts_positive.append(conv_pos.get_prompt() + ", ".join(positives) + '.')
            #     prompts_negative.append(conv_neg.get_prompt() + negatives)

            ques = conv_pos.get_prompt().split(conv_pos.system)[-1]
            prompts_positive, prompts_negative = [], []
            for data in tqdm(data_demos, desc="Construct positives and negatives"):
                positive_shot = data['positive']
                negative_shot = data['negative']
     
                prompts_positive.append(positive_shot + ques)
                prompts_negative.append(negative_shot + ques)

                prompts_positive.append(positive_shot)
                prompts_negative.append(negative_shot)

            input_ids_positive = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_positive]
            input_ids_negative = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_negative]

        else:
            from transformers import InstructBlipProcessor
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

            input_ids_positive = []
            input_ids_negative = []

            for k in data_demos:
                image_path = os.path.join(args.data_file, 'train2014', k['image'])

                image_raw = Image.open(image_path).convert("RGB")
                input_ids_positive.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt").to(model.device))
                input_ids_negative.append(processor(images=image_raw, text=question + k['h_value'], return_tensors="pt").to(model.device))
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
        # pdb.set_trace()
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

def visualize_last_token_attention(
    pos_out_img,  # 模型输出（含attentions）
    img_token_slice=slice(34, 34+576),  # 图像token位置（复用你代码中的切片）
    target_layer=-1,  # 目标层：-1表示最后一层（全局注意力更有意义）
    head_strategy="max_sum",  # 多头处理策略：max_sum（选注意力总和最大的头）/ average（所有头平均）/ 指定头索引（如0）
    img_patch_size=(24, 24)  # 图像token对应的2D patch形状（576=24×24，必须与img_token数量匹配）
):
    """
    可视化最后一个token对图像token的注意力分数图
    Args:
        pos_out_img: 模型前向输出（需开启output_attentions=True）
        img_token_slice: 图像token在序列中的切片
        target_layer: 要可视化的注意力层（-1为最后一层）
        head_strategy: 多头注意力处理方式
        img_patch_size: 图像token对应的2D patch行列数（需满足 H×W = 图像token总数）
    """
    # -------------------------- 1. 验证输入合法性 --------------------------
    assert hasattr(pos_out_img, "attentions"), "model输出需开启output_attentions=True"
    assert len(pos_out_img.attentions) > 0, "未提取到注意力权重"
    # 验证图像token数量与2D patch形状匹配
    img_token_num = img_token_slice.stop - img_token_slice.start
    assert img_patch_size[0] * img_patch_size[1] == img_token_num, \
        f"patch尺寸不匹配：{img_patch_size[0]}×{img_patch_size[1]}≠{img_token_num}（图像token数）"

    # -------------------------- 2. 提取目标层的注意力矩阵 --------------------------
    # 注意力矩阵形状：[batch_size=1, num_heads, seq_len, seq_len]
    attentions = pos_out_img.attentions[target_layer]  # 选目标层（如最后一层）
    batch_size, num_heads, seq_len, _ = attentions.shape
    assert batch_size == 1, "当前仅支持单batch输入"

    # -------------------------- 3. 定位关键token索引 --------------------------
    last_token_idx = seq_len - 1  # 最后一个token（通常是EOS）的索引
    img_token_idx = torch.arange(img_token_slice.start, img_token_slice.stop, device=attentions.device)  # 图像token的索引列表

    # -------------------------- 4. 处理多头注意力：选择/聚合注意力头 --------------------------
    # 提取「最后一个token对所有图像token」的注意力分数：[num_heads, img_token_num]
    attn_scores = attentions[0, :, last_token_idx, img_token_idx].cpu().detach()  # 转CPU并脱离计算图

    # 按策略处理多头：
    if isinstance(head_strategy, int):
        # 1. 指定单个头（如head_strategy=5）
        assert 0 <= head_strategy < num_heads, f"头索引需在0~{num_heads-1}之间"
        selected_attn = attn_scores[head_strategy]
        title_suffix = f"（Head {head_strategy}）"
    elif head_strategy == "average":
        # 2. 所有头注意力平均
        selected_attn = attn_scores.mean(dim=0)
        title_suffix = "（所有头平均）"
    elif head_strategy == "max_sum":
        # 3. 选「注意力总和最大」的头（通常该头对图像关注最强）
        head_sum = attn_scores.sum(dim=1)  # 每个头的注意力总和：[num_heads]
        best_head = torch.argmax(head_sum).item()
        selected_attn = attn_scores[best_head]
        title_suffix = f"（最优头 Head {best_head}，注意力总和最大）"
    else:
        raise ValueError("head_strategy仅支持：int（头索引）/ 'average' / 'max_sum'")

    # -------------------------- 5. 整理为2D图像patch结构 --------------------------
    # 将1D注意力分数（576个）reshape为2D patch（24×24），贴合图像原始空间结构
    attn_2d = selected_attn.reshape(img_patch_size[0], img_patch_size[1])

    # -------------------------- 6. 可视化注意力热力图 --------------------------
    plt.figure(figsize=(10, 8))  # 画布大小
    # 自定义颜色映射：低注意力（蓝）→ 高注意力（红），更直观
    cmap = LinearSegmentedColormap.from_list("attn_cmap", ["#4575b4", "#ffffbf", "#d73027"])
    # 绘制热力图：interpolation='bilinear'使边缘平滑
    im = plt.imshow(attn_2d, cmap=cmap, interpolation="bilinear", aspect="equal")

    # 添加标注信息
    plt.title(
        f"最后一个token对图像patch的注意力分数\n"
        f"Layer {target_layer if target_layer >=0 else len(pos_out_img.attentions)+target_layer} {title_suffix}",
        fontsize=12, pad=15
    )
    plt.xlabel("图像Patch 列索引", fontsize=10)
    plt.ylabel("图像Patch 行索引", fontsize=10)
    # 添加颜色条（标注注意力强度）
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("注意力分数（越高表示越关注）", fontsize=10)

    # 可选：在热力图上标注注意力最高的5个patch位置
    top5_idx = torch.topk(selected_attn, k=5).indices  # 前5高注意力的1D索引
    top5_xy = [(idx % img_patch_size[1], idx // img_patch_size[1]) for idx in top5_idx]  # 转2D坐标
    for (x, y) in top5_xy:
        plt.scatter(x, y, color="white", s=30, marker="*", edgecolor="black", label="Top 5 Attention")
    # 避免重复标注图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    plt.tight_layout()
    # plt.show()  # 显示图像
    # 可选：保存图像
    plt.savefig("/home/zhangcs/zhangcs/code/VISTA/cache_utils/test.jpg", dpi=300, bbox_inches="tight")
    plt.close()

    return selected_attn, attn_2d  # 返回处理后的注意力分数（1D和2D）


def sample_fewshot_examples(example, dataset, args):
    """
    Sample the few-shot examples from the training set.

    Args:
        task_object: The task object
        num_fewshot: The number of few-shot examples
        example: The example to avoid

    Returns:
        The few-shot examples.
    """
    # num_fewshot_examples=5, sample_selection_method="random"
    fewshot_map = {}

    seed = args.seed
    random.seed(seed)
    sampled_docs = random.sample(dataset, args.num_fewshot_examples)
 
    images = [doc['image_file'] for doc in sampled_docs]
    while example['image_file'] in images:
        seed += 1
        random.seed(seed)
        sampled_docs = random.sample(dataset, args.num_fewshot_examples)
        images = [doc['image_file'] for doc in sampled_docs]

    # fewshot_map[example["id"]] = [e["id"] for e in sampled_docs]

    return sampled_docs

def process_instruct_example(tokenizer, fewshot_examples, coco, evaluator=None, type='Object', add_question=True):
        positive, negative = [], []
         
        # Add the system prompt
        # positive.append({"role": "system", "content": SYSTEM_MESSAGE})
        # negative.append({"role": "system", "content": SYSTEM_MESSAGE})

        # Add the few-shot examples
        # for fewshot_example in fewshot_examples:
        #     question = "Describe this image in detail."
        #     positive.append({"role": "user", "content": question})
        #     negative.append({"role": "user", "content": question})
        #     positive_answer = fewshot_example[type]

        #     # negative_answer = remove_words_from_desc(positive_answer, fewshot_example['desc']) 
        #     negative_answer = fewshot_example['desc']

        #     positive_shot = ", ".join(positive_answer) + '.'
        #     negative_shot = negative_answer

        #     # positive_answer, negative_answer = self.get_contrastive_answers(fewshot_example)
        #     positive.append({"role": "assistant", "content": positive_shot})
        #     negative.append({"role": "assistant", "content": negative_shot})

        # positive = tokenizer.apply_chat_template(positive, tokenize=False, add_generation_prompt=True)
        # negative = tokenizer.apply_chat_template(negative, tokenize=False, add_generation_prompt=True)
        # return positive, negative

        for fewshot_example in fewshot_examples:
            question = "Describe this image in detail."
            imageid = int(fewshot_example['image_file'].split('_')[-1]) 
            coco_objects = evaluator.imid_to_objects[imageid]

            positive_answer = fewshot_example[type]

            # pdb.set_trace()
            positive_answer = coco_objects
            negative_answer = fewshot_example['desc']

            positive.extend(positive_answer)
            negative.append(negative_answer)

            # negative_answer = remove_words_from_desc(positive_answer, fewshot_example['desc']) 
        
        positive_shot = ", ".join(positive) + '.'
        negative_shot = " ".join(negative) 

        return positive_shot, negative_shot


def process_image(image_processor, image_raw):
    answer = image_processor(image_raw)

    # Check if the result is a dictionary and contains 'pixel_values' key
    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]
    
    # Convert numpy array to torch tensor if necessary
    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    
    
    # If it's already a tensor, return it directly
    elif isinstance(answer, torch.Tensor):
        return answer
    
    else:
        raise ValueError("Unexpected output format from image_processor.")
    

    return answer

def get_token_to_append(steering_config: SteeringConfig, tokens, task=None):
    tokenizer = steering_config.tokenizer

    if steering_config.add_generation_prompt:
        if task in ['POPE']:
            token = tokenizer(":", add_special_tokens=False)["input_ids"][0]
        else:
            token = tokenizer(" ", add_special_tokens=False)["input_ids"][0]

    else:
        pdb.set_trace()
        if tokenizer.name_or_path in ["HuggingFaceTB/SmolLM2-360M-Instruct"]:
            token = tokenizer.bos_token_id

        elif tokenizer.name_or_path in [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        ]:
            token = 128006

    return torch.ones(tokens.shape[0], 1, device=tokens.device, dtype=tokens.dtype) * token

def compute_steering_similarity(steering_kv_text, steering_kv_img):
    """
    计算文本和图像steering keys和values的余弦相似度。
    
    参数:
        steering_kv_text: 文本steering字典，包含'values'和'keys'，每个是层ID到张量的映射。
        steering_kv_img: 图像steering字典，包含'values'和'keys'，每个是层ID到张量的映射。
        
    返回:
        similarities: 字典，键为层ID，值为包含values和keys相似度的字典（平均相似度和每个头的相似度列表）。
    """
    similarities = {}
    # 确保比较的层ID在两个字典中都存在
    common_layers = set(steering_kv_text['values'].keys()) & set(steering_kv_img['values'].keys())
    for layer_id in common_layers:
        # 获取文本和图像的values和keys张量
        text_values = steering_kv_text['values'][layer_id]
        img_values = steering_kv_img['values'][layer_id]
        text_keys = steering_kv_text['keys'][layer_id]
        img_keys = steering_kv_img['keys'][layer_id]
        
        # 确保张量形状为 [n_heads, head_dim]（如果使用PCA聚合，可能有多余维度）
        if text_values.dim() == 3:
            text_values = text_values.squeeze(0)
        if img_values.dim() == 3:
            img_values = img_values.squeeze(0)
        if text_keys.dim() == 3:
            text_keys = text_keys.squeeze(0)
        if img_keys.dim() == 3:
            img_keys = img_keys.squeeze(0)
        
        # 计算余弦相似度（每个头单独计算）
        sim_values = F.cosine_similarity(text_values, img_values, dim=1)  # 形状: [n_heads]
        sim_keys = F.cosine_similarity(text_keys, img_keys, dim=1)
        
        # 计算平均相似度
        avg_sim_values = sim_values.mean().item()
        avg_sim_keys = sim_keys.mean().item()
        
        # 存储结果
        similarities[layer_id] = {
            'values_similarity_per_head': sim_values.tolist(),
            'keys_similarity_per_head': sim_keys.tolist(),
            'avg_values_similarity': avg_sim_values,
            'avg_keys_similarity': avg_sim_keys
        }
    
    return similarities

def orthogonalize_per_layer(s_img_dict, s_txt_dict):
    # s_*_dict[layer_id] -> torch.Tensor of shape [H, Dh] or [1,H,Dh]
    out = {}
    for lid in s_txt_dict.keys():
        s_img = s_img_dict.get(lid, None)
        s_txt = s_txt_dict[lid]
        if s_img is None: 
            out[lid] = s_txt
            continue
        s_img_f = s_img.reshape(-1)
        s_txt_f = s_txt.reshape(-1)
        # 避免全零
        if torch.norm(s_img_f) < 1e-8 or torch.norm(s_txt_f) < 1e-8:
            out[lid] = s_txt
            continue
        u = s_img_f / (torch.norm(s_img_f) + 1e-8)
        v = s_txt_f
        v_perp = v - (torch.dot(v, u) * u)
        if torch.norm(v_perp) < 1e-8:
            v_perp = v  # 几乎共线时退回原向量
        v_perp = v_perp / torch.norm(v_perp)
        out[lid] = v_perp.reshape_as(s_txt)
    return out

def repeat_past_for_beams(past, num_beams: int):
    """
    将 KV 缓存按 batch 维复制到 beam 大小。
    同时兼容 DynamicCache 和旧式 tuple[(k,v), ...]。
    """
    if isinstance(past, DynamicCache):
        new_past = DynamicCache()
        new_past.key_cache   = [k.repeat_interleave(num_beams, dim=0) for k in past.key_cache]
        new_past.value_cache = [v.repeat_interleave(num_beams, dim=0) for v in past.value_cache]
        # 兼容部分版本：如果有其他属性，比如 seen_tokens 等，一般不需要手动改
        return new_past
    else:
        # 旧式 past_key_values: tuple of (k, v)
        return tuple(
            (k.repeat_interleave(num_beams, dim=0), v.repeat_interleave(num_beams, dim=0))
            for (k, v) in past
        )


def visualize_tensor_comparison(tensor1, tensor2, names=["Tensor 1", "Tensor 2"]):
    """
    可视化对比两个形状为(1, 32, 128)的tensor的数值范围趋势
    
    参数:
        tensor1, tensor2: 输入的两个tensor
        names: 两个tensor的名称，用于图表标注
    """
    # 转换为numpy数组并去除大小为1的维度 (1,32,128) -> (32,128)
    arr1 = tensor1.squeeze().cpu().numpy() if isinstance(tensor1, torch.Tensor) else tensor1.squeeze()
    arr2 = tensor2.squeeze().cpu().numpy() if isinstance(tensor2, torch.Tensor) else tensor2.squeeze()
    
    # 确保形状正确
    assert arr1.shape == (32, 128) and arr2.shape == (32, 128), "输入tensor形状必须为(1,32,128)"
    
    # 创建画布
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 热图展示整体数值分布
    ax1 = fig.add_subplot(221)
    sns.heatmap(arr1, cmap="viridis", ax=ax1)
    ax1.set_xlabel("dims (128)")
    ax1.set_ylabel("length (32)")
    
    ax2 = fig.add_subplot(222)
    sns.heatmap(arr2, cmap="viridis", ax=ax2)
    ax2.set_xlabel("dims (128)")
    ax2.set_ylabel("length (32)")
    
    # 2. 计算每行的统计量（均值、最大、最小）用于趋势分析
    stats1 = {
        "mean": arr1.mean(axis=1),    # 按列求均值，得到(32,)
        "max": arr1.max(axis=1),
        "min": arr1.min(axis=1)
    }
    
    stats2 = {
        "mean": arr2.mean(axis=1),
        "max": arr2.max(axis=1),
        "min": arr2.min(axis=1)
    }
    
    # 绘制趋势线
    ax3 = fig.add_subplot(212)
    x = np.arange(32)  # 32个时间步/行
    
    # 绘制均值趋势
    ax3.plot(x, stats1["mean"], label=f"{names[0]} mean", color="blue", linestyle="-")
    ax3.plot(x, stats2["mean"], label=f"{names[1]} mean", color="orange", linestyle="-")
    
    # 绘制最大值和最小值范围
    ax3.fill_between(x, stats1["min"], stats1["max"], color="blue", alpha=0.2, label=f"{names[0]} range")
    ax3.fill_between(x, stats2["min"], stats2["max"], color="orange", alpha=0.2, label=f"{names[1]} range")
    
    ax3.set_xlabel("length (32)")
    ax3.set_ylabel("value")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
   
    
    # 打印关键统计信息
    print(f"{names[0]} range: [{arr1.min():.4f}, {arr1.max():.4f}], mean: {arr1.mean():.4f}")
    print(f"{names[1]} range: [{arr2.min():.4f}, {arr2.max():.4f}], mean: {arr2.mean():.4f}")

    plt.tight_layout()
    plt.savefig("/home/zhangcs/zhangcs/code/VISTA/cache_utils/test.jpg", dpi=300, bbox_inches="tight")


def extract_steering_kv_img_v2(
    model,
    tokenizer,
    input_images,
    contrast_images,
    input_ids,
    steering_config: SteeringConfig,
    device="cpu",
):
    # 初始化存储结构：新增token维度（默认保留所有token差异）
    steering_values_img = defaultdict(lambda: torch.tensor([]))  # 最终shape: [num_demos, n_tokens, n_heads, head_dim]
    steering_keys_img = defaultdict(lambda: torch.tensor([]))    # 最终shape: [num_demos, n_tokens, n_heads, head_dim]
    output_dict_img = {}

    # 先获取图像token总数（从第一个样本推断，假设所有样本token数一致）
    # 提取图像token切片（原代码中34:34+576，共576个图像patch token）
    img_token_slice = slice(34, 34 + 5)
    n_tokens = img_token_slice.stop - img_token_slice.start  # 图像token数量：576

    for example_id in tqdm(range(len(contrast_images)), desc='Obtaining visual direction (with token diff)'):
        neg_tokens, pos_tokens = input_ids[example_id]
        neg_images, pos_images = contrast_images[example_id]
        batch_indices = torch.arange(pos_tokens.size(0), device=pos_tokens.device)

        # 初始化缓存
        cache_positive_img, cache_negative_img = DynamicCache(), DynamicCache()
        with torch.no_grad():
            # 前向传播获取KV缓存和隐藏状态
            pos_kwargs_img = {'input_ids': pos_tokens, 'images': pos_images.unsqueeze(0).half().to(pos_tokens.device)}
            neg_kwargs_img = {'input_ids': neg_tokens, 'images': neg_images.unsqueeze(0).half().to(pos_tokens.device)}
            pos_out_img = model(**pos_kwargs_img, output_hidden_states=True, output_attentions=True, past_key_values=cache_positive_img)
            neg_out_img = model(**neg_kwargs_img, output_hidden_states=True, output_attentions=True, past_key_values=cache_negative_img)

            # visualize_last_token_attention(pos_out_img=pos_out_img,img_token_slice=slice(0, 841),target_layer=-1, head_strategy="max_sum",img_patch_size=(29, 29))
            # visualize_masked_image(pos_images[0], '/home/zhangcs/zhangcs/code/VISTA/cache_utils/test.jpg')
            # pdb.set_trace()


        # 遍历每个层，提取KV并保留token差异
        for layer_id in range(len(cache_positive_img.value_cache)):
            # -------------------------- 关键修改1：取消token维度的mean，保留所有token --------------------------
            # 原代码：.mean(2) → 抹除token差异；修改后：删除.mean(2)，保留token维度（第2维）
            # pos_values_img shape: [batch_size, n_heads, n_tokens, head_dim]
            pos_values_img = cache_positive_img.value_cache[layer_id][batch_indices, :, img_token_slice, :]
            # neg_values_img shape: [batch_size, n_heads, n_tokens, head_dim]
            neg_values_img = cache_negative_img.value_cache[layer_id][batch_indices, :, img_token_slice, :]
            
            # 同理处理key
            pos_keys_img = cache_positive_img.key_cache[layer_id][batch_indices, :, img_token_slice, :]
            neg_keys_img = cache_negative_img.key_cache[layer_id][batch_indices, :, img_token_slice, :]

            # -------------------------- 关键修改2：调整维度顺序，便于后续按token处理 --------------------------
            # 从 [batch_size, n_heads, n_tokens, head_dim] → [batch_size, n_tokens, n_heads, head_dim]
            # 目的：将token维度提前，后续按token分组计算PCA
            pos_values_img = pos_values_img.permute(0, 2, 1, 3)  # 交换n_heads和n_tokens维度
            neg_values_img = neg_values_img.permute(0, 2, 1, 3)
            pos_keys_img = pos_keys_img.permute(0, 2, 1, 3)
            neg_keys_img = neg_keys_img.permute(0, 2, 1, 3)


            # 计算正例-负例差异（保留token维度）
            if steering_config.take_difference:
                diff_values = pos_values_img - neg_values_img  # [batch_size, n_tokens, n_heads, head_dim]
                diff_keys = pos_keys_img - neg_keys_img        # [batch_size, n_tokens, n_heads, head_dim]
                # 拼接当前样本到全局存储（按demo维度堆叠）
                steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], diff_values.detach().cpu()], dim=0)
                steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], diff_keys.detach().cpu()], dim=0)
            else:
                steering_values_img[layer_id] = torch.cat([steering_values_img[layer_id], pos_values_img], dim=0)
                steering_keys_img[layer_id] = torch.cat([steering_keys_img[layer_id], pos_keys_img], dim=0)

    # -------------------------- 关键修改3：聚合时按token单独计算（mean/PCA） --------------------------
    if steering_config.aggregation_method == AggregationMethods.mean:
        for layer_id in steering_values_img:
            # 原shape: [num_demos, n_tokens, n_heads, head_dim]
            # 按demo维度求平均 → 保留token差异，shape: [n_tokens, n_heads, head_dim]
            steering_values_img[layer_id] = torch.mean(steering_values_img[layer_id], dim=0)
            steering_keys_img[layer_id] = torch.mean(steering_keys_img[layer_id], dim=0)
        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    elif steering_config.aggregation_method == AggregationMethods.pca:
        for layer_id in steering_values_img:
            # 1. 处理Values：按token单独做PCA
            values_data = steering_values_img[layer_id]  # shape: [num_demos, n_tokens, n_heads, head_dim]
            num_demos, n_tokens, n_heads, head_dim = values_data.shape
            values_direction = []  # 存储每个token的PCA direction

            # 遍历每个token，单独计算该token的PCA方向
            for token_idx in range(n_tokens):
                # 提取当前token的所有demo数据：[num_demos, n_heads*head_dim]（展平heads和head_dim）
                token_values = values_data[:, token_idx, :, :].view(num_demos, -1)
                # 拟合PCA（n_components=1）
                pca = PCA(n_components=1).to(token_values.device).fit(token_values.float())
                # 恢复维度：[1, n_heads, head_dim] → 匹配原KV结构
                token_dir = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0).view(1, n_heads, head_dim)
                values_direction.append(token_dir)
            
            # 拼接所有token的direction → shape: [n_tokens, n_heads, head_dim]
            steering_values_img[layer_id] = torch.cat(values_direction, dim=0)

            # 2. 同理处理Keys：按token单独做PCA
            keys_data = steering_keys_img[layer_id]  # shape: [num_demos, n_tokens, n_heads, head_dim]
            keys_direction = []
            for token_idx in range(n_tokens):
                token_keys = keys_data[:, token_idx, :, :].view(num_demos, -1)
                pca = PCA(n_components=1).to(token_keys.device).fit(token_keys.float())
                token_dir = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0).view(1, n_heads, head_dim)
                keys_direction.append(token_dir)
            steering_keys_img[layer_id] = torch.cat(keys_direction, dim=0)

        output_dict_img["values"] = dict(steering_values_img)
        output_dict_img["keys"] = dict(steering_keys_img)

    # pdb.set_trace()
    return output_dict_img

# 保存函数
def save_steering_data(values, keys, save_path, device='cuda'):
    # 为避免lambda序列化问题，先转换为普通dict并保存设备信息
    data = {
        "values": {k: v.cpu() for k, v in values.items()},  # 移到CPU保存
        "keys": {k: v.cpu() for k, v in keys.items()},
        "device": str(device)  # 记录原始设备
    }
    torch.save(data, save_path)  # 用torch.save处理张量更可靠


# 读取函数
def load_steering_data(load_path, device='cuda'):
    data = torch.load(load_path)
    # 重建defaultdict，指定默认工厂（避免lambda序列化问题）
    def default_tensor():
        return torch.tensor([]).to(device)
    
    steering_values = defaultdict(default_tensor)
    steering_keys = defaultdict(default_tensor)
    
    # 恢复数据并移到目标设备
    for k, v in data["values"].items():
        steering_values[k] = v.to(device)
    for k, v in data["keys"].items():
        steering_keys[k] = v.to(device)
    
    return steering_values, steering_keys