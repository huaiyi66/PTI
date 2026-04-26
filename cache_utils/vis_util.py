import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import cv2  # 用于resize
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers import DynamicCache
import pdb
import seaborn as sns

# ---------- 可配置（与你工程保持一致） ----------
IMG_TOKEN_START = 34
IMG_TOKEN_LEN   = 576     # 通常对应 24x24 patch
IMG_TOKEN_SLICE = slice(IMG_TOKEN_START, IMG_TOKEN_START + IMG_TOKEN_LEN)
PATCH_GRID_HW   = (24, 24)  # 24x24
EPS = 1e-8

# 你工程里的设定
IMG_SIZE   = 336         # 输入图分辨率
PATCH_SIZE = 14          # ViT patch 边长
GRID_H = GRID_W = IMG_SIZE // PATCH_SIZE   # 24


def resize_mask_to_336(mask_3chw: torch.Tensor, target_size=(3, 336, 336)) -> torch.Tensor:
    """
    将 mask 从任意尺寸调整为 (3, 336, 336)
    - mask_3chw: [C, H, W]，原始的掩码张量
    - target_size: 目标尺寸，默认为 (3, 336, 336)
    返回：[3, 336, 336] 大小的掩码张量
    """
    # 确保掩码张量是 float 类型，避免后续操作出现问题
    mask_3chw = mask_3chw.float()

    # 目标形状：target_size 应该是 (C, H, W)
    C, H, W = mask_3chw.shape
    target_C, target_H, target_W = target_size

    # 如果输入通道数与目标通道数不匹配，进行通道扩展
    if C != target_C:
        raise ValueError(f"Input mask has {C} channels, but target has {target_C} channels.")
    
    # 使用 F.interpolate 对图像进行 resize
    mask_resized = F.interpolate(mask_3chw.unsqueeze(0), size=(target_H, target_W), mode='nearest')  # [1, C, target_H, target_W]
    return mask_resized.squeeze(0)  # 返回 [C, target_H, target_W]


def mask3chw_to_object_mask_24x24(mask_3chw: torch.Tensor,
                                  threshold: float = 0.5,
                                  channel_reduce: str = "max",
                                  binarize: bool = True) -> torch.Tensor:
    """
    将 (3,336,336) 的像素级 mask 转成 (24,24) 的 patch 级 mask（物体=1, 背景=0）。
    - mask_3chw: torch.Tensor, 形状 [3,336,336]，取值可为 {0,1} 或 [0,1]
    - channel_reduce: 'max' | 'mean'。有些 mask 三通道相同，用 'max' 更稳。
    - threshold: 降采样后阈值，大于该值视为该 patch 包含物体
    - binarize: 是否二值化。若想保留“物体覆盖比”，设为 False 返回 [0,1] 概率图
    返回: torch.Tensor, 形状 [24,24]，dtype=float32
    """
    assert mask_3chw.dim() == 3 and mask_3chw.shape[-2:] == (IMG_SIZE, IMG_SIZE), \
        f"expected (3,{IMG_SIZE},{IMG_SIZE}), got {tuple(mask_3chw.shape)}"

    # 1) 通道融合到 [1,336,336]
    if channel_reduce == "max":
        mask_hw = mask_3chw.max(dim=0).values
    elif channel_reduce == "mean":
        mask_hw = mask_3chw.mean(dim=0)
    else:
        raise ValueError("channel_reduce must be 'max' or 'mean'")

    # 保证范围在 [0,1]
    mask_hw = mask_hw.clamp(0, 1).unsqueeze(0).unsqueeze(0)  # [1,1,336,336]

    # 2) 降采样到 [1,1,24,24]
    #   最近邻：更“离散”；双线性：更“平滑”。若想精确按 patch 求平均，见下一段 block reduce 实现。
    # obj24 = F.interpolate(mask_hw, size=(GRID_H, GRID_W), mode='nearest')  # 或 'bilinear', align_corners=False
    # 更严格的“按 14×14 patch 求平均”的 block reduce：
    B,C,H,W = mask_hw.shape
    assert H % PATCH_SIZE == 0 and W % PATCH_SIZE == 0, "H/W must be multiples of PATCH_SIZE"
    mask_blocks = mask_hw.view(B, C, GRID_H, PATCH_SIZE, GRID_W, PATCH_SIZE) \
                        .mean(dim=(3,5))  # [1,1,24,24]
    obj24 = mask_blocks[0,0]               # [24,24], float32 in [0,1]

    # 3) 二值化（可选）
    if binarize:
        obj24 = (obj24 > threshold).float()

    return obj24  # [24,24]

def object_background_masks_from_3chw(mask_3chw: torch.Tensor,
                                      **kwargs):
    """
    返回 (obj_24x24, bg_24x24) 两个 24×24 掩码
    """
    obj = mask3chw_to_object_mask_24x24(mask_3chw, **kwargs)   # [24,24]
    bg  = 1.0 - obj
    return obj, bg


# =============== 1) 工具函数：定量矩阵（layer×head） ===============
@torch.no_grad()
def layer_head_metrics_value_cache(cache_before: DynamicCache,
                                   cache_after: DynamicCache,
                                   token_slice=IMG_TOKEN_SLICE):
    """
    只看 value_cache，在图像 token 段上比较 before vs after：
      - cosine: 每个 head 对各 token 的余弦相似度，再在 token 维平均 → [L,H]
      - norm ratio: ||after||/||before||（沿 D 求范数，再 token 平均） → [L,H]
    """
    L = len(cache_before.value_cache)
    BIDX = 0  # 取第一个 batch（如有 beams，请在外部先对齐）


    val_cos, val_ratio = [], []
    for l in range(L):
        v0 = cache_before.value_cache[l][BIDX, :, token_slice, :].float()   # [H, T, D]
        v1 = cache_after .value_cache[l][BIDX, :, token_slice, :].float()   # [H, T, D]
        H, T, D = v0.shape

        # 余弦：先按 token 逐个算，再在 token 维平均（得到每头一个值）
        a = F.normalize(v0.reshape(H*T, D), dim=-1)
        b = F.normalize(v1.reshape(H*T, D), dim=-1)
        cos = (a*b).sum(-1).view(H, T).mean(dim=1)          # [H]
        # 范数比：先沿 D 求范数，再 token 平均
        n0 = torch.linalg.vector_norm(v0, ord=2, dim=-1)    # [H,T]
        n1 = torch.linalg.vector_norm(v1, ord=2, dim=-1)
        ratio = ((n1+EPS)/(n0+EPS)).mean(dim=1)             # [H]

        val_cos.append(cos.cpu().numpy())
        val_ratio.append(ratio.cpu().numpy())

    return {
        "val_cos":        np.stack(val_cos, axis=0),     # [L,H]
        "val_norm_ratio": np.stack(val_ratio, axis=0),   # [L,H]
    }

def plot_layer_head_heatmaps(M, title_prefix="Image Steering (Values)"):
    mats = [
        ("cosine(before, after)", M['val_cos'],        "coolwarm", (-1, 1)),
        ("norm ratio (after/before)", M['val_norm_ratio'], "viridis", None),
    ]
    L, H = M['val_cos'].shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (ttl, mat, cmap, vlim) in zip(axes, mats):
        im = ax.imshow(mat.T, aspect='auto', origin='lower', cmap=cmap,
                       vmin=(vlim[0] if vlim else None),
                       vmax=(vlim[1] if vlim else None))
        ax.set_title(f"{title_prefix} — {ttl}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Head")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.show()
    plt.savefig("/home/zhangcs/zhangcs/code/VISTA/cache_utils/test_vis_value.jpg", dpi=300, bbox_inches="tight")


# =============== 2) 工具函数：方向投影的 patch 热图（定性） ===============
@torch.no_grad()
def steering_projection_maps(v,
                             steering_vec_layer: torch.Tensor,
                             head_reduce: str = "mean",
                             ):
    """
    将 value_cache 在某层的“每个图像 token 向量”投影到 steering 方向上，得到长度为 576 的分数，
    再 reshape 为 24x24 热图。
      steering_vec_layer: [1, H, D] 或 [H, D]（来自你的 steering_kv_img['values'][layer]）
      head_reduce: 'mean' | 'max' | 'none'
    返回：proj_before, proj_after （若只传一个 cache 就返回该 cache 的投影）
    """
    # v = cache.value_cache[layer_idx][0, :, token_slice, :].float()  # [H, T, D]
    # v = cache.key_cache[layer_idx][0, :, token_slice, :].float()  # [H, T, D]
    H, T, D = v.shape
    # 方向单位化
    # pdb.set_trace()
    # sv_layer = steering_kv_img["values"][target_layer_idx]  # e.g., -1 取最后一层

    sv = steering_vec_layer
    if sv.dim() == 3: sv = sv[0]
    sv = F.normalize(sv.float(), dim=-1)              # [H,D]

    # 每个 token 在每头上的投影（内积）
    # [H,T,D] ⋅ [H,D] -> [H,T]
    proj = (F.normalize(v, dim=-1) * sv.unsqueeze(1)).sum(-1)  # [H,T]

    if head_reduce == "mean":
        proj = proj.mean(dim=0)   # [T]
    elif head_reduce == "max":
        proj = proj.max(dim=0).values
    elif head_reduce == "none":
        pass
    else:
        raise ValueError("head_reduce must be in {'mean','max','none'}")

    # 归一化到 [0,1] 便于显示
    p = proj
    if p.dim() == 1:
        p = (p - p.min()) / (p.max() - p.min() + EPS)  # [T]
        p = p.view(PATCH_GRID_HW)                       # 24x24
    else:
        # 不聚头：返回 [H,24,24]
        p = (p - p.min(dim=1, keepdim=True).values) / \
            (p.max(dim=1, keepdim=True).values - p.min(dim=1, keepdim=True).values + EPS)
        p = p.view(H, *PATCH_GRID_HW)
    return p.cpu().numpy()

# def show_patch_map(map2d, title="Projection on steering direction", save_path="/home/zhangcs/zhangcs/code/VISTA/cache_utils/test_Projection.jpg"):
#     plt.figure(figsize=(4.5,4.5))
#     plt.imshow(map2d, origin="lower", aspect="equal")
#     plt.title(title)
#     plt.axis("off")
#     plt.colorbar(fraction=0.046, pad=0.04)
#     # plt.show()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")

def show_patch_map(map2d, title="Projection on steering direction", save_path=None):
    """
    map2d: [24,24] 或 [H,24,24]（未聚头）
    若是 [H,24,24]，自动拼成多子图网格显示。
    """
    if isinstance(map2d, torch.Tensor):
        map2d = map2d.detach().cpu().numpy()

    if map2d.ndim == 2:
        plt.figure(figsize=(4.5, 4.5))
        plt.imshow(map2d, origin="lower", aspect="equal")
        plt.title(title)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    elif map2d.ndim == 3:
        H = map2d.shape[0]
        cols = min(8, H)
        rows = (H + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.2))
        axes = axes.ravel() if isinstance(axes, (list, np.ndarray)) else [axes]
        for i in range(rows*cols):
            ax = axes[i]
            ax.axis("off")
            if i < H:
                im = ax.imshow(map2d[i], origin="lower", aspect="equal")
                ax.set_title(f"Head {i}", fontsize=9)
        fig.suptitle(title)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        raise ValueError(f"Unexpected map shape: {map2d.shape}")

@torch.no_grad()
def last_step_attn_object_vs_background(out,
                                        object_mask_24x24: torch.Tensor,
                                        return_layer=-1,
                                        step=-1):
    """
    用注入过的 past_cache 做“一步解码”，拿到最后一步的 attentions，
    统计图像 token 段中“物体区域 vs 背景区域”的注意力占比。
    - object_mask_24x24: [24,24]，二值或0/1（会展平到576对齐图像token）
    返回：(obj_pct_per_head, bg_pct_per_head, total_pct_img_tokens)
    """
    # 只喂最后一个 token 进入解码
    # last_input_ids = input_ids[:, -1:]
    # import pdb
    # pdb.set_trace()
    # out = model(input_ids=last_input_ids, images=images, past_key_values=past_cache, output_attentions=True, use_cache=True, return_dict=True)

    # attentions: list[L]，每层 [B, H, Q, K]

    # attn = out.attentions[return_layer][0]  # [H, Q, K]
    # # 取解码 query 的最后一个位置（Q 的最后一维）
    # attn_last = attn[:, -1, :]             # [H, K]

    attn = out.attentions[step]  # [H, Q, K]
    # 取解码 query 的最后一个位置（Q 的最后一维）
    attn_last = attn[return_layer][0, :, 0, :]           # [H, K]

    # 取图像 token 段
    img_attn = attn_last[:, IMG_TOKEN_START:IMG_TOKEN_START+IMG_TOKEN_LEN]  # [H, 576]
    img_attn = img_attn / (img_attn.sum(dim=1, keepdim=True) + EPS)         # 归一化到图像段内

    # 物体/背景掩码
    m = object_mask_24x24.to(img_attn.device).reshape(-1).float()   # [576]
    m = (m > 0.5).float()
    obj_mass = (img_attn * m.unsqueeze(0)).sum(dim=1)               # [H]
    bg_mass  = (img_attn * (1-m).unsqueeze(0)).sum(dim=1)           # [H]
    # 整体“看图像 token”的比例（相对全 K）
    total_mass_img = attn_last[:, IMG_TOKEN_START:IMG_TOKEN_START+IMG_TOKEN_LEN].sum(dim=1) / (attn_last.sum(dim=1)+EPS)

    return obj_mass.cpu().numpy(), bg_mass.cpu().numpy(), total_mass_img.cpu().numpy()


def bar_three_curves(obj_before, obj_after, bg_before, bg_after, ttl="Attn to object vs background (per head)"):
    H = len(obj_before)
    x = np.arange(H)
    w = 0.35
    plt.figure(figsize=(10,4.5))
    plt.bar(x-w/2, obj_before, width=w, edgecolor="black", label="object (before)")
    plt.bar(x+w/2, obj_after,  width=w, edgecolor="black", label="object (after)")
    plt.plot(x, bg_before, marker="o", linestyle="--", label="background (before)")
    plt.plot(x, bg_after,  marker="o", linestyle="-.", label="background (after)")
    plt.title(ttl)
    plt.xlabel("Head")
    plt.ylabel("Attention mass within image tokens")
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("/home/zhangcs/zhangcs/code/VISTA/cache_utils/test_Attn_object_bg.jpg", dpi=300, bbox_inches="tight")


# =============== 4) 一键主流程：产出三类证据 ===============
# @torch.no_grad()
# def analyze_image_steering_values(model,
#                                   input_ids: torch.Tensor,
#                                   images: torch.Tensor,
#                                   steering_kv_img: dict,
#                                   steering_config,
#                                   object_mask_24x24: torch.Tensor,
#                                   past_before,
#                                   past_after,
#                                   out_before,
#                                   out_after,
#                                   target_layer_idx: int = -1):
@torch.no_grad()
def analyze_image_steering_values(
                                  object_mask_24x24: torch.Tensor,
                                  out_before,
                                  out_after,
                                  ):
    """
    只对 value cache 的图像 token 段注入 steering（来自“物体vs背景”的差分），
    给出：
      (i) layer×head 的 cosine / norm ratio 热力图；
      (ii) 图像 token 在 steering 方向上的投影热图（before/after）；
      (iii) 最后一步解码对图像 token 的注意力在物体vs背景的占比变化。
    你需要把你工程里的 precompute_kv_cache 与 steer_kv_cache 传进来。
    """
  

    # A) 定量：layer×head
    # M = layer_head_metrics_value_cache(past_before, past_after, token_slice=IMG_TOKEN_SLICE)
    # plot_layer_head_heatmaps(M, title_prefix="Image Steering (Values)")

    # # B) 定性：投影热图（target layer）
    # #   取该层的 steering 向量（values），形如 [1,H,D] 或 [H,D]
    # import pdb
    # # sv_layer = steering_kv_img["values"][target_layer_idx]  # e.g., -1 取最后一层
    # sv_layer = steering_kv_img["values"][target_layer_idx]  # torch.Size([1, 32, 128])
    # past_before_key = past_before.value_cache[target_layer_idx][0, :, IMG_TOKEN_SLICE, :].float() # torch.Size([32, 576, 128])
    # past_after_key = past_after.value_cache[target_layer_idx][0, :, IMG_TOKEN_SLICE, :].float()  # torch.Size([32, 576, 128])
    # map_before = steering_projection_maps(past_before_key, sv_layer, head_reduce="mean")
    # map_after  = steering_projection_maps(past_after_key , sv_layer, head_reduce="mean")
    # show_patch_map(map_before, title=f"Projection BEFORE (layer {target_layer_idx})", save_path="/home/zhangcs/zhangcs/code/VISTA/cache_utils/test_Projection_before.jpg")
    # show_patch_map(map_after , title=f"Projection AFTER  (layer {target_layer_idx})", save_path="/home/zhangcs/zhangcs/code/VISTA/cache_utils/test_Projection_after.jpg")

    # C) 注意力：最后一步对图像 token 的路由（物体vs背景）
    # obj_b, bg_b, imgmass_b = last_step_attn_object_vs_background(
    #     model, input_ids, images, past_before, object_mask_24x24, return_layer=target_layer_idx
    # )
    # # obj_a, bg_a, imgmass_a = last_step_attn_object_vs_background(
    # #     model, input_ids, images, past_after, object_mask_24x24, return_layer=target_layer_idx
    # # )
    # for step in range(min(len(out_before.attentions), len(out_after.attentions))):
    #     obj_b, bg_b, imgmass_b = last_step_attn_object_vs_background(out_before, object_mask_24x24, return_layer=target_layer_idx, step=step)
    #     obj_a, bg_a, imgmass_a = last_step_attn_object_vs_background(out_after, object_mask_24x24, return_layer=target_layer_idx, step=step)
    #     print(f"step {step}")
    #     print(f"  Δ object-attn (mean over heads)     = {(obj_a-obj_b).mean():+.4f}")
    #     print(f"  Δ background-attn (mean over heads) = {(bg_a-bg_b).mean():+.4f}")
    #     print(f"  Δ total image-attn (mean over heads)= {(imgmass_a-imgmass_b).mean():+.4f}")

    # (1) 最后一个 step 的 Δ热力图
    d_obj, d_bg, d_tot = None, None, None
    d_obj, d_bg, d_tot, obj_b, bg_b, tot_b, obj_a, bg_a, tot_a = plot_last_step_heatmaps(
        out_before,
        out_after,
        object_mask_24x24,
        save_path="/home/zhangcs/zhangcs/code/VISTA/cache_utils/last_step_obj_bg_delta.png",   # 可选
    )


    mb_list, ma_list  = plot_total_image_attn_change_per_step(out_before, out_after, object_mask_24x24, title="Δ total image-attn per step", relative=True,  n_steps=80, 
                                                              save_path="/home/zhangcs/zhangcs/code/VISTA/cache_utils/all_step_attn_delta.png"
                                                              )
    
    return d_obj, d_bg, d_tot,  mb_list, ma_list, obj_b, bg_b, tot_b, obj_a, bg_a, tot_a

    # 可视化 per-head 的对象/背景注意力变化
    # bar_three_curves(obj_b, obj_a, bg_b, bg_a,
    #                  ttl=f"Object vs Background attention (layer {target_layer_idx})")

    # # 同时打印几个摘要指标
    # print("[Summary]")
    # print(f"Layer {target_layer_idx}:")
    # print(f"  mean cosine(before,after) over heads = {M['val_cos'][target_layer_idx].mean():.4f}")
    # print(f"  mean norm ratio (after/before)      = {M['val_norm_ratio'][target_layer_idx].mean():.4f}")
    # print(f"  Δ object-attn (mean over heads)     = {(obj_a-obj_b).mean():+.4f}")
    # print(f"  Δ background-attn (mean over heads) = {(bg_a-bg_b).mean():+.4f}")
    # print(f"  Δ total image-attn (mean over heads)= {(imgmass_a-imgmass_b).mean():+.4f}")



import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 你已有的常量
# IMG_TOKEN_START = 34
# IMG_TOKEN_LEN   = 576
# EPS = 1e-8

# ---------- (A) 逐层逐头统计：object/bg/total image-attn ----------
@torch.no_grad()
def attn_object_bg_all_layers(out, object_mask_24x24: torch.Tensor, step: int):
    """
    对某个 step，统计所有 layer、所有 head 上：
      - object-attn mass （图像段内落在物体mask上的注意力占比）
      - background-attn mass （图像段内落在背景上的占比）
      - total image-attn mass （相对全K，落在图像token段的注意力占比）
    返回：
      obj_mat  : [L, H]
      bg_mat   : [L, H]
      total_mat: [L, H]
    """
    L = len(out.attentions[step])               # 层数
    obj_list, bg_list, total_list = [], [], []

    m = object_mask_24x24.reshape(-1).float().to(out.attentions[step][0].device)  # [576]
    m = (m > 0.5).float()

    for layer in range(L):
        # attn_layer: [B, H, Q, K]
        attn_layer = out.attentions[step][layer]    # tensor
        # 取 batch=0, query 的第 0 个位置
        attn_last = attn_layer[0, :, -1, :]          # [B, H, K, L]  torch.Size([1, 32, 1, 729])

        # try:
        #     attn_last = attn_layer[0, :, 615, :]          # [B, H, K, L]  torch.Size([1, 32, 1, 729])
        # except:
        #     pdb.set_trace()

        # img_slice = slice(IMG_TOKEN_START, IMG_TOKEN_START + IMG_TOKEN_LEN)
        img_slice = slice(34, 34 + 576)
        # img_slice = slice(41, 41 + 576)
        # 图像段（用来归一化object/bg分配）
        img_attn = attn_last[:, img_slice]          # [H, L]
        
        img_attn_norm = img_attn / (img_attn.sum(dim=1, keepdim=True))
        # img_attn_norm = img_attn 

        obj_mass = (img_attn_norm * m.unsqueeze(0)).sum(dim=1) 
        bg_mass  = (img_attn_norm * (1 - m).unsqueeze(0)).sum(dim=1)
        # bg_mass  = (img_attn_norm ).sum(dim=1)

        # obj_mass = (img_attn_norm * m.unsqueeze(0)).sum(dim=1) /  (m.sum())      # [H]
        # bg_mass  = (img_attn_norm * (1 - m).unsqueeze(0)).sum(dim=1) / (IMG_TOKEN_LEN - m.sum())   # [H]
        
        # obj_mass = (img_attn_norm * m.unsqueeze(0)).sum(dim=1) / (img_attn_norm.sum(dim=1) + EPS)  # [H]
        # bg_mass  = (img_attn_norm * (1 - m).unsqueeze(0)).sum(dim=1) / (img_attn_norm.sum(dim=1) + EPS)  # [H]


        # obj_mass = (img_attn * m.unsqueeze(0)).sum(dim=1)        # [H]
        # bg_mass  = (img_attn * (1 - m).unsqueeze(0)).sum(dim=1)  # [H]
        # obj_mass = obj_mass / (obj_mass.sum(dim=1, keepdim=True) + EPS)
        # bg_mass = bg_mass / (bg_mass.sum(dim=1, keepdim=True) + EPS)


        total_img = attn_last[:, img_slice].sum(dim=1) / (attn_last.sum(dim=1) + EPS)  # [H]

        # pdb.set_trace()

        obj_list.append(obj_mass.detach().cpu().numpy())
        bg_list.append(bg_mass.detach().cpu().numpy())
        # total_list.append(total_img.detach().cpu().numpy())
        total_list.append(total_img.detach().cpu().type(torch.float32).numpy())

        # #  对应Value部分：attn_last的Value（第0个batch）对应的value部分
        # value_last = out.values[step][layer][0, :, img_slice, :]  # [H, 576, D]
        # # 计算Value Attention Score
        # value_attn_obj = (img_attn_norm * m.unsqueeze(0)).sum(dim=1)  # [H]
        # value_attn_bg  = (img_attn_norm * (1 - m).unsqueeze(0)).sum(dim=1)  # [H]

        # # Value Attention Score = Attention Score * Value
        # value_obj = (img_attn_norm * value_last).sum(dim=2)  # 对应 Value 的得分变化
        # value_bg  = (img_attn_norm * value_last).sum(dim=2)  # 对应背景部分的 Value 变化

        # value_obj_list.append(value_obj.detach().cpu().numpy())
        # value_bg_list.append(value_bg.detach().cpu().numpy())

    obj_mat   = np.stack(obj_list, axis=0)    # [L, H]
    bg_mat    = np.stack(bg_list, axis=0)     # [L, H]
    total_mat = np.stack(total_list, axis=0)  # [L, H]
    return obj_mat, bg_mat, total_mat


# ---------- (B) 画“最后一个 step”的 Δ(After−Before) 热力图 ----------
def plot_last_step_heatmaps(out_before, out_after, object_mask_24x24, save_path=None):
    # last_step_b = len(out_before.attentions) - 1
    # last_step_a = len(out_after.attentions)  - 1
    # pdb.set_trace()
    # out_before.attentions[0][0].shape
    # out_after.attentions[0][0].shape
    last_step_b = 0
    last_step_a = 0
    # 统一用“各自的最后一个 step”
    obj_b, bg_b, tot_b = attn_object_bg_all_layers(out_before, object_mask_24x24, last_step_b)
    obj_a, bg_a, tot_a = attn_object_bg_all_layers(out_after,  object_mask_24x24, last_step_a)
    
    # img_slice = slice(34, 34 + 576)
    img_slice = slice(41, 41 + 576)

    attn_layers_b, attn_layers_a = [], []
    step_attentions_b = out_before.attentions[last_step_b]
    for attn_layer in step_attentions_b:
        attn_last = attn_layer[0, :, -1, img_slice].sum(dim=1)
        attn_layers_b.append(attn_last) 
    step_attentions_a = out_after.attentions[last_step_a]
    for attn_layer in step_attentions_a:
        attn_last = attn_layer[0, :, -1, img_slice].sum(dim=1)
        attn_layers_a.append(attn_last)     
    
    attn_layers_b = torch.stack(attn_layers_b, dim=0).detach().cpu().type(torch.float32).numpy()
    attn_layers_a = torch.stack(attn_layers_a, dim=0).detach().cpu().type(torch.float32).numpy()
    d_tot = (attn_layers_a - attn_layers_b) # / (abs(attn_layers_b) + 1e-8)


    # 对齐层数和头数（极少数模型可能层数不同；做截断对齐）
    L = min(obj_b.shape[0], obj_a.shape[0])
    H = min(obj_b.shape[1], obj_a.shape[1])
    d_obj = (obj_a[:L, :H] - obj_b[:L, :H])   # [L,H]
    d_bg  = (bg_a [:L, :H] - bg_b [:L, :H])   # [L,H]

    # d_obj = (obj_a[:L, :H] - obj_b[:L, :H])  / (abs(obj_b[:L, :H]) + 1e-8)  # [L,H]
    # d_bg  = (bg_a [:L, :H] - bg_b [:L, :H])  / (abs(bg_b[:L, :H]) + 1e-8)  # [L,H]

    # pdb.set_trace()

    vmax = np.max(np.abs(np.concatenate([d_obj.reshape(-1), d_bg.reshape(-1)])))
    vmin = -vmax

    # return d_obj, d_bg
    # return d_obj, d_bg, obj_b, bg_b, tot_b, obj_a, bg_a, tot_a

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    # im0 = axes[0].imshow(d_obj, aspect='auto', origin='lower', cmap='RdBu', vmin=-np.max(np.abs(d_obj)), vmax=np.max(np.abs(d_obj)))
    im0 = axes[0].imshow(d_obj, aspect='auto', origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[0].set_title("Δ Object-Attn (After vs. Before)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # im1 = axes[1].imshow(d_bg, aspect='auto', origin='lower', cmap='RdBu', vmin=-np.max(np.abs(d_bg)), vmax=np.max(np.abs(d_bg)))
    im1 = axes[1].imshow(d_bg, aspect='auto', origin='lower', cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[1].set_title("Δ Background-Attn (After vs. Before)")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return d_obj, d_bg, d_tot, obj_b, bg_b, tot_b, obj_a, bg_a, tot_a


def plot_layer_head_object_background_heatmaps(obj_mat, bg_mat, total_mat, relative=False, save_path=None):
    # pdb.set_trace()

    if relative:
        obj_mat = (obj_mat - total_mat) / (abs(total_mat) + 1e-8)
        bg_mat = (bg_mat - total_mat) / (abs(total_mat) + 1e-8)

    vmax = np.max(np.abs(np.concatenate([obj_mat.reshape(-1), bg_mat.reshape(-1)])))
    vmin = 0

    # return d_obj, d_bg

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    # im0 = axes[0].imshow(d_obj, aspect='auto', origin='lower', cmap='RdBu', vmin=-np.max(np.abs(d_obj)), vmax=np.max(np.abs(d_obj)))
    im0 = axes[0].imshow(obj_mat, aspect='auto', origin='lower', cmap='Blues', vmin=vmin, vmax=vmax)
    axes[0].set_title("Object-Attention-Score (Local)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # im1 = axes[1].imshow(d_bg, aspect='auto', origin='lower', cmap='RdBu', vmin=-np.max(np.abs(d_bg)), vmax=np.max(np.abs(d_bg)))
    im1 = axes[1].imshow(bg_mat, aspect='auto', origin='lower', cmap='Blues', vmin=vmin, vmax=vmax)
    axes[1].set_title("Background-Attention-Score (Local)")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(total_mat, aspect='auto', origin='lower', cmap='Blues')
    axes[2].set_title("Image-Attention-Scor (Global)")
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Layer")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    # return d_obj, d_bg

# ---------- (C) 画“早期/中期/晚期”的 total image-attn 变化率柱状图 ----------
@torch.no_grad()
def total_image_attn_mean_over_LH(out, step, object_mask_24x24):
    """对某个 step，计算 total image-attn 在所有 layer 与 head 上的平均值（标量）。"""
    _, _, total_mat = attn_object_bg_all_layers(out, object_mask_24x24, step)  # [L,H]
    return float(total_mat.mean())  # 标量


def plot_total_image_attn_change_per_step(out_before,
                                          out_after,
                                          object_mask_24x24,
                                          title="Total image-attn change rate per step",
                                          relative: bool = True,
                                          n_steps=None,
                                          eps: float = 1e-8,
                                          save_path=None):
    """
    逐 step 画柱状图：纵轴为变化率/差值，横轴为 step。
    - 为对齐长度，最多画到 n_steps = min(len(before), len(after)) - 1。
    - relative=True: 变化率 = (after - before) / (|before| + eps)
      relative=False: 画差值 = (after - before)
    """
    xs, ys, mb_vals, ma_vals = [], [], [], []
    
    # 与你之前的一致：计算“total image-attn（对 L,H 均值）”的标量
    def _mean_total_img_attn(out, step):
        return total_image_attn_mean_over_LH(out, step, object_mask_24x24)
    if n_steps is None:
        n_steps = min(len(out_before.attentions), len(out_after.attentions))
    if n_steps == 0:
        print("[warn] no attentions to plot.")
        return
    
    n_steps  = 10
    try:
        for s in range(0, n_steps+1):
            sb = int(s * 0.1 * len(out_before.attentions)) -1
            mb = _mean_total_img_attn(out_before, sb)
            sa = int(s * 0.1 * len(out_after.attentions)) -1
            # print(s, sb, sa)
            ma = _mean_total_img_attn(out_after,  sa)

            # mb = _mean_total_img_attn(out_before, s)
            # ma = _mean_total_img_attn(out_after,  s)
            if relative:
                y = (ma - mb) / (abs(mb) + eps)
            else:
                y = (ma - mb)
            xs.append(s)
            ys.append(y)
            mb_vals.append(mb)
            ma_vals.append(ma)
    except IndexError:
        for i in range(64):
            mb_vals.append(0)
            ma_vals.append(0)
    #     return  mb_vals, ma_vals
    
    return mb_vals, ma_vals


    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    plt.figure(figsize=(14, 6))

    ax1 = plt.gca()

    # 绘制柱状图（ma 和 mb 分别为不同颜色）
    width = 0.35  # 柱状图宽度
    ax1.bar(np.array(xs) - width/2, mb_vals, width, label="Before (mb)", color="lightcoral", edgecolor="black")
    ax1.bar(np.array(xs) + width/2, ma_vals, width, label="After (ma)", color="skyblue", edgecolor="black")

    # 变化率的折线图（圆点）
    ax2 = ax1.twinx()  # 使用右 y 轴
    ax2.plot(xs, ys, 'o-', color="#ffc685", label="Change rate (After−Before)", markersize=6,  markeredgecolor='black')

    # 设置轴和标题
    ax1.set_xlabel("Generate Step")
    ax1.set_ylabel("Total Attention (Before / After)")
    ax2.set_ylabel("Change rate (After vs. Before)")
    plt.title(title)

    # 每 4 步显示一个刻度；如有必要把最后一个 step 也补上
    tick_step = 1
    tick_positions = list(range(0, n_steps, tick_step))
    if (n_steps - 1) not in tick_positions:
        tick_positions.append(n_steps - 1)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_positions)


    # 设置图例
    ax1.legend(loc="upper left")  # 显示柱状图的图例
    ax2.legend(loc="upper right")  # 显示折线图的图例

    # 布局优化
    plt.tight_layout()

    # 保存和显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return mb_vals, ma_vals

def plot_attention_map(
    attentions,
    token_regions,
    raw_img_norm,
    token_idx=-1,
    img_patch_size=(24, 24),
    save_path=None,
    layer_indices=[0, 15, 31]
):
   
    """
    对每个层的所有注意力头取平均，绘制热力图并标记token区域
    """
    steps = len(attentions)
    # num_layers = attentions.shape[0]

    # 创建子图（数量与layer_indices一致）
    fig, axes = plt.subplots(1, len(layer_indices), figsize=(6 * len(layer_indices), 6))
    axes = axes if isinstance(axes, np.ndarray) else [axes]  # 处理单子图情况


    # 获取图像token的索引
    img_start, img_end = token_regions["img"]
    img_token_len = img_end - img_start
    
    # 原始图像大小
    H_patch, W_patch = img_patch_size
    H_img = int(np.sqrt(img_token_len)) * H_patch
    W_img = int(np.sqrt(img_token_len)) * W_patch
    raw_img_resized = cv2.resize(raw_img_norm, (H_img, W_img))

    # step_attention = outputs.attentions[0]
    # layer_attn = step_attention[1][0,0,-1,34:34+576]
    # layer_attn = step_attention[0]

    step_attention = attentions[token_idx]  # list[layer]
    for i, layer_idx in enumerate(layer_indices):
        layer_attention = step_attention[layer_idx]  # 最后一层
        # last_layer_attention = torch.stack(step_attention).mean(dim=0)  # 对所有层求平均, shape: (batch, num_heads, seq_len, seq_len)
        
        # last_layer_attention shape: (batch, heads, seq_len, seq_len)
        batch_size, num_heads, seq_len, _ = layer_attention.shape
        
        # max head
        # head_sum = layer_attention.sum(dim=1)
        # best_head = torch.argmax(head_sum).item()
        # avg_attention = layer_attention[best_head]
        head_attention_sums = layer_attention.sum(dim=-1).sum(dim=-1)  # shape: (batch, num_heads)
        max_attention_head_idx = head_attention_sums.argmax(dim=1)[0].item() # shape: (batch_size,)
        avg_attention = layer_attention[:, max_attention_head_idx, :, :]  # batch=0, 选中的head
        

        # 平均所有head
        # avg_attention = layer_attention.mean(dim=1)  # shape: (batch, seq_len, seq_len)
        
        token_attention = avg_attention[0, -1, :]  # shape: (seq_len,)
        
        # 只保留对图像token的注意力
        img_attention = token_attention[img_start:img_end]  # shape: (img_token_len,)
        img_attention_map = img_attention.reshape(int(np.sqrt(img_token_len)), int(np.sqrt(img_token_len))).cpu().numpy()
        img_attention_map = img_attention_map / (img_attention_map.max() + 1e-6)  # 归一化
        
        # 将注意力resize到原始图像大小
        img_attention_map_resized = np.kron(img_attention_map, np.ones((H_patch, W_patch)))

        # 可视化
        ax = axes[i]
        ax.imshow(raw_img_resized)
        map = ax.imshow(img_attention_map_resized, cmap='jet', alpha=0.5)
        ax.set_title(f'Layer {layer_idx} Head {max_attention_head_idx}')
        ax.axis('off')
    
        fig.colorbar(map, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def track_multi_token_prob(target_token_ids,  new_tokens, outputs,  max_new_tokens=50):

    n_target_tokens = len(target_token_ids[0])
   
    # 追踪目标token序列的连续匹配
    match_progress = 0  # 记录当前匹配到目标序列的第几个token
    target_occurrences = []  # 存储完整匹配出现的位置和概率
    current_sequence_probs = []  # 存储当前匹配中的token概率
    
    for step, (token_id, score) in enumerate(zip(new_tokens, outputs.scores)):
        # 计算当前token的概率 (log10)
        probs = torch.softmax(score, dim=-1).squeeze().cpu().detach().numpy()
        token_prob_log10 = np.log10(probs[token_id] + 1e-10)
        
        # 检查是否匹配目标序列的当前位置  target_token_ids[0][match_progress]
        if token_id == target_token_ids[0, match_progress].item():
            current_sequence_probs.append((step, token_id, token_prob_log10))
            match_progress += 1
            
            # 如果完整匹配了整个目标序列
            if match_progress == n_target_tokens:
                # 计算联合概率 (log10之和)
                joint_log10 = sum(p for _, _, p in current_sequence_probs)
                start_step = current_sequence_probs[0][0]
                end_step = current_sequence_probs[-1][0]
                
                target_occurrences.append({
                    "start_step": start_step,
                    "end_step": end_step,
                    "tokens": [t for _, t, _ in current_sequence_probs],
                    "individual_probs_log10": [p for _, _, p in current_sequence_probs],
                    # "individual_probs_log10": current_sequence_probs[0][-1],
                    "joint_prob_log10": joint_log10
                })
                # pdb.set_trace()
                
                # 重置匹配进度，允许识别多次出现
                match_progress = 0
                current_sequence_probs = []
        
        # 如果不匹配，且之前有部分匹配，则重置进度
        elif match_progress > 0:
            match_progress = 0
            current_sequence_probs = []
    
    tot_prob  = []
    # 6. 输出结果
    if target_occurrences:
        # print(f"\n共发现 {len(target_occurrences)} 次目标单词完整出现:")
        for i, occurrence in enumerate(target_occurrences, 1):
            # print(f"\n第 {i} 次出现:")
            # print(f"  位置: 步骤 {occurrence['start_step']} 到 {occurrence['end_step']}")
            # # print(f"  各token的log10概率: {[f'{p:.4f}' for p in occurrence['individual_probs_log10']]}")
            # print(f"  各token的log10概率: {[f'{p:.4f}' for p in occurrence['individual_probs_log10']]}")
            # print(f"  联合log10概率: {occurrence['joint_prob_log10']:.4f}")
            tot_prob.append(occurrence['joint_prob_log10'])
    else:
        # print(f"\n未在生成结果中发现完整的目标单词")
        # tot_prob.append(0.0)
        return 0.0
    
    # return  tot_prob
    return  torch.tensor(tot_prob).mean().item()