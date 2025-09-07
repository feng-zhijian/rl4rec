import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
# from utility import pad_history, extract_axis_1
import pdb
from tqdm import tqdm
import logging


def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

def calculate_hit(sorted_list, topk, actions, rewards, reward_click, total_reward, hit_click, ndcg_click, hit_follow, ndcg_follow, hit_forward, ndcg_forward, hit_like, ndcg_like, scenarios_reward_dict, types):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(actions)):
            if actions[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == actions[j])
                total_reward[i] += rewards[j]
                # 统计不同类型
                if "c" in types[j]:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                if "f" in types[j]:
                    hit_follow[i] += 1.0
                    ndcg_follow[i] += 1.0 / np.log2(rank + 1)
                if "s" in types[j]:
                    hit_forward[i] += 1.0
                    ndcg_forward[i] += 1.0 / np.log2(rank + 1)
                if "l" in types[j]:
                    hit_like[i] += 1.0
                    ndcg_like[i] += 1.0 / np.log2(rank + 1)
                # 场景统计
                if actions[j] in scenarios_reward_dict:
                    scenarios_reward_dict[actions[j]][i] += rewards[j]


def double_qlearning_loss(q_values, actions, rewards, discounts, target_qs, target_qs_selector):
    """
    q_values: (batch, num_actions) - main Q网络输出
    actions: (batch,) - 当前动作
    rewards: (batch,) - 奖励
    discounts: (batch,) - 折扣因子
    target_qs: (batch, num_actions) - target Q网络输出
    target_qs_selector: (batch, num_actions) - main Q网络输出（用于选动作）
    """
    
    next_actions = torch.argmax(target_qs_selector, dim=1)  # (batch,)
    next_q = target_qs.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (batch,)
    # rewards: list
    # discounts: torch.Size([256])
    # next_q: torch.Size([256])
    td_target = rewards + discounts * next_q
    q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = F.mse_loss(q_pred, td_target.detach())
    return loss
