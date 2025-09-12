import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import argparse
from utility import pad_history, calculate_hit, double_qlearning_loss
import pdb
from NextItNetModules_pytorch import NextItNetResidualBlock
from tqdm import tqdm
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised NextItNet (PyTorch version).")
    parser.add_argument('--epoch', type=int, default=30, help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='./data', help='data directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding size.')

    parser.add_argument('--r_click', type=float, default=1.0,help='reward for the click behavior.')
    parser.add_argument('--r_follow', type=float, default=3.0,help='reward for the purchase behavior.')
    parser.add_argument('--r_like', type=float, default=3.0,help='reward for the purchase behavior.')
    parser.add_argument('--r_forward', type=float, default=2.0,help='reward for the purchase behavior.')

    parser.add_argument('--neg_len', type=int, default=10,help='contrast learning negative sample length.')    

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5, help='Discount factor for RL.')
    return parser.parse_args()

class NextItNet(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dilations=None, kernel_size=3, scenario_num=2):
        super(NextItNet, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.scenario_num = scenario_num
        self.embedding = nn.Embedding(self.item_num + 1, hidden_size, padding_idx=self.item_num)
        self.dilations = dilations if dilations is not None else [1, 2, 1, 2, 1, 2]
        self.blocks = nn.ModuleList([
            NextItNetResidualBlock(hidden_size, kernel_size, dilation, causal=True)
            for dilation in self.dilations
        ])

        self.fc_q = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, self.scenario_num),
        ])

        self.fc_nextitem = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ])

        # self.fc_q = nn.Linear(hidden_size, self.item_num)

    def forward(self, inputs, len_state, neg_items=None, next_true_item=None, true_scenario=None):
        # 行为序列：inputs: torch.Size([256, 20])
        # 状态序列长度：len_state: torch.Size([256])
        # 负样本: torch.Size([256, 10])
        # len_state: torch.Size([256])
        mask = (inputs != self.item_num).float().unsqueeze(-1)
        x = self.embedding(inputs) * mask
        for block in self.blocks:
            x = block(x) * mask
        idx = (len_state - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_size)
        state_hidden = x.gather(1, idx).squeeze(1)

        # all_q_values：输出所有 场景 的 Q-value： torch.Size([256, 2])
        x = state_hidden
        for layer in self.fc_q:
            x = layer(x)
        all_q_values = torch.softmax(x, dim=-1)

        # fc_nextitem: 输出所有 next item 的 embedding ： torch.Size([256, 64])
        x = state_hidden
        for layer in self.fc_nextitem:
            x = layer(x)
        pred_next_item_emb = x

        if self.training:
            true_scenario_probability = all_q_values.gather(1, true_scenario.unsqueeze(1)).squeeze(1) # torch.Size([256])
            gate = 1 / (1 - true_scenario_probability + 1e-8)  # torch.Size([256])

            # === 对比学习 Triplet Loss ===
            neg_item_embs = self.embedding(neg_items) # torch.Size([256, 10, 64])
            next_true_item_emb = self.embedding(next_true_item) # torch.Size([256, 64])

            pos_euclid = torch.norm(pred_next_item_emb - next_true_item_emb, dim=1, p=2)  # torch.Size([256])
            expand_output2 = state_hidden.unsqueeze(1)  # torch.Size([256, 1, 64])
            neg_euclid = torch.norm(expand_output2 - neg_item_embs, dim=2, p=2)    # torch.Size([256, 10])
            pos_euclid_exp = pos_euclid.unsqueeze(-1)  # torch.Size([256, 1])
            cl_loss = pos_euclid_exp - neg_euclid + 70
            cl_loss = torch.where(cl_loss < 0, torch.zeros_like(cl_loss), cl_loss)  # torch.Size([256, 10])
            cl_loss = torch.mean(torch.mean(cl_loss, dim=-1) * gate)  # torch.Size([256]) => value

            return all_q_values, cl_loss, pred_next_item_emb
        else:
            return all_q_values, pred_next_item_emb

def evaluate(mainQN, device, data_directory, state_size, item_num, reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=None):
    mainQN.eval()
    eval_sessions = pd.read_json(os.path.join(data_directory, 'sampled_test.json'))
    eval_ids = eval_sessions.user_id.unique()
    groups = eval_sessions.groupby('user_id')
    batch = 10
    evaluated = 0
    total_clicks = 0.0
    total_follow = 0.0
    total_forward = 0.0
    total_like = 0.0

    total_reward = [0, 0, 0, 0]

    scenarios_reward_dict = dict()
    for idx in range(scenario_num):
        scenarios_reward_dict[idx] = [0,0,0,0]

    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_follow=[0,0,0,0]
    ndcg_follow=[0,0,0,0]
    hit_forward=[0,0,0,0]
    ndcg_forward=[0,0,0,0]
    hit_like=[0,0,0,0]
    ndcg_like=[0,0,0,0]

    pbar = tqdm(total=len(eval_ids), desc='Evaluating', ncols=80)
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards, true_items, types = [], [], [], [], [], []

        for i in range(batch):
            if evaluated == len(eval_ids):
                break
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            for index, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                state = pad_history(state, state_size, item_num)
                states.append(state)

                action=row['scenario_id']
                true_item = row['item_id']
                is_click=row['click']
                is_forward=row['forward']
                is_follow=row['follow']
                is_like=row['like']

                reward = 0
                type = ""
                if is_click == 1:
                    reward += reward_click
                    type += "c"
                if is_forward == 1:
                    reward += reward_forward
                    type += "s"
                if is_follow == 1:
                    reward += reward_follow
                    type += "f"
                if is_like == 1:
                    reward += reward_like
                    type += "l"
                if is_click != 1 and is_forward != 1 and is_follow != 1 and is_like != 1:
                    type += "N"

                if is_click==1:
                    total_clicks += 1.0
                if is_forward == 1:
                    total_forward += 1.0
                if is_follow == 1:
                    total_follow += 1.0
                if is_like == 1:
                    total_like += 1.0

                true_items.append(true_item)
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
                types.append(type)

            evaluated += 1
            pbar.update(1)
        states_tensor = torch.tensor(states, dtype=torch.long, device=device)
        len_states_tensor = torch.tensor(len_states, dtype=torch.long, device=device)

        with torch.no_grad():
            _, pred_next_item_emb = mainQN(states_tensor, len_states_tensor)
            pred_next_item_emb = pred_next_item_emb.cpu().numpy() 

        topk_max = max(topk)
        # 取每个样本的 topk_max 个最大值的索引（未排序）
        topk_indices = np.argpartition(pred_next_item_emb, -topk_max, axis=1)[:, -topk_max:]
        # 再对 topk 内部排序（从大到小）
        topk_scores = np.take_along_axis(pred_next_item_emb, topk_indices, axis=1)
        sorted_idx = np.argsort(-topk_scores, axis=1)
        sorted_list = np.take_along_axis(topk_indices, sorted_idx, axis=1)

        calculate_hit(sorted_list, topk, true_items, rewards, reward_click, total_reward, hit_clicks, ndcg_clicks, hit_follow, ndcg_follow, hit_forward, ndcg_forward, hit_like, ndcg_like, scenarios_reward_dict, types)
    pbar.close()
        
    msg_lines = []
    msg_lines.append('#############################################################')
    msg_lines.append('total clicks: %d, total follow: %d, total forward: %d, total like: %d' % (total_clicks, total_follow, total_forward, total_like))
    for i in range(len(topk)):
        hr_click = float(hit_clicks[i]) / float(total_clicks) if total_clicks > 0 else 0.0
        hr_follow = float(hit_follow[i]) / float(total_follow) if total_follow > 0 else 0.0
        hr_forward = float(hit_forward[i]) / float(total_forward) if total_forward > 0 else 0.0
        hr_like = float(hit_like[i]) / float(total_like) if total_like > 0 else 0.0

        ng_click = float(ndcg_clicks[i]) / float(total_clicks) if total_clicks > 0 else 0.0
        ng_follow = float(ndcg_follow[i]) / float(total_follow) if total_follow > 0 else 0.0
        ng_forward = float(ndcg_forward[i]) / float(total_forward) if total_forward > 0 else 0.0
        ng_like = float(ndcg_like[i]) / float(total_like) if total_like > 0 else 0.0

        k = int(topk[i]) if hasattr(topk[i], '__int__') else topk[i]
        msg_lines.append('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        msg_lines.append('cumulative reward @ %d: %f' % (k, float(total_reward[i])))
        msg_lines.append('clicks   hr ndcg @ %d : %f, %f' % (k, hr_click, ng_click))
        msg_lines.append('follow   hr ndcg @ %d : %f, %f' % (k, hr_follow, ng_follow))
        msg_lines.append('forward  hr ndcg @ %d : %f, %f' % (k, hr_forward, ng_forward))
        msg_lines.append('like     hr ndcg @ %d : %f, %f' % (k, hr_like, ng_like))
    msg_lines.append('#############################################################')
    msg = '\n'.join(msg_lines)
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler('train_eval.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_directory = args.data
    data_statis = pd.read_json(os.path.join(data_directory, 'data_statis.json'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    scenario_num = data_statis['scenario_num'][0]

    neg_len = args.neg_len

    reward_click = args.r_click
    reward_follow = args.r_follow
    reward_like = args.r_like
    reward_forward = args.r_forward
    
    topk = [5, 10, 15, 20]
    print("load statis data finished")

    NextRec1 = NextItNet(hidden_size=args.hidden_factor, item_num=item_num, state_size=state_size, scenario_num=scenario_num).to(device)
    NextRec2 = NextItNet(hidden_size=args.hidden_factor, item_num=item_num, state_size=state_size, scenario_num=scenario_num).to(device)
    print("initialize model finished")

    optimizer1 = torch.optim.Adam(NextRec1.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(NextRec2.parameters(), lr=args.lr)
    replay_buffer = pd.read_json(os.path.join(data_directory, 'replay_buffer.json'))
    print("load replay buffer finished")
    total_step = 0
    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)
    print("start training")

    for epoch in range(args.epoch):
        logger.info(f"Epoch {epoch+1}/{args.epoch}")
        with tqdm(total=num_batches, desc='Training', ncols=80) as pbar:
            for j in range(num_batches):
                batch = replay_buffer.sample(n=args.batch_size).to_dict()
                next_state = torch.tensor(list(batch['next_state'].values()), dtype=torch.long, device=device)
                len_next_state = torch.tensor(list(batch['len_next_states'].values()), dtype=torch.long, device=device)


                state = torch.tensor(list(batch['state'].values()), dtype=torch.long, device=device)
                len_state = torch.tensor(list(batch['len_state'].values()), dtype=torch.long, device=device)
                action = torch.tensor(list(batch['action'].values()), dtype=torch.long, device=device) # action 就是 label：0/1 只有两个场景
                true_scenario = torch.tensor(list(batch['true_scenario'].values()), dtype=torch.long, device=device) # 下一个场景的id 就是 label：0/1 只有两个场景

                next_true_item = list(batch['true_item'].values())
                negative_item = list()

                for idx,val in enumerate(next_true_item):
                    temp_neg_list = []
                    for i in range(neg_len):
                        neg_item = random.randint(0, item_num)
                        while neg_item in state[idx]:
                            neg_item = random.randint(0, item_num)
                        temp_neg_list.append(neg_item)
                    negative_item.append(temp_neg_list)
                negative_item = torch.tensor(negative_item, dtype=torch.long, device=device)
                next_true_item = torch.tensor(list(batch['true_item'].values()), dtype=torch.long, device=device)

                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN, target_QN, optimizer = NextRec1, NextRec2, optimizer1
                else:
                    mainQN, target_QN, optimizer = NextRec2, NextRec1, optimizer2
                # Double Q-learning 的改进思路是 分开动作选择和动作评估
                # 目标网络输出的 Q(s’, a)，用来 评估动作的价值
                # 主网络输出的 Q(s’, a)，用来 选择动作 (argmax)
                with torch.no_grad():
                    target_Qs, _, _ = target_QN(next_state, len_next_state, negative_item, next_true_item, true_scenario)
                    target_Qs_selector, _, _ = mainQN(next_state, len_next_state, negative_item, next_true_item, true_scenario)

                is_done = list(batch['is_done'].values())
                for index in range(target_Qs.shape[0]):
                    if is_done[index]:
                        target_Qs[index] = torch.zeros(scenario_num, device=target_Qs.device, dtype=target_Qs.dtype)


                is_click = list(batch['is_click'].values())
                is_like = list(batch['is_like'].values())
                is_forward = list(batch['is_forward'].values())
                is_follow = list(batch['is_follow'].values())
                reward = []
                for k in range(len(is_click)):
                    tmp_reward=0
                    if is_click[k] == 1:
                        tmp_reward += reward_click
                    if is_forward[k] == 1:
                        tmp_reward += reward_forward
                    if is_follow[k] == 1:
                        tmp_reward += reward_follow
                    if is_like[k] == 1:
                        tmp_reward += reward_like
                    reward.append(tmp_reward)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                discount = torch.tensor([args.discount] * len(action), dtype=torch.float, device=device)

                q_values, cl_loss, pred_next_item_emb = mainQN(state, len_state, negative_item, next_true_item, true_scenario)
                # celoss = F.cross_entropy(logits, action)
                
                qloss = double_qlearning_loss(q_values, action, reward, discount, target_Qs, target_Qs_selector)

                loss = cl_loss + qloss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_step += 1

                if total_step % 200 == 0:
                    logger.info(f"the loss in {total_step}th batch is: {loss.item():.6f}")
                pbar.update(1)
        logger.info(f"Evaluation after epoch {epoch+1}:")
        evaluate(mainQN, device, data_directory, state_size, item_num, reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=logger)
        mainQN.train()
