import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
import argparse
from utility import pad_history, calculate_hit, double_qlearning_loss, insert_flag, contrastive_loss_cosine
import pdb
from tqdm import tqdm
import math
import logging
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Run STAR-based RUIE for multi-scenario modeling.")
    parser.add_argument('--epoch', type=int, default=30, help='Number of max epochs.')
    # parser.add_argument('--data', nargs='?', default='./data', help='data directory')
    parser.add_argument('--data', nargs='?', default='./data_v2', help='data directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding size.')
    parser.add_argument('--top_k', type=list, default=[20,50,100,200], help='top k.')

    # BERT4Rec specific parameters
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers.')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length.')

    parser.add_argument('--r_click', type=float, default=1.0, help='reward for the click behavior.')
    parser.add_argument('--r_follow', type=float, default=3.0, help='reward for the purchase behavior.')
    parser.add_argument('--r_like', type=float, default=3.0, help='reward for the purchase behavior.')
    parser.add_argument('--r_forward', type=float, default=2.0, help='reward for the purchase behavior.')

    parser.add_argument('--neg_len', type=int, default=10, help='contrast learning negative sample length.')    
    parser.add_argument('--seq_max_len', type=int, default=20, help='sequence length.')    

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5, help='Discount factor for RL.')
    parser.add_argument('--log_file', type=str, default='star_ruie_train_eval.log', help='Log file name.')
    return parser.parse_args()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(attention_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class BERT4Rec(nn.Module):
    def __init__(self, item_num, hidden_size, max_len, num_layers=2, num_heads=2, dropout=0.1, scenario_num=2):
        super(BERT4Rec, self).__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.scenario_num = scenario_num
        
        self.item_embedding = nn.Embedding(self.item_num + 2, hidden_size, padding_idx=self.item_num)
        self.scenario_embedding = nn.Embedding(self.scenario_num + 1, hidden_size, padding_idx=self.scenario_num)

        self.pos_encoding = PositionalEncoding(hidden_size, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, hidden_size * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)


        # 強化学习
        self.fc_q = nn.ModuleList([
            nn.Linear(hidden_size*2, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, self.scenario_num),
        ])

        # next item
        self.fc_nextitem = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ])

        # 对比学习
        self.fc_cl_dnn = nn.ModuleList([
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        ])

        self.logits_layer = nn.Linear(hidden_size, item_num)

        # Multi-Head Attention
        self.attention_heads = 4
        self.attention_d_model = hidden_size
        self.attention_depth = hidden_size // self.attention_heads
        assert hidden_size % self.attention_heads == 0, f"hidden_size {hidden_size} must be divisible by attention_heads {self.attention_heads}"
        
        self.Wq = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wk = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wv = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wo = nn.Parameter(torch.empty(hidden_size, hidden_size))
        
        nn.init.trunc_normal_(self.Wq, std=0.02, a=-2*0.02, b=2*0.02)
        nn.init.trunc_normal_(self.Wk, std=0.02, a=-2*0.02, b=2*0.02)
        nn.init.trunc_normal_(self.Wv, std=0.02, a=-2*0.02, b=2*0.02)
        nn.init.trunc_normal_(self.Wo, std=0.02, a=-2*0.02, b=2*0.02)

    def split_heads(self, x, num_heads):
        batch_size, seq_len, d_model = x.size()
        depth = d_model // num_heads
        x = x.view(batch_size, seq_len, num_heads, depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, num_heads, seq_len, depth)
        dk = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=Q.device))
        if mask is not None:
            # mask: (batch_size, 1, seq_len, seq_len) or broadcastable
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def multi_head_attention(self, q, k, v, mask=None):
        batch_size, seq_len, d_model = q.size()
        
        Q = torch.matmul(q, self.Wq)  # (batch_size, seq_len, d_model)
        K = torch.matmul(k, self.Wk)
        V = torch.matmul(v, self.Wv)

        Q = self.split_heads(Q, self.attention_heads)  # (batch_size, num_heads, seq_len, depth)
        K = self.split_heads(K, self.attention_heads)
        V = self.split_heads(V, self.attention_heads)

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, depth)
        concat_attn = attn_output.view(batch_size, seq_len, d_model)  # (batch_size, seq_len, d_model)

        output = torch.matmul(concat_attn, self.Wo)  # (batch_size, seq_len, d_model)
        return output, attn_weights

    def create_padding_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def forward(self, inputs, len_state, neg_items=None, next_true_item=None, true_scenario=None, attention_state=None, attention_scenario=None, mask_array=None, scenario_ids=None, get_emb=False):
  
        seq_len = inputs.size(1)      
        x = self.item_embedding(inputs)  # [batch_size, seq_len, hidden_size]
        scenario_embeddings = self.scenario_embedding(scenario_ids) # torch.Size([256, 64])
        
        x = x + scenario_embeddings.unsqueeze(1)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)

        transformer_mask = self.create_padding_mask(inputs, self.item_num)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, transformer_mask)
        x = self.layer_norm(x)
        last_rep = self.output_layer(x)[:, -1, :]  # [batch_size, item_num]

        logits = self.logits_layer(last_rep) 
        if get_emb:
            return logits
        
        # fc_nextitem: 輸出所有 next item 的 embedding：torch.Size([256, 64])
        nextitem_input = last_rep
        for layer in self.fc_nextitem:
            nextitem_input = layer(nextitem_input)
        pred_next_item_emb = nextitem_input

        # === Multi-Head Attention ===
        attention_embedding = self.item_embedding(attention_state)  # [batch, seq_len, hidden_size]
        attention_scenario_embedding = self.scenario_embedding(attention_scenario) # [batch, scenario_num, hidden_size]
        attention_embedding = attention_embedding + attention_scenario_embedding
        mask_array = mask_array.unsqueeze(1) # [batch, 1, seq_len+1, seq_len+1]
        attention_ret, attention_weights = self.multi_head_attention(attention_embedding, attention_embedding, attention_embedding, mask_array)
        # 取attention最后一个
        idx = (len_state - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.hidden_size)
        last_attention_ret = attention_ret.gather(1, idx).squeeze(1) # torch.Size([256, 64])

        # 强化学习
        # all_q_values：输出所有场景的 Q-value：torch.Size([256, 2])
        q_input = torch.concat([last_rep, last_attention_ret], dim=-1)
        for layer in self.fc_q:
            q_input = layer(q_input)
        all_q_values = torch.softmax(q_input, dim=-1)

        # === Gate ===
        true_scenario_probability = all_q_values.gather(1, true_scenario.unsqueeze(1)).squeeze(1) # torch.Size([256])
        gate = 1 / (1 - true_scenario_probability + 1e-6)  # torch.Size([256])

        # === 对比学习 ===
        cl_pred_next_item_emb = torch.concat([last_rep, last_attention_ret], dim=-1)
        for layer in self.fc_cl_dnn:
            cl_pred_next_item_emb = layer(cl_pred_next_item_emb)
        
        return all_q_values, logits, cl_pred_next_item_emb, gate


def evaluate(mainQN, device, data_directory, state_size, item_num, reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=None):
    mainQN.eval()
    eval_sessions = pd.read_json(os.path.join(data_directory, 'sampled_test.json'))
    eval_ids = eval_sessions.user_id.unique()
    groups = eval_sessions.groupby('user_id')
    batch = 20 
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

    # # 預先計算所有物品的嵌入，避免重複計算
    # with torch.no_grad():
    #     all_item_ids = torch.arange(mainQN.item_num, device=device)
    #     all_item_embeddings = mainQN.item_embedding(all_item_ids)
    #     all_item_embeddings_norm = F.normalize(all_item_embeddings, p=2, dim=1)
    
    # 預先計算 topk_max 以避免重複計算
    topk_max = max(topk)

    pbar = tqdm(total=len(eval_ids), desc='Evaluating', ncols=80)
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards, true_items, types = [], [], [], [], [], []
        scenario_ids = []
        scenario_history = []

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
                scenario_id = row['scenario_id']
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
                scenario_ids.append(scenario_id)
                history.append(row['item_id'])
                scenario_history.append(row['scenario_id'])
                types.append(type)
            evaluated += 1
            pbar.update(1)
            
        states_tensor = torch.tensor(states, dtype=torch.long, device=device)
        len_states_tensor = torch.tensor(len_states, dtype=torch.long, device=device)
        scenario_ids_tensor = torch.tensor(scenario_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            prediction = mainQN(
                states_tensor, len_states_tensor,
                true_scenario=scenario_ids_tensor,
                scenario_ids=scenario_ids_tensor,
                get_emb=True
            )
            _, topk_indices = torch.topk(prediction, topk_max, dim=1, largest=True, sorted=True)
            sorted_list = topk_indices.cpu().numpy()

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
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(args.log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_directory = args.data
    data_statis = pd.read_json(os.path.join(data_directory, 'data_statis.json'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    scenario_num = data_statis['scenario_num'][0]

    neg_len = args.neg_len
    seq_max_len = args.seq_max_len

    reward_click = args.r_click
    reward_follow = args.r_follow
    reward_like = args.r_like
    reward_forward = args.r_forward
    
    topk = args.top_k
    print("load statis data finished")

    bert4rec_model_1 = BERT4Rec(
        item_num=item_num,
        hidden_size=args.hidden_factor,
        max_len=args.max_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout_rate,
        scenario_num=scenario_num
    ).to(device)

    bert4rec_model_2 = BERT4Rec(
        item_num=item_num,
        hidden_size=args.hidden_factor,
        max_len=args.max_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout_rate,
        scenario_num=scenario_num
    ).to(device)
    
    print("initialize HiNet RUIE model finished")

    optimizer_1 = torch.optim.Adam(bert4rec_model_1.parameters(), lr=args.lr)
    optimizer_2 = torch.optim.Adam(bert4rec_model_2.parameters(), lr=args.lr)
    
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
                # 准备数据
                batch = replay_buffer.sample(n=args.batch_size).to_dict()
                
                # double q learning 用到的
                next_state = torch.tensor(list(batch['next_state'].values()), dtype=torch.long, device=device)
                len_next_state = torch.tensor(list(batch['len_next_states'].values()), dtype=torch.long, device=device)

                # 主网络用到的
                state = torch.tensor(list(batch['state'].values()), dtype=torch.long, device=device)
                len_state = torch.tensor(list(batch['len_state'].values()), dtype=torch.long, device=device)
                action = torch.tensor(list(batch['action'].values()), dtype=torch.long, device=device) # action 就是 label：0/1 只有两个场景
                true_scenario = torch.tensor(list(batch['true_scenario'].values()), dtype=torch.long, device=device) # 下一个场景的id 就是 label：0/1 只有两个场景

                # 当前场景ID
                last_scenario_list = []
                for sublist in list(batch['scenario'].values()):
                    found = 0
                    for num in reversed(sublist):
                        if num != 2:
                            found = num
                            break
                    last_scenario_list.append(found)
                scenario_ids = torch.tensor(last_scenario_list, dtype=torch.long, device=device)

                # 对比学习用到的, 生成負樣本
                next_true_item = list(batch['true_item'].values())
                negative_item = list()
                for idx,val in enumerate(next_true_item):
                    temp_neg_list = []
                    for i in range(neg_len):
                        neg_item = random.randint(0, item_num )  # 確保不超出有效範圍
                        while neg_item in state[idx]:
                            neg_item = random.randint(0, item_num )
                        temp_neg_list.append(neg_item)
                    negative_item.append(temp_neg_list)
                negative_item = torch.tensor(negative_item, dtype=torch.long, device=device)
                next_true_item = torch.tensor(list(batch['true_item'].values()), dtype=torch.long, device=device)
                
                # 构造 mask矩阵，True 表示有效位置  
                mask_array = list()
                for tmp_len in list(batch['len_state'].values()):
                    mask = np.zeros((seq_max_len+1, seq_max_len+1))
                    for t_l in range(tmp_len+1):
                        mask[t_l,:t_l+1] = 1
                    mask_array.append(mask.astype(bool))
                mask_array = np.array(mask_array)
                mask_array = torch.tensor(mask_array, dtype=torch.bool, device=device)

                # Attention的输入
                attention_state = [item.copy() for item in batch['state'].values()] # 256 * 20
                for idx in range(len(attention_state)):
                    attention_state[idx] = insert_flag(attention_state[idx], item_num, True)
                attention_state = torch.tensor(attention_state, dtype=torch.long, device=device)

                attention_scenario = [item.copy() for item in batch['scenario'].values()] # 256 * 20
                for idx in range(len(attention_scenario)):
                    attention_scenario[idx] = insert_flag(attention_scenario[idx], scenario_num, False)
                attention_scenario = torch.tensor(attention_scenario, dtype=torch.long, device=device)

                # 选主网络
                pointer = np.random.randint(0, 2)
                if pointer == 0:
                    mainQN, target_QN, optimizer = bert4rec_model_1, bert4rec_model_2, optimizer_1
                else:
                    mainQN, target_QN, optimizer = bert4rec_model_2, bert4rec_model_1, optimizer_2

                # Double Q-learning 的改进思路是 分开动作选择和动作评估
                # 目标网络输出的 Q(s', a)，用来 评估动作的价值
                # 主网络输出的 Q(s', a)，用来 选择动作 (argmax)
                with torch.no_grad():
                    target_Qs, _, _, _ = target_QN(next_state, len_next_state, negative_item, next_true_item, true_scenario, attention_state, attention_scenario, mask_array, scenario_ids)
                    target_Qs_selector, _, _, _ = mainQN(next_state, len_next_state, negative_item, next_true_item, true_scenario, attention_state, attention_scenario, mask_array, scenario_ids)

                # Set target_Qs to 0 for states where episode ends
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

                # 主网络开始训练
                q_values, logits, cl_pred_next_item_emb, gate = mainQN(state, len_state, negative_item, next_true_item, true_scenario, attention_state, attention_scenario, mask_array, scenario_ids)

                # ce_loss
                ce_loss = F.cross_entropy(logits, next_true_item)

                # Double Q-learning
                qloss = double_qlearning_loss(q_values, action, reward, discount, target_Qs, target_Qs_selector)

                # 对比学习
                neg_item_embs = mainQN.item_embedding(negative_item) # torch.Size([256, 10, 64])
                next_true_item_emb = mainQN.item_embedding(next_true_item) # torch.Size([256, 64])
                cl_loss = contrastive_loss_cosine(cl_pred_next_item_emb, next_true_item_emb, neg_item_embs, gate, margin=2.0)

                total_loss = qloss + cl_loss + ce_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                total_step += 1

                if total_step % 200 == 0:
                    logger.info(f"the loss in {total_step}th batch is: {total_loss.item():.6f}, qloss: {qloss.item():.6f}, cl_loss: {cl_loss.item():.6f}, ce_loss: {ce_loss.item():.6f}")
                pbar.update(1)
        logger.info(f"Evaluation after epoch {epoch+1}:")
        evaluate(mainQN, device, data_directory, state_size, item_num, reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=logger)
        mainQN.train()
