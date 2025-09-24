import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
from utility import pad_history, calculate_hit, double_qlearning_loss
import pdb

from tqdm import tqdm
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Run STAR-based NextItNet for multi-scenario modeling.")
    parser.add_argument('--epoch', type=int, default=50, help='Number of max epochs.')
    # parser.add_argument('--data', nargs='?', default='./data', help='data directory')
    parser.add_argument('--data', nargs='?', default='./data_v2', help='data directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding size.')
    parser.add_argument('--top_k', type=list, default=[20,50,100,200], help='top k.')
    
    # STAR specific parameters
    parser.add_argument('--scenario_embedding_size', type=int, default=32, help='Scenario embedding size.')
    parser.add_argument('--shared_hidden_size', type=int, default=128, help='Shared layer hidden size.')
    parser.add_argument('--domain_hidden_size', type=int, default=64, help='Domain-specific layer hidden size.')
    parser.add_argument('--num_shared_layers', type=int, default=2, help='Number of shared layers.')
    parser.add_argument('--num_domain_layers', type=int, default=2, help='Number of domain-specific layers.')

    parser.add_argument('--r_click', type=float, default=1.0, help='reward for the click behavior.')
    parser.add_argument('--r_follow', type=float, default=3.0, help='reward for the purchase behavior.')
    parser.add_argument('--r_like', type=float, default=3.0, help='reward for the purchase behavior.')
    parser.add_argument('--r_forward', type=float, default=2.0, help='reward for the purchase behavior.')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5, help='Discount factor for RL.')
    parser.add_argument('--log_file', type=str, default='nextitem_star_train_eval.log', help='Log file name.')
    return parser.parse_args()

class SharedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(SharedLayer, self).__init__()
        self.layers = nn.ModuleList()
        
        # 第一層
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.1))
        
        # 中間層
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.1))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DomainSpecificLayer(nn.Module):
    def __init__(self, input_size, hidden_size, scenario_embedding_size, num_layers=2):
        super(DomainSpecificLayer, self).__init__()
        self.scenario_embedding_size = scenario_embedding_size
        self.input_size = input_size
        
        self.scenario_projection = nn.Linear(scenario_embedding_size, input_size)
        
        self.domain_layers = nn.ModuleList()
        self.domain_layers.append(nn.Linear(input_size, hidden_size))
        self.domain_layers.append(nn.ReLU())
        self.domain_layers.append(nn.Dropout(0.1))
        
        for _ in range(num_layers - 1):
            self.domain_layers.append(nn.Linear(hidden_size, hidden_size))
            self.domain_layers.append(nn.ReLU())
            self.domain_layers.append(nn.Dropout(0.1))
    
    def forward(self, x, scenario_embeddings):
        scenario_projected = self.scenario_projection(scenario_embeddings)
        x = x + scenario_projected  
        
        for layer in self.domain_layers:
            x = layer(x)
        return x

class StarFusion(nn.Module):
    def __init__(self, shared_hidden_size, domain_hidden_size, output_size):
        super(StarFusion, self).__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(shared_hidden_size + domain_hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_size, output_size)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(shared_hidden_size + domain_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Softmax(dim=-1)
        )
    
    def forward(self, shared_output, domain_output):
        combined = torch.cat([shared_output, domain_output], dim=-1)
        
        attention_weights = self.attention(combined)  # [batch_size, 2]
        
        weighted_shared = shared_output * attention_weights[:, 0:1]
        weighted_domain = domain_output * attention_weights[:, 1:2]
        
        fused_output = self.fusion_layer(combined)
        
        return fused_output

class STARNextItNet(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, scenario_num, 
                 dilations=None, kernel_size=3, scenario_embedding_size=32,
                 shared_hidden_size=128, domain_hidden_size=64,
                 num_shared_layers=2, num_domain_layers=2):
        super(STARNextItNet, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.scenario_num = int(scenario_num)
        
        self.item_embedding = nn.Embedding(self.item_num + 1, hidden_size, padding_idx=self.item_num)
        self.scenario_embedding = nn.Embedding(self.scenario_num + 1, scenario_embedding_size, padding_idx=self.scenario_num)
        
        self.shared_layer = SharedLayer(hidden_size, shared_hidden_size, num_shared_layers)
        
        self.domain_layers = nn.ModuleDict({
            str(i): DomainSpecificLayer(hidden_size, domain_hidden_size, scenario_embedding_size, num_domain_layers)
            for i in range(scenario_num)
        })
        
        self.fusion = StarFusion(shared_hidden_size, domain_hidden_size, hidden_size)
        
        self.fc_ce = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.item_num),
        ])

    def forward(self, inputs, len_state, scenario_ids):
        """
        Args:
            inputs: [batch_size, seq_len] - 物品序列
            len_state: [batch_size] - 序列长度
            scenario_ids: [batch_size] - 场景ID
        """
        item_embeddings = self.item_embedding(inputs)
        scenario_embeddings = self.scenario_embedding(scenario_ids)

        batch_size, max_len, embed_dim = item_embeddings.shape
        mask = torch.arange(max_len, device=item_embeddings.device).expand(batch_size, max_len) < len_state.unsqueeze(1)
        masked_embeddings = item_embeddings * mask.unsqueeze(-1).float()  # [batch, max_len, embed_dim]

        # 求和后除以长度（避免除0）
        sum_embeddings = masked_embeddings.sum(dim=1)   # [batch, embed_dim]
        mean_embeddings = sum_embeddings / len_state.unsqueeze(1).float()  # [batch, embed_dim]        
        state_hidden = mean_embeddings

        shared_output = self.shared_layer(state_hidden)
        
        domain_outputs = []
        for i in range(len(scenario_ids)):
            scenario_id = scenario_ids[i].item()
            domain_layer = self.domain_layers[str(scenario_id)]
            domain_output = domain_layer(state_hidden[i:i+1], scenario_embeddings[i:i+1])
            domain_outputs.append(domain_output)
        
        domain_output = torch.cat(domain_outputs, dim=0)
        fused_output = self.fusion(shared_output, domain_output)
        
        x = fused_output
        for layer in self.fc_ce:
            x = layer(x)
        all_ce_logits = x

        return all_ce_logits

def evaluate(mainQN, device, data_directory, state_size, item_num,
             reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=None):
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

    topk_max = max(topk)

    pbar = tqdm(total=len(eval_ids), desc='Evaluating', ncols=80)
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards, true_items, types = [], [], [], [], [], []
        scenario_ids = []

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
                types.append(type)
            evaluated += 1
            pbar.update(1)
            
        states_tensor = torch.tensor(states, dtype=torch.long, device=device)
        len_states_tensor = torch.tensor(len_states, dtype=torch.long, device=device)
        scenario_ids_tensor = torch.tensor(scenario_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            prediction = mainQN(states_tensor, len_states_tensor, scenario_ids_tensor)
            _, topk_indices = torch.topk(prediction, topk_max, dim=1, largest=True, sorted=True)
            sorted_list = topk_indices.cpu().numpy()

        calculate_hit(sorted_list, topk, true_items, rewards, reward_click, total_reward, 
                     hit_clicks, ndcg_clicks, hit_follow, ndcg_follow, hit_forward, 
                     ndcg_forward, hit_like, ndcg_like, scenarios_reward_dict, types)
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

    reward_click = args.r_click
    reward_follow = args.r_follow
    reward_like = args.r_like
    reward_forward = args.r_forward
    
    topk = args.top_k
    print("load statis data finished")

    star_model = STARNextItNet(
        hidden_size=args.hidden_factor, 
        item_num=item_num, 
        state_size=state_size,
        scenario_num=scenario_num,
        scenario_embedding_size=args.scenario_embedding_size,
        shared_hidden_size=args.shared_hidden_size,
        domain_hidden_size=args.domain_hidden_size,
        num_shared_layers=args.num_shared_layers,
        num_domain_layers=args.num_domain_layers
    ).to(device)
    
    print("initialize STAR NextItNet model finished")

    optimizer = torch.optim.Adam(star_model.parameters(), lr=args.lr)
    
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
                next_true_item = torch.tensor(list(batch['true_item'].values()), dtype=torch.long, device=device)
                
                last_scenario_list = []
                for sublist in list(batch['scenario'].values()):
                    found = 0
                    for num in reversed(sublist):
                        if num != 2:
                            found = num
                            break
                    last_scenario_list.append(found)
                scenario_ids = torch.tensor(last_scenario_list, dtype=torch.long, device=device)
                
                logits = star_model(state, len_state, scenario_ids)
                celoss = F.cross_entropy(logits, next_true_item)
                
                loss = celoss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_step += 1

                if total_step % 200 == 0:
                    logger.info(f"the loss in {total_step}th batch is: {loss.item():.6f}")
                pbar.update(1)
        logger.info(f"Evaluation after epoch {epoch+1}:")
        evaluate(star_model, device, data_directory, state_size, item_num, reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=logger)
        star_model.train()
