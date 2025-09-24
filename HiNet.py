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
    parser = argparse.ArgumentParser(description="Run HiNet-based NextItNet for multi-scenario multi-task modeling.")
    parser.add_argument('--epoch', type=int, default=50, help='Number of max epochs.')
    # parser.add_argument('--data', nargs='?', default='./data', help='data directory')
    parser.add_argument('--data', nargs='?', default='./data_v2', help='data directory')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding size.')
    parser.add_argument('--top_k', type=list, default=[20,50,100,200], help='top k.')
    
    # HiNet specific parameters
    parser.add_argument('--scenario_embedding_size', type=int, default=64, help='Scenario embedding size.')
    parser.add_argument('--num_scenario_experts', type=int, default=2, help='Number of scenario experts.')
    parser.add_argument('--num_task_experts', type=int, default=2, help='Number of task experts.')
    parser.add_argument('--expert_hidden_size', type=int, default=128, help='Expert network hidden size.')
    parser.add_argument('--attention_hidden_size', type=int, default=64, help='Attention network hidden size.')

    parser.add_argument('--r_click', type=float, default=1.0, help='reward for the click behavior.')
    parser.add_argument('--r_follow', type=float, default=3.0, help='reward for the purchase behavior.')
    parser.add_argument('--r_like', type=float, default=3.0, help='reward for the purchase behavior.')
    parser.add_argument('--r_forward', type=float, default=2.0, help='reward for the purchase behavior.')

    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5, help='Discount factor for RL.')
    parser.add_argument('--log_file', type=str, default='nextitem_hinet_train_eval.log', help='Log file name.')
    return parser.parse_args()

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(ExpertNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)

class ScenarioExtractionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, scenario_embedding_size, num_experts, attention_hidden_size):
        super(ScenarioExtractionLayer, self).__init__()
        self.input_dim = input_dim
        self.scenario_embedding_size = scenario_embedding_size
        self.num_experts = num_experts
        
        self.scenario_embedding = nn.Embedding(10, scenario_embedding_size)
        
        self.scenario_shared_experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        
        self.scenario_specific_experts = nn.ModuleList([
            ExpertNetwork(input_dim + scenario_embedding_size, hidden_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        
        self.scenario_attention = nn.Sequential(
            nn.Linear(input_dim + scenario_embedding_size, attention_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        
        self.scenario_gate = nn.Sequential(
            nn.Linear(input_dim + scenario_embedding_size, attention_hidden_size),
            nn.ReLU(),
            nn.Linear(attention_hidden_size, 2), 
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, scenario_ids):

        batch_size = x.size(0)
        
        scenario_embeddings = self.scenario_embedding(scenario_ids)  # [batch_size, scenario_embedding_size]
        
        scenario_aware_input = torch.cat([x, scenario_embeddings], dim=-1)  # [batch_size, input_dim + scenario_embedding_size]
        
        shared_outputs = []
        for expert in self.scenario_shared_experts:
            shared_output = expert(x)  # [batch_size, hidden_dim]
            shared_outputs.append(shared_output)
        shared_outputs = torch.stack(shared_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
        
        specific_outputs = []
        for expert in self.scenario_specific_experts:
            specific_output = expert(scenario_aware_input)  # [batch_size, hidden_dim]
            specific_outputs.append(specific_output)
        specific_outputs = torch.stack(specific_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
        
        attention_weights = self.scenario_attention(scenario_aware_input)  # [batch_size, num_experts]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
        
        weighted_shared = torch.sum(attention_weights * shared_outputs, dim=1)  # [batch_size, hidden_dim]
        weighted_specific = torch.sum(attention_weights * specific_outputs, dim=1)  # [batch_size, hidden_dim]
        
        gate_weights = self.scenario_gate(scenario_aware_input)  # [batch_size, 2]
        gate_shared = gate_weights[:, 0:1]  # [batch_size, 1]
        gate_specific = gate_weights[:, 1:2]  # [batch_size, 1]
        
        scenario_output = gate_shared * weighted_shared + gate_specific * weighted_specific
        
        return scenario_output

class TaskExtractionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_tasks):
        super(TaskExtractionLayer, self).__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        
        self.task_shared_experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        
        self.task_specific_experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, hidden_dim)
            for _ in range(num_experts)
        ])
        
        self.task_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
        
        self.task_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2), 
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, x):

        batch_size = x.size(0)
        task_outputs = []
        
        for task_id in range(self.num_tasks):
            shared_outputs = []
            for expert in self.task_shared_experts:
                shared_output = expert(x)  # [batch_size, hidden_dim]
                shared_outputs.append(shared_output)
            shared_outputs = torch.stack(shared_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
            
            specific_outputs = []
            for expert in self.task_specific_experts:
                specific_output = expert(x)  # [batch_size, hidden_dim]
                specific_outputs.append(specific_output)
            specific_outputs = torch.stack(specific_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
            
            attention_weights = self.task_attention[task_id](x)  # [batch_size, num_experts]
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, num_experts, 1]
            
            weighted_shared = torch.sum(attention_weights * shared_outputs, dim=1)  # [batch_size, hidden_dim]
            weighted_specific = torch.sum(attention_weights * specific_outputs, dim=1)  # [batch_size, hidden_dim]
            
            gate_weights = self.task_gates[task_id](x)  # [batch_size, 2]
            gate_shared = gate_weights[:, 0:1]  # [batch_size, 1]
            gate_specific = gate_weights[:, 1:2]  # [batch_size, 1]
            
            task_output = gate_shared * weighted_shared + gate_specific * weighted_specific
            task_outputs.append(task_output)
        
        return task_outputs

class HiNetNextItNet(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, scenario_num, 
                 dilations=None, kernel_size=3, scenario_embedding_size=32,
                 num_scenario_experts=1, num_task_experts=1, expert_hidden_size=128,
                 attention_hidden_size=64, num_tasks=1):
        super(HiNetNextItNet, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.scenario_num = int(scenario_num)
        self.num_tasks = num_tasks
        
        self.item_embedding = nn.Embedding(self.item_num + 1, hidden_size, padding_idx=self.item_num)
        
        
        self.scenario_extraction = ScenarioExtractionLayer(
            hidden_size, expert_hidden_size, scenario_embedding_size, 
            num_scenario_experts, attention_hidden_size
        )
        
        self.task_extraction = TaskExtractionLayer(
            expert_hidden_size, expert_hidden_size, num_task_experts, num_tasks
        )
        
        self.task_output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_size, expert_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden_size, self.item_num)
            ) for _ in range(num_tasks)
        ])

    def forward(self, inputs, len_state, scenario_ids):
        
        item_embeddings = self.item_embedding(inputs)
        mask = (inputs != self.item_num).float().unsqueeze(-1)
        item_embeddings = item_embeddings * mask
        
        batch_size, max_len, embed_dim = item_embeddings.shape
        mask = torch.arange(max_len, device=item_embeddings.device).expand(batch_size, max_len) < len_state.unsqueeze(1)
        
        masked_embeddings = item_embeddings * mask.unsqueeze(-1).float()  # [batch, max_len, embed_dim]

        sum_embeddings = masked_embeddings.sum(dim=1)   # [batch, embed_dim]
        mean_embeddings = sum_embeddings / len_state.unsqueeze(1).float()  # [batch, embed_dim]
        
        state_hidden = mean_embeddings
        
        # HiNet
        scenario_features = self.scenario_extraction(state_hidden, scenario_ids)
        
        task_features = self.task_extraction(scenario_features)
        
        task_predictions = []
        for task_id in range(self.num_tasks):
            task_pred = self.task_output_layers[task_id](task_features[task_id])
            task_predictions.append(task_pred)
        
        main_prediction = task_predictions[0]  #torch.Size([256, 279081])
        return main_prediction

def evaluate(mainQN, device, data_directory, state_size, item_num,
             reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=None):
    mainQN.eval()
    eval_sessions = pd.read_json(os.path.join(data_directory, 'sampled_test.json'))
    eval_ids = eval_sessions.user_id.unique()
    groups = eval_sessions.groupby('user_id')
    batch = 20  # 適度提高批量大小，從10提高到20
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

    # 預先計算 topk_max 以避免重複計算
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
            
            # 使用 PyTorch 的 topk 函數直接在 GPU 上進行排序，更高效
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

    hinet_model = HiNetNextItNet(
        hidden_size=args.hidden_factor, 
        item_num=item_num, 
        state_size=state_size,
        scenario_num=scenario_num,
        scenario_embedding_size=args.scenario_embedding_size,
        num_scenario_experts=args.num_scenario_experts,
        num_task_experts=args.num_task_experts,
        expert_hidden_size=args.expert_hidden_size,
        attention_hidden_size=args.attention_hidden_size
    ).to(device)
    
    print("initialize HiNet NextItNet model finished")

    optimizer = torch.optim.Adam(hinet_model.parameters(), lr=args.lr)
    
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
                
                logits = hinet_model(state, len_state, scenario_ids)
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
        evaluate(hinet_model, device, data_directory, state_size, item_num,
                reward_click, reward_follow, reward_like, reward_forward, topk, scenario_num, logger=logger)
        hinet_model.train()
