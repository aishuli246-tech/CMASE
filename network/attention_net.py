import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class AttentionNet(nn.Module):
    def __init__(self, args):
        super(AttentionNet, self).__init__()
        self.args = args

        self.W_q = nn.Linear(args.embedding_hidden_dim, args.embedding_hidden_dim)
        self.W_k = nn.Linear(args.state_shape, args.embedding_hidden_dim)
        self.W_v = nn.Linear(1, args.embedding_hidden_dim)
        self.W_o = nn.Linear(args.embedding_hidden_dim, 1)
        self.dropout = nn.Dropout(args.dropout)



    def forward(self, q, k, v):
        #q为(episode_num*max_episode_len, n_agents, embedding_hidden_dim)
        #k为(episode_num * max_episode_len, state_shape)
        #v为(episode_num * max_episode_len, n_agents)

        q = self.W_q(q)    #[B*T, n, e]
        k = self.W_k(k).unsqueeze(1)   #[B*T, state_shape] -> [B*T, e] -> [B*T, 1, e]
        k = k.expand(-1, self.args.n_agents, -1)     #[B*T, 1, e]->[B*T, n, e]
        v = v.unsqueeze(-1)      #[B*T, n] -> [B*T, n, 1]
        v = self.W_v(v)          #[B*T, n, 1] -> [B*T, n, e]

        head_dim = self.args.embedding_hidden_dim // self.args.num_heads
        q = q.view(-1, self.args.n_agents, self.args.num_heads, head_dim).transpose(1,2)  #[B*T, n, e] -> [B*T, n, head_n, head_d]->[B*T, head_n, n, head_d]
        k = k.view(-1, self.args.n_agents, self.args.num_heads, head_dim).transpose(1,2)  #[B*T, n, e] -> [B*T, n, head_n, head_d]->[B*T, head_n, n, head_d]
        v = v.view(-1, self.args.n_agents, self.args.num_heads, head_dim).transpose(1,2)  #[B*T, n, e] -> [B*T, n, head_n, head_d]->[B*T, head_n, n, head_d]

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale     #[B*T, head_n, n, head_d] * [B*T, head_n, head_d, n] -> [B*T, head_n, n, n]
        weights = f.softmax(scores, dim=-1)      #[B*T, head_n, n, n]
        weights = self.dropout(weights)
        attended = torch.matmul(weights, v)      #[B*T, head_n, n, n] * [[B*T, head_n, n, head_d] -> [B*T, head_n, n, head_d]
        attended = attended.transpose(1, 2).contiguous()    #[B*T, head_n, n, head_d] -> [B*T, n, head_n, head_d]
        attended = attended.reshape(-1, self.args.n_agents, self.args.embedding_hidden_dim)   #[B*T, n, head_n, head_d] -> [B*T, n, e]

        nse_weighted = self.W_o(attended)     #[B*T, n, e] -> [B*T, n, 1]
        nse_weighted = nse_weighted.squeeze(-1)   #[B*T, n, 1] -> [B*T, n]
        return nse_weighted