import torch
import torch.nn as nn
import torch.nn.functional as f

class EmbeddingNet(nn.Module):
    def __init__(self, args):
        super(EmbeddingNet, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.obs_shape + args.n_actions, args.embedding_hidden_dim)
        self.fc2 = nn.Linear(args.embedding_hidden_dim, args.embedding_hidden_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.norm1 = nn.LayerNorm(args.embedding_hidden_dim)
        self.norm2 = nn.LayerNorm(args.embedding_hidden_dim)

    def forward(self, obs, u):
        obs = obs.reshape(-1, self.args.n_agents, self.args.obs_shape)
        u = u.reshape(-1, self.args.n_agents, self.args.n_actions)
        e = torch.cat([obs, u], dim = -1)
        e = e.reshape(-1, self.args.obs_shape + self.args.n_actions)
        ve = self.fc1(e)
        ve = self.norm1(ve)
        ve = f.relu(ve)
        ve = self.dropout(ve)
        ve = self.fc2(ve)
        ve = self.norm2(ve)
        ve = f.relu(ve)
        ve = self.dropout(ve)
        ve = ve.reshape(-1, self.args.n_agents, self.args.embedding_hidden_dim)
        return ve

