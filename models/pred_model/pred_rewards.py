import torch
import torch.nn as nn


class RewardPred(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, out_dim, seq_len):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.backbone = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.len = nn.Linear(self.seq_len, 1)
        self.out1 = nn.Linear(self.hidden_dim, 32)
        self.out2 = nn.Linear(32, self.out_dim)



    def forward(self, states, actions):
        seq_len = states.shape[1]
        actions_enc = actions.unsqueeze(1).repeat(1, seq_len, 1)
        feat = torch.cat((states, actions_enc), dim=2)
        feat = self.backbone(feat)
        feat = feat.view(-1, self.hidden_dim,self.seq_len)
        feat = self.len(feat)
        feat = feat.squeeze(dim=2)
        feat = self.out1(feat)
        reward = torch.tanh(self.out2(feat))
        return reward


