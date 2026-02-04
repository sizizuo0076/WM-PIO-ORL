from .attention_blocks import PositionalEncoding1D, AttentionBlock, get_subsequent_mask_with_batch_length
from .eps_utils import *


class TemporalTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, device, cond_dim=8, embed_dim=512, max_length=1024, num_heads=8,
                 num_layers=2, dropout=0.1):
        super(TemporalTransformer, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.cond_dim = cond_dim
        horizon = self.cond_dim
        self.embed_dim = embed_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 2 * embed_dim),
            nn.Mish(),
            nn.Linear(2 * embed_dim, embed_dim // 2)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 2 * embed_dim),
            nn.Mish(),
            nn.Linear(2 * embed_dim, embed_dim // 2)
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.final_conv = nn.Conv1d(embed_dim, embed_dim // 4, 1)

        if horizon % 2 != 0:
            out_horizon = horizon + 1
        else:
            out_horizon = horizon
        self.mid_layer = nn.Sequential(
            nn.Linear(in_features=out_horizon * embed_dim // 4 + (embed_dim * 3) // 2 + embed_dim, out_features=512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.Mish(),
            nn.Linear(512, 512),
            nn.Mish())

        self.final_layer = torch.nn.Linear(512, self.state_dim)

        self.position_encoding = PositionalEncoding1D(max_length=max_length, embed_dim=embed_dim)
        self.layer_stack = nn.ModuleList([
            AttentionBlock(feat_dim=embed_dim, hidden_dim=embed_dim * 2, num_heads=num_heads, dropout=dropout) for _ in
            range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)


    def forward(self, x, time, action, state_condition, mask):
        '''
            x : [ batch x horizon x transition ]
        '''
        batch_size = x.shape[0]
        horizon = state_condition.shape[1]

        encoded_noised_state = self.state_encoder(x)
        encoded_action = self.action_encoder(action)
        encoded_state_conditions = self.state_encoder(state_condition)

        noised_state_rpt = torch.repeat_interleave(encoded_noised_state.reshape(batch_size, 1, -1), repeats=horizon,
                                                   dim=1)

        x = torch.cat([noised_state_rpt, encoded_state_conditions], dim=2)

        t = self.time_mlp(time)

        x = self.position_encoding(x)
        x = self.layer_norm(x)

        for enc_layer in self.layer_stack:
            x, attn = enc_layer(x, mask)

        x = einops.rearrange(x, 'b h t -> b t h')

        x = self.final_conv(x)  # [32,128,30]

        x = einops.rearrange(x, 'b t h -> b h t')

        info = x.reshape(batch_size, -1)  # [32,3840]
        output = self.mid_layer(torch.cat([info, encoded_noised_state, encoded_action,
                                           encoded_state_conditions[:, -1], t], dim=1))
        output = self.final_layer(output)
        return output



