from torch import nn
from nets.graph_layers import MultiHeadAttentionLayerforCritic, ValueDecoder


class Critic(nn.Module):

    def __init__(self,
                 problem_name,
                 embedding_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 normalization,
                 ):
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.encoder_u = nn.Sequential(*(
            MultiHeadAttentionLayerforCritic(self.n_heads,
                                             self.embedding_dim,
                                             self.hidden_dim,
                                             self.normalization)
            for _ in range(1)))
        self.encoder_f = nn.Sequential(*(
            MultiHeadAttentionLayerforCritic(self.n_heads,
                                             self.embedding_dim,
                                             self.hidden_dim,
                                             self.normalization)
            for _ in range(1)))

        self.value_head = ValueDecoder(n_heads=self.n_heads,
                                       input_dim=self.embedding_dim,
                                       embed_dim=self.embedding_dim)

    def forward(self, input, cost):
        u_features = input[0].detach()
        f_features = input[1].detach()

        u_em = self.encoder_u(u_features)
        f_em = self.encoder_f(f_features)
        # baseline_value = self.value_head(u_em, cost)
        baseline_value = self.value_head(u_em, f_em, cost)

        return baseline_value.detach().squeeze(), baseline_value.squeeze()
