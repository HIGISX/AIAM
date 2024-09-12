from torch import nn
import torch
# from nets.graph_layers_old import MultiHeadEncoder, MultiHeadDecoder, EmbeddingNet, MultiHeadPosCompat

from nets.graph_layers import MultiHeadEncoder, MultiHeadDecoder, EmbeddingNet, MultiHeadDecoder_pair


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def get_action_sig(action_record):
    action_record_tensor = torch.stack(action_record)
    return torch.cat((action_record_tensor[-3:].transpose(0, 1), action_record_tensor.mean(0).unsqueeze(1)), 1)


class Actor(nn.Module):

    def __init__(self,
                 problem_name,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 v_range,
                 seq_length,
                 ):
        super(Actor, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.seq_length = seq_length
        self.clac_stacks = problem_name == 'pdtspl'
        self.node_dim = 2

        # networks
        self.embedder = EmbeddingNet(
            self.node_dim,
            self.embedding_dim)

        self.encoder = mySequential(*(
            MultiHeadEncoder(self.n_heads_actor,
                             self.embedding_dim,
                             self.hidden_dim,
                             self.normalization,
                             )
            for _ in range(self.n_layers)))
        self.decoder = MultiHeadDecoder(input_dim=self.embedding_dim,
                                        embed_dim=self.embedding_dim,
                                        v_range=self.range)  # the two propsoed decoders
        # self.decoder = MultiHeadDecoder_pair(input_dim=self.embedding_dim,
        #                                      embed_dim=self.embedding_dim,
        #                                      v_range=self.range)  # the two propsoed decoders

        print(self.get_parameter_number())

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, problem, users, facilities, solution, exchange, action_record, do_sample=False,
                fixed_action=None, require_entropy=False, to_critic=False, only_critic=False):

        # the users embedding and facilities embedding
        users_embed, facilities_embed = self.embedder(users, facilities)

        if only_critic:
            return users_embed, facilities_embed
            # return users_em, facilities_em
        # pass through encoder, including N (MHA+FF) layers
        users_em, facilities_em = self.encoder(users_embed, facilities_embed)

        # pass through decoder
        action, log_ll, entropy = self.decoder(problem,
                                               users_em,
                                               facilities_em,
                                               solution,
                                               exchange,
                                               get_action_sig(action_record).to(users.device),
                                               fixed_action,
                                               require_entropy=require_entropy)

        if require_entropy:
            return action, log_ll.squeeze(), (users_embed, facilities_embed if to_critic else None), entropy
        else:
            return action, log_ll.squeeze(), (users_embed, facilities_embed if to_critic else None)
