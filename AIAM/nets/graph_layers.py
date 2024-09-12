import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch import nn
import math

TYPE_REMOVAL = 'N2S'
# TYPE_REMOVAL = 'random'
# TYPE_REMOVAL = 'greedy'

TYPE_INSERTION = 'N2S'


# TYPE_INSERTION = 'random'
# TYPE_INSERTION = 'greedy'

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadInteractiveAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadInteractiveAttention, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query_u = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key_u = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val_u = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_query_f = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key_f = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val_f = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out_u = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            self.W_out_f = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, u, f):

        # h should be (batch_size, graph_size, input_dim)
        batch_size, users_size, input_dim = u.size()
        _, fac_size, _ = f.size()

        uflat = u.contiguous().view(-1, input_dim)  #################   reshape
        fflat = f.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp_u = (self.n_heads, batch_size, users_size, -1)
        shp_f = (self.n_heads, batch_size, fac_size, -1)

        # Calculate queries, (n_heads, bs, u_s/f_s, key/val_size)
        Q_u = torch.matmul(uflat, self.W_query_u).view(shp_u)
        K_u = torch.matmul(uflat, self.W_key_u).view(shp_u)
        V_u = torch.matmul(uflat, self.W_val_u).view(shp_u)

        Q_f = torch.matmul(fflat, self.W_query_f).view(shp_f)
        K_f = torch.matmul(fflat, self.W_key_f).view(shp_f)
        V_f = torch.matmul(fflat, self.W_val_f).view(shp_f)

        # Calculate compatibility (n_heads, batch_size, u_s, f_s)
        compatibility_u = self.norm_factor * torch.matmul(Q_u, K_u.transpose(2, 3))
        compatibility_f = self.norm_factor * torch.matmul(Q_f, K_f.transpose(2, 3))
        compatibility_Int = self.norm_factor * torch.matmul(Q_u, K_f.transpose(2, 3))

        ## Interaction attention encoder
        compatibility_finally = torch.matmul((torch.cat((compatibility_u, compatibility_Int), dim=-1)),
                                             (torch.cat((compatibility_f, compatibility_Int), dim=2)))
        attn_u = F.softmax(compatibility_finally, dim=-1)
        attn_f = attn_u.permute(0, 1, 3, 2)
        heads_u = torch.matmul(attn_u, V_f)
        heads_f = torch.matmul(attn_f, V_u)

        out_u = torch.mm(
            heads_u.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out_u.view(-1, self.embed_dim)
        ).view(batch_size, users_size, self.embed_dim)

        out_f = torch.mm(
            heads_f.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out_f.view(-1, self.embed_dim)
        ).view(batch_size, fac_size, self.embed_dim)

        return out_u, out_f


class MultiHeadAttentionNew(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionNew, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.score_aggr = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, out_source_attn):

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()

        hflat = h.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (4, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, bastch_size, n_query, graph_size)
        compatibility = torch.cat((torch.matmul(Q, K.transpose(2, 3)), out_source_attn), 0)

        attn_raw = compatibility.permute(1, 2, 3, 0)
        attn = self.score_aggr(attn_raw).permute(3, 0, 1, 2)
        heads = torch.matmul(F.softmax(attn, dim=-1), V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)

        return out, out_source_attn


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(input.var((1, 2)).view(-1, 1, 1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadEncoder, self).__init__()

        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input1, input2):
        out1, out2 = self.MHA_sublayer(input1, input2)
        out_u, out_f = self.FFandNorm_sublayer(out1, out2)
        return out_u, out_f


class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadInteractiveAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input1, input2):
        # Attention and Residual connection
        out1, out2 = self.MHA(input1, input2)

        # Normalization
        return self.Norm(out1 + input1), self.Norm(out2 + input2)


class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()

        self.FF_u = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim, bias=False)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim, bias=False)

        self.FF_f = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim, bias=False)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim, bias=False)

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, u, f):
        # FF and Residual connection
        out_u = self.FF_u(u)
        out_f = self.FF_f(f)

        # Normalization
        return self.Norm(out_u + u), self.Norm(out_f + f)


class EmbeddingNet(nn.Module):

    def __init__(
            self,
            node_dim,
            embedding_dim,
    ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(node_dim, embedding_dim, bias=False)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, users, facilities):
        x = torch.cat((users, facilities), dim=1)
        embedding = self.embedding(x)
        u_embedding, f_embedding = torch.split(embedding, users.shape[1], 1)

        return u_embedding, f_embedding


class MultiHeadDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            v_range=6,
    ):
        super(MultiHeadDecoder, self).__init__()
        self.n_heads = n_heads = 1
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.range = v_range
        self.mask_logits = True

        if TYPE_REMOVAL == 'N2S':
            self.removal = Removal(n_heads,
                                   input_dim,
                                   embed_dim,
                                   key_dim,
                                   val_dim)
            self.insertion = Insertion(n_heads,
                                       input_dim,
                                       embed_dim,
                                       key_dim,
                                       val_dim)
        self.project_graph_u = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_u = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        #
        # self.project_graph_f = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # self.project_node_f = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.project_graph_p = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_p = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.project_graph_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, problem, users_em, facilities_em, solution, pre_action, selection_sig, fixed_action=None,
                require_entropy=False):

        bs, us, dim = users_em.size()
        _, fs, _ = facilities_em.size()
        _, p_size = solution.size()
        cand_fac_size = fs - p_size

        arrange = torch.arange(fs).repeat(bs, 1).to(solution.device)
        mask = torch.zeros((bs, fs), device=solution.device)
        mask.scatter_(1, solution, 1)
        indices = (~mask.bool()).type(torch.int)
        C = torch.masked_select(arrange, indices == 1)
        cand_list = C.reshape(-1, cand_fac_size)

        ## Split F into medians set and candidate facilities set
        p_index = solution.unsqueeze(-1).expand_as(torch.Tensor(bs, p_size, dim))
        p_em = facilities_em.gather(1, p_index)
        q_index = cand_list.unsqueeze(-1).expand_as(torch.Tensor(bs, cand_fac_size, dim))
        q_em = facilities_em.gather(1, q_index)

        selection_sig = selection_sig.permute(0, 2, 1)
        action1_index = solution.unsqueeze(-1).expand_as(torch.Tensor(bs, p_size, 4))
        action2_index = cand_list.unsqueeze(-1).expand_as(torch.Tensor(bs, cand_fac_size, 4))
        selection_sig_action1 = selection_sig.gather(1, action1_index)
        selection_sig_action2 = selection_sig.gather(1, action2_index)

        u = self.project_node_u(users_em) + self.project_graph_u(users_em.max(1)[0])[:, None, :].expand(bs, us, dim)
        p = self.project_node_p(p_em) + self.project_graph_p(p_em.max(1)[0])[:, None, :].expand(bs, p_size, dim)
        q = self.project_node_q(q_em) + self.project_graph_q(q_em.max(1)[0])[:, None, :].expand(bs, cand_fac_size, dim)

        ## Action 1 Removal
        if TYPE_REMOVAL == 'N2S':
            compat_removal = self.removal(u, p, selection_sig_action1)
            action_removal = torch.tanh(compat_removal.squeeze()) * self.range
            log_ll_removal = F.log_softmax(action_removal, dim=-1) if self.training and TYPE_REMOVAL == 'N2S' else None
            probs_removal = F.softmax(action_removal, dim=-1)

        elif TYPE_REMOVAL == 'random':
            probs_removal = torch.rand(bs, p_size).to(p.device)
        else:
            pass
            # epi-greedy
            # first_row = torch.arange(gs, device = rec.device).long().unsqueeze(0).expand(bs, gs)
            # d_i =  x_in.gather(1, first_row.unsqueeze(-1).expand(bs, gs, 2))
            # d_i_next = x_in.gather(1, rec.long().unsqueeze(-1).expand(bs, gs, 2))
            # d_i_pre = x_in.gather(1, rec.argsort().long().unsqueeze(-1).expand(bs, gs, 2))
            # cost_ = ((d_i_pre  - d_i).norm(p=2, dim=2) + (d_i  - d_i_next).norm(p=2, dim=2) - (d_i_pre  - d_i_next).norm(p=2, dim=2))[:,1:]
            # probs_removal = (cost_[:,:gs//2] + cost_[:,gs//2:])
            # probs_removal_random = torch.rand(bs, gs//2).to(h_em.device)

        if fixed_action is not None:
            action_removal = fixed_action[:, :1]
            action_removal_index = torch.where(action_removal == solution)[1].unsqueeze(-1)

        else:
            if TYPE_REMOVAL == 'greedy':
                pass
                # action_removal_random = probs_removal_random.multinomial(1)
                # action_removal_greedy = probs_removal.max(-1)[1].unsqueeze(1)
                # action_removal = torch.where(torch.rand(bs, 1).to(h_em.device) < 0.1, action_removal_random,
                #                              action_removal_greedy)
            else:
                action_removal_index = probs_removal.multinomial(1)
                action_removal = torch.gather(solution, 1, action_removal_index)

        selected_log_ll_action1 = log_ll_removal.gather(1, action_removal_index) \
            if self.training and TYPE_REMOVAL == 'N2S' else torch.zeros((bs, 1)).to(u.device)

        ## Action 2 Insertion
        if TYPE_INSERTION == "N2S":
            compat_insertion = self.insertion(u, q, selection_sig_action2)
            action_insertion = torch.tanh(compat_insertion.squeeze()) * self.range
            log_ll_insertion = F.log_softmax(action_insertion,
                                             dim=-1) if self.training and TYPE_REMOVAL == 'N2S' else None
            probs_insertion = F.softmax(action_insertion, dim=-1)
        elif TYPE_INSERTION == 'random':
            probs_insertion = torch.rand(bs, cand_fac_size).to(q.device)
        else:
            pass
            # epi-greedy
            # first_row = torch.arange(gs, device = rec.device).long().unsqueeze(0).expand(bs, gs)
            # d_i =  x_in.gather(1, first_row.unsqueeze(-1).expand(bs, gs, 2))
            # d_i_next = x_in.gather(1, rec.long().unsqueeze(-1).expand(bs, gs, 2))
            # d_i_pre = x_in.gather(1, rec.argsort().long().unsqueeze(-1).expand(bs, gs, 2))
            # cost_ = ((d_i_pre  - d_i).norm(p=2, dim=2) + (d_i  - d_i_next).norm(p=2, dim=2) - (d_i_pre  - d_i_next).norm(p=2, dim=2))[:,1:]
            # probs_removal = (cost_[:,:gs//2] + cost_[:,gs//2:])
            # probs_removal_random = torch.rand(bs, gs//2).to(h_em.device)
        if fixed_action is not None:
            action_insertion = fixed_action[:, 1:]
            action_insertion_index = torch.where(action_insertion == cand_list)[1].unsqueeze(-1)
        else:
            if TYPE_INSERTION == 'greedy':
                pass
            else:
                action_insertion_index = probs_insertion.multinomial(1)
                action_insertion = torch.gather(cand_list, 1, action_insertion_index)
        selected_log_ll_action2 = log_ll_insertion.gather(1, action_insertion_index) \
            if self.training and TYPE_REMOVAL == 'N2S' else torch.zeros((bs, 1)).to(u.device)

        action = torch.cat([action_removal.view(bs, -1), action_insertion.view(bs, -1)], dim=-1)
        # pos_pickup = (1 + action_removal).view(-1)
        # pos_delivery = pos_pickup + half_pos
        # mask_table = problem.get_swap_mask(action_removal + 1, visited_order_map, top2).expand(bs, gs, gs).cpu()

        log_ll = selected_log_ll_action1 + selected_log_ll_action2

        if require_entropy and self.training:
            dist = Categorical(probs_insertion, validate_args=False)
            entropy = dist.entropy()
        else:
            entropy = None

        return action, log_ll, entropy


class MultiHeadDecoder_pair(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            v_range=6,
    ):
        super(MultiHeadDecoder_pair, self).__init__()
        self.n_heads = n_heads = 1
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.range = v_range

        if TYPE_REMOVAL == 'N2S':
            self.compat_action_pair = Compat_action_pair(n_heads,
                                                         input_dim,
                                                         embed_dim,
                                                         key_dim,
                                                         val_dim)

        self.project_graph_f = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_f = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, problem, users_em, facilties_em, solution, pre_action, selection_sig, fixed_action=None,
                require_entropy=False):

        bs, us, dim = users_em.size()
        _, fs, _ = facilties_em.size()
        _, p_size = solution.size()
        cand_fac_size = fs - p_size

        mask = torch.zeros((bs, fs, fs)).to(solution.device)
        mask[torch.arange(bs).unsqueeze(1), solution] = 1
        mask[torch.arange(bs).unsqueeze(1), :, solution] = 0
        bool_mask = mask.bool()

        f = self.project_node_f(facilties_em) + self.project_graph_f(facilties_em.max(1)[0])[:, None, :].expand(bs, fs,
                                                                                                                dim)

        ## Action pair
        if TYPE_REMOVAL == 'N2S':
            compat_action_pair = self.compat_action_pair(f, bool_mask)
            compat_action_pair = torch.tanh(compat_action_pair.squeeze()) * self.range
            compat_action_pair[~bool_mask] = -np.infty
            action = compat_action_pair.view(bs, -1)
            log_ll = F.log_softmax(action, dim=-1) if self.training and TYPE_REMOVAL == 'N2S' else None
            # log_mask = bool_mask.view(bs, -1)
            # log_ll[~log_mask] = 0
            probs = F.softmax(action, dim=-1)
        elif TYPE_REMOVAL == 'random':
            probs_removal = torch.rand(bs, p_size).to(solution.device)
            probs_insertion = torch.rand(bs, cand_fac_size).to(f.device)
        else:
            pass

        if fixed_action is not None:
            action_pair = fixed_action
            action_removal = fixed_action[:, :1]
            action_insertion = fixed_action[:, 1:]
            action = action_removal * fs + action_insertion
            # action_removal_index = torch.where(action_removal == solution)[1].unsqueeze(-1)
            # action_insertion_index = torch.where(action_insertion == cand_list)[1].unsqueeze(-1)

        else:
            if TYPE_REMOVAL == 'greedy':
                pass
                # action_removal_random = probs_removal_random.multinomial(1)
                # action_removal_greedy = probs_removal.max(-1)[1].unsqueeze(1)
                # action_removal = torch.where(torch.rand(bs, 1).to(h_em.device) < 0.1, action_removal_random,
                #                              action_removal_greedy)
            else:
                action = probs.multinomial(1)
                action_removal = action // fs
                action_insertion = action % fs
                # action_insertion_index = probs_insertion.multinomial(1)
                # action_insertion = torch.gather(cand_list, 1, action_insertion_index)
                action_pair = torch.cat([action_removal.view(bs, -1), action_insertion.view(bs, -1)], dim=-1)

        if require_entropy and self.training:
            dist = Categorical(probs, validate_args=False)
            entropy = dist.entropy()
        else:
            entropy = None

        selected_log_ll = log_ll.gather(1, action) \
            if self.training and TYPE_REMOVAL == 'N2S' else torch.zeros((bs, 1)).to(f.device)

        return action_pair, selected_log_ll, entropy


class Compat_action_pair(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(Compat_action_pair, self).__init__()

        n_heads = 4

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_Q_f = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K_f = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, f, mask):
        batch_size, f_size, input_dim = f.size()

        f_flat = f.contiguous().view(-1, input_dim)  #################   reshape

        # last dimension can be different for keys and values
        f_shp = (self.n_heads, batch_size, f_size, -1)

        # Calculate facilities queries, (n_heads, bs, u_size, key/val_size)
        f_hidden_Q = torch.matmul(f_flat, self.W_Q_f).view(f_shp)
        f_hidden_K = torch.matmul(f_flat, self.W_K_f).view(f_shp)

        # Calculate compatibility (n_heads, batch_size, p_size, 2*u_size)
        compatibility = self.norm_factor * torch.matmul(f_hidden_Q, f_hidden_K.transpose(2, 3))
        compatibility_pair = compatibility.permute(1, 2, 3, 0).sum(-1)

        compatibility_pair[~mask] = - np.infty

        return compatibility_pair


class Removal(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(Removal, self).__init__()

        n_heads = 4

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_Q_u = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K_u = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val_u = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_Q_p = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K_p = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val_p = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.agg = MLP(n_heads + 4, 32, 32, 1, 0)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, u, p, selection_sig):
        batch_size, u_size, input_dim = u.size()
        _, p_size, _ = p.size()

        u_flat = u.contiguous().view(-1, input_dim)  #################   reshape
        p_flat = p.contiguous().view(-1, input_dim)  #################   reshape

        # last dimension can be different for keys and values
        u_shp = (self.n_heads, batch_size, u_size, -1)
        p_shp = (self.n_heads, batch_size, p_size, -1)

        # Calculate users queries, (n_heads, bs, u_size, key/val_size)
        u_hidden_Q = torch.matmul(u_flat, self.W_Q_u).view(u_shp)
        u_hidden_K = torch.matmul(u_flat, self.W_K_u).view(u_shp)

        u_Q_sum = torch.mean(u_hidden_Q, dim=2).view(self.n_heads, batch_size, 1, -1)
        u_K_sum = torch.mean(u_hidden_K, dim=2).view(self.n_heads, batch_size, 1, -1)

        # Calculate medians queries, (n_heads, n_query, p_size, key/val_size)
        p_hidden_Q = torch.matmul(p_flat, self.W_Q_p).view(p_shp)
        p_hidden_K = torch.matmul(p_flat, self.W_K_p).view(p_shp)

        # Calculate compatibility (n_heads, batch_size, p_size, 2*u_size)
        compatibility_1 = self.norm_factor * torch.matmul(p_hidden_K, u_Q_sum.transpose(2, 3))
        compatibility_2 = self.norm_factor * torch.matmul(p_hidden_Q, u_K_sum.transpose(2, 3))
        compatibility_p = torch.cat([compatibility_1, compatibility_2], dim=-1).sum(-1)

        # compatibility_remove = self.agg(compatibility_p.permute(1, 2, 0).squeeze())

        # compatibility_pairing = compatibility.permute(1, 2, 0, 3).contiguous().view(batch_size, p_size,
        #                                                                             self.n_heads * 2 * u_size)

        compatibility_remove = self.agg(torch.cat((compatibility_p.permute(1, 2, 0),
                                                   selection_sig.permute(0, 1, 2)), -1)).squeeze()
        # compatibility_insertion = self.agg2(torch.cat((compatibility_cf.permute(1, 2, 0),
        #                                             selection_sig2.permute(0, 1, 2)), -1)).squeeze()

        return compatibility_remove


class Insertion(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(Insertion, self).__init__()

        n_heads = 4

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_Q_u = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K_u = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val_u = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_Q_cf = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K_cf = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val_cf = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.agg = MLP(n_heads + 4, 32, 32, 1, 0)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, u, cf, selection_sig):
        batch_size, u_size, input_dim = u.size()
        _, cf_size, _ = cf.size()

        u_flat = u.contiguous().view(-1, input_dim)  #################   reshape
        cf_flat = cf.contiguous().view(-1, input_dim)  #################   reshape

        # last dimension can be different for keys and values
        u_shp = (self.n_heads, batch_size, u_size, -1)
        cf_shp = (self.n_heads, batch_size, cf_size, -1)

        # Calculate users queries, (n_heads, bs, u_size, key/val_size)
        u_hidden_Q = torch.matmul(u_flat, self.W_Q_u).view(u_shp)
        u_hidden_K = torch.matmul(u_flat, self.W_K_u).view(u_shp)

        u_Q_sum = torch.mean(u_hidden_Q, dim=2).view(self.n_heads, batch_size, 1, -1)
        u_K_sum = torch.mean(u_hidden_K, dim=2).view(self.n_heads, batch_size, 1, -1)

        # Calculate candiate facility queries, (n_heads, n_query, cf_size, key/val_size)
        cf_hidden_Q = torch.matmul(cf_flat, self.W_Q_cf).view(cf_shp)
        cf_hidden_K = torch.matmul(cf_flat, self.W_K_cf).view(cf_shp)

        # Calculate compatibility (n_heads, batch_size, p_size, 2*u_size)
        compatibility_3 = self.norm_factor * torch.matmul(cf_hidden_K, u_Q_sum.transpose(2, 3))
        compatibility_4 = self.norm_factor * torch.matmul(cf_hidden_Q, u_K_sum.transpose(2, 3))
        compatibility_cf = torch.cat([compatibility_3, compatibility_4], dim=-1).sum(-1)

        # compatibility_insertion = self.agg(compatibility_cf.permute(1, 2, 0).squeeze())

        # compatibility_pairing = compatibility.permute(1, 2, 0, 3).contiguous().view(batch_size, p_size,
        #                                                                             self.n_heads * 2 * u_size)

        # compatibility_remove = self.agg1(torch.cat((compatibility_p.permute(1, 2, 0),
        #                                             selection_sig1.permute(0, 1, 2)), -1)).squeeze()
        compatibility_insertion = self.agg(torch.cat((compatibility_cf.permute(1, 2, 0),
                                                      selection_sig.permute(0, 1, 2)), -1)).squeeze()

        return compatibility_insertion


class MLP(torch.nn.Module):
    def __init__(self,
                 input_dim=128,
                 feed_forward_dim=64,
                 embedding_dim=64,
                 output_dim=1,
                 p_dropout=0.01
                 ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.ReLU = nn.ReLU(inplace=True)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q):

        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)  #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class MultiHeadAttentionLayerforCritic(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(feed_forward_hidden, embed_dim, )
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class ValueDecoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            input_dim,
    ):
        super(ValueDecoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.project_graph_u = nn.Linear(self.input_dim, self.embed_dim // 2)
        self.project_graph_f = nn.Linear(self.input_dim, self.embed_dim // 2)

        self.project_node_u = nn.Linear(self.input_dim, self.embed_dim // 2)
        self.project_node_f = nn.Linear(self.input_dim, self.embed_dim // 2)

        self.MLP = MLP(input_dim * 2 + 1, embed_dim)

    def forward(self, u_em, f_em, cost):
        # get embed feature
        #        max_pooling = h_em.max(1)[0]   # max Pooling
        mean_pooling_u = u_em.mean(1)  # mean Pooling
        mean_pooling_f = f_em.mean(1)  # mean Pooling

        graph_feature_u = self.project_graph_u(mean_pooling_u)[:, None, :]
        graph_feature_f = self.project_graph_f(mean_pooling_f)[:, None, :]
        node_feature_u = self.project_node_u(u_em)
        node_feature_f = self.project_node_f(f_em)

        # pass through value_head, get estimated value
        fusion_u = node_feature_u + graph_feature_u.expand_as(node_feature_u)  # torch.Size([2, 50, 128])
        fusion_f = node_feature_f + graph_feature_f.expand_as(node_feature_f)  # torch.Size([2, 50, 128])

        fusion_feature = torch.cat((fusion_u.mean(1),
                                    fusion_u.max(1)[0],
                                    fusion_f.mean(1),
                                    fusion_f.max(1)[0],
                                    cost.to(u_em.device),
                                    ), -1)

        value = self.MLP(fusion_feature)

        return value
