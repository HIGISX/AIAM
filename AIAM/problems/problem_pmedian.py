from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np
import random
# TYPE_LOSS = "PROFIT"
TYPE_LOSS = "ORIGIN"


class pMedian(object):
    NAME = 'pMedian'  # p-Median Problem

    def __init__(self, n_users, n_facilities, p, init_val_met='p2d', with_assert=False):

        self.n_users = n_users  # the number of demand points in p-Median problem
        self.n_facilities = n_facilities  # the number of facilities in p-Median problem
        self.p = p
        self.do_assert = with_assert
        self.init_val_met = init_val_met
        self.state = 'eval'
        print(f'pMedian with {self.n_users} users, {self.n_facilities} facilities, {self.p} medians.'
              ' Do assert:', with_assert, )

    def input_feature_encoding(self, batch):
        users = batch['users']
        facilities = batch['facilities']
        return users, facilities

    def get_initial_solutions(self, batch, val_m=1):
        users = batch['users']
        facilities = batch['facilities']
        m = self.n_facilities
        p = self.p
        batch_size = batch['facilities'].size(0)

        def get_solution(methods):

            if methods == 'random':

                recs = []
                for i in range(batch_size):
                    rec = random.sample(range(m), p)
                    recs.append(rec)
                return torch.tensor(recs, dtype=torch.int64)
            # if methods == 'random':
            #     torch.manual_seed(0)
            #     rec = torch.randint(low=0, high=20, size=(batch_size, 4))
            #     return rec
            # *------------------------------------------------------------------
            # | Greedy: adds facilities to s one at a time, until p facilities
            # |   are inserted. The running time is O(nk), where k is the
            # |   number of times cities 'change hands' during the execution of
            # |   the algorithm. One can find trivial upper and lower bounds for
            # |   k: n <= k <= np (n is a lower bound if we start from an empty
            # |   solution; if we start with something else we may have k<n).
            # |
            # | The input solution need not be empty; all facilities already
            # | there will be preserved.
            # *-----------------------------------------------------------------*
            elif methods == 'greedy':
                dist = (users[:, :, None, :] - facilities[:, None, :, :]).norm(p=2, dim=-1)
                dist_m = torch.sum(dist, dim=1)
                min_dist, rec = torch.topk(dist_m, k=p, dim=1, largest=False)
                # return torch.tensor(rec, dtype=torch.int64)
                return rec

            else:
                raise NotImplementedError()

        return get_solution(methods="greedy").expand(batch_size, self.p).clone()

    def step(self, batch, rec, exchange, pre_bsf, action_record):

        bs, gs = rec.size()
        pre_bsf = pre_bsf.view(bs, -1)

        cur_vec = action_record.pop(0) * 0.
        cur_vec[torch.arange(bs), exchange[:, 0]] = 1
        cur_vec[torch.arange(bs), exchange[:, 1]] = 1
        action_record.append(cur_vec)

        action_removal = exchange[:, 0].view(bs, 1)
        action_insertion = exchange[:, 1].view(bs, 1)

        next_state = self.interchange(rec, action_removal, action_insertion)
        new_obj = self.get_costs(batch, next_state)

        now_bsf = torch.min(torch.cat((new_obj[:, None], pre_bsf[:, -1, None]), -1), -1)[0]
        reward = pre_bsf[:, -1] - now_bsf
        # reward = pre_bsf[:, -1] - new_obj
        return next_state, reward, torch.cat((new_obj[:, None], now_bsf[:, None]), -1), action_record

    def interchange(self, solution, action_removal, action_insertion):

        rec = solution.clone()
        bs, ps = rec.size()

        mask = rec != action_removal

        af_removal = rec[mask].reshape(bs, ps - 1)
        af_insertion = torch.cat([af_removal, action_insertion], dim=1)

        return af_insertion

    def get_costs(self, batch, rec):

        batch_size, size = rec.size()
        n = self.n_users
        m = self.n_facilities
        p = self.p

        users = batch['users']
        facilities = batch['facilities']
        dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)

        facility_tensor = rec.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n))
        dist_p = dist.gather(1, facility_tensor)
        length = torch.min(dist_p, 1)
        lengths = length[0].sum(-1)

        return lengths

    def get_loss(self, batch, rec, action_removal):
        batch_size, size = rec.size()
        n = self.n_users
        m = self.n_facilities
        p = self.p

        sort_rec = torch.sort(rec, dim=1)[0]
        ids = torch.nonzero(torch.eq(sort_rec, action_removal))
        action_id = ids[:, 1].unsqueeze(-1)

        users = batch['users']
        facilities = batch['facilities']
        dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)

        facility_tensor = rec.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n))
        dist_p = dist.gather(1, facility_tensor)
        dist_top2, indices = torch.topk(dist_p, k=2, largest=False, dim=1)
        mask_indices = indices[:, 0, :]
        mask = torch.eq(mask_indices, action_id)
        mask = mask.unsqueeze(1).repeat(1, 2, 1)
        dist_top2[~mask] = 0
        phi_1 = dist_top2[:, 0, :]
        phi_2 = dist_top2[:, 1, :]

        loss = (phi_2 - phi_1).sum(-1)

        return loss

    def get_gain(self, batch, rec, action_insertion):
        batch_size, size = rec.size()
        n = self.n_users
        m = self.n_facilities
        p = self.p
        users = batch['users']
        facilities = batch['facilities']
        dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)
        facility_tensor = rec.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n))
        dist_p = dist.gather(1, facility_tensor)
        dist_min = torch.min(dist_p, 1)[0]

        facility_insertion = action_insertion.unsqueeze(-1).expand_as(torch.Tensor(batch_size, 1, n))
        dist_insertion = dist.gather(1, facility_insertion).squeeze(1)

        now_cloest_fac = torch.min(torch.cat((dist_insertion[:, None, :], dist_min[:, None, :]), dim=1), dim=1)[0]

        gain = (dist_min - now_cloest_fac).sum(1)
        return gain


    @staticmethod
    def make_dataset(*args, **kwargs):
        return PMDataset(*args, **kwargs)


class PMDataset(Dataset):
    def __init__(self, filename=None, n_users=20, n_facilities=10, p=4, num_samples=10000, offset=0, distribution=None):

        super(PMDataset, self).__init__()

        self.data = []
        self.n_users = n_users
        self.n_facilities = n_facilities
        self.p = p

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = data

        else:
            self.data = [{
                'users': torch.FloatTensor(self.n_users, 2).uniform_(0, 1),
                'facilities': torch.FloatTensor(self.n_facilities, 2).uniform_(0, 1)} for i in range(num_samples)]

        self.N = len(self.data)

        print(f'{self.N} instances initialized.')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]
