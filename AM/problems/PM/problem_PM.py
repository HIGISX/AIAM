from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.PM.state_PM import StatePM


class PM(object):
    NAME = 'PM'

    @staticmethod
    def get_total_dis(dataset, pi):
        users = dataset['users']
        facilities = dataset['facilities']

        batch_size, n_users, _ = users.size()
        _, n_facilities, _ = facilities.size()
        _, p = pi.size()

        dist = (facilities[:, :, None, :] - users[:, None, :, :]).norm(p=2, dim=-1)
        facility_tensor = pi.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_users))
        dist_p = dist.gather(1, facility_tensor)
        length = torch.min(dist_p, 1)
        lengths = length[0].sum(-1)

        return lengths


    @staticmethod
    def make_dataset(*args, **kwargs):
        return PMDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePM.initialize(*args, **kwargs)


class PMDataset(Dataset):
    def __init__(self, filename=None, n_users=100, n_facilities=50, num_samples=5000, offset=0, p=8, r=1, distribution=None):
        super(PMDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [row for row in (data[offset:offset + num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                              facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                              p=p, r=r) for i in range(num_samples)
                         ]

        self.size = len(self.data)
        self.p = p
        self.r = r

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
