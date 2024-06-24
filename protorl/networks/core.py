import os
import torch as T


class NetworkCore:
    def __init__(self, device=None, *args, **kwargs):
        super().__init__()
        self.checkpoint_dir = kwargs['chkpt_dir']
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            kwargs['name'])
        self.name = kwargs['name']

        self.device = device or T.device('cuda:0' if T.cuda.is_available()
                                         else 'cpu')

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
