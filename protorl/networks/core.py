import torch as T


class NetworkCore:
    def __init__(self, device=None, *args, **kwargs):
        super().__init__()

        self.device = device or T.device('cuda:0' if T.cuda.is_available()
                                         else 'cpu')
