from torch.distributions import Categorical
import torch.nn.functional as F
import torch.distributions as D
import torch



def diag_normal(x, min_std=0.1, max_std=2.0):
    mean, std = x.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)

def trunc_normal(x, min_std=0.1, max_std=2.0, min=-1, max=1):
    mean, std = x.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    dist = D.independent.Independent(D.normal.Normal(mean, std), 1)
    sample_func = dist.sample
    dist.sample = lambda: torch.clip(sample_func(), min=min, max=max)
    return dist

def normal_tanh(x, min_std=0.01, max_std=1.0):
    mean_, std_ = x.chunk(2, -1)
    mean = torch.tanh(mean_)
    std = max_std * torch.sigmoid(std_) + min_std
    normal = D.normal.Normal(mean, std)
    normal = D.independent.Independent(normal, 1)
    return normal

def categorical(x):
    probs = F.softmax(x, dim=-1)
    return Categorical(probs)


DISTRIBUTIONS = {
    "discrete": categorical,
    "normal": diag_normal,
    "trunc_normal": trunc_normal,
    "normal_tanh": normal_tanh
}
def get_distribution_func(name):
    return DISTRIBUTIONS[name]