import numpy as np
import torch
import torch.distributions as td

# multivariate gaussian distribution, represented by mean mu and cov matrix Sigma

# diagonal gaussian are a special case with diagonal covariance matrix (then represented by the diagonal vector)
mu = torch.Tensor([0.0, 0.0, 0.0])
stddev = torch.Tensor([1.0, 1.0, 1.0])

# common choice: represent std dev as log std dev
mu = torch.Tensor([0.0, 0.0, 0.0])
logstddev = torch.Tensor([0.0, 0.0, 0.0])  # in this way, it can be any real value

# normal
mu = torch.Tensor([0.0, 2.0])
logstddev = torch.Tensor([0.0, -1.0])  # in this way, it can be any real value
dist = td.Normal(loc=mu, scale=np.exp(logstddev))
samples = dist.sample(sample_shape=[10000])

import matplotlib.pyplot as plt

plt.hist(samples[:, 0])
plt.hist(samples[:, 1])
plt.show()

# pytorch accept Normal dist with negative std dev, however not sampling with neg stddev when using torch.normal(...)
plt.clf()
dist1 = td.Normal(loc=0, scale=-15)
dist2 = td.Normal(loc=0, scale=15)
samples1 = dist1.sample(sample_shape=[1000000])
samples2 = dist2.sample(sample_shape=[1000000])
bins = np.linspace(-100, 100, 100)
plt.hist(samples1, bins=bins)
plt.hist(samples2, bins=bins)
plt.show()

# log prob: computed vs analytical
mu = torch.Tensor([0.0])
logstddev = torch.Tensor([0.0])
dist = td.Normal(loc=mu, scale=np.exp(logstddev))
action = torch.tensor([0.0])
torch_logprob = dist.log_prob(action)

n_log_2pi = action.shape[0] * np.log(2 * np.pi)
my_logprob = -.5 * (sum(((action - mu) ** 2) / (np.exp(logstddev) ** 2) + 2 * logstddev) + n_log_2pi)
print(f"log prob = {torch_logprob} -> prob = {np.exp(torch_logprob)}")
print(f"my log prob = {my_logprob} -> my prob = {np.exp(my_logprob)}")
