import torch
from torch import distributions as td
import numpy as np

# categorical
n_samples = 100

# probabilities that sum-up to one -> equal distributions of 2 classes
probabilities = torch.tensor([0.5, 0.5])
dist = td.Categorical(probs=probabilities)
sample = dist.sample(sample_shape=(1, n_samples)).numpy()
print(np.sum(sample))   # expect ~ half of the sampled size

# unnormalized non-zero probs -> torch will normalize them to sum up to 1
probabilities = torch.tensor([0.1, 0.1])
dist = td.Categorical(probs=probabilities)
sample = dist.sample(sample_shape=(1, n_samples)).numpy()
print(np.sum(sample))   # so this should be comparable to the previous result

# logits can be any real number -> torch inteprets them as unnormalized log-prob
logits = torch.tensor([0.1, -0.5])
dist = td.Categorical(logits=logits)
my_probs = np.exp(logits) / sum(np.exp(logits))    # compute norm probs from logits (softmax)
print(f"dist probs: {dist.probs.numpy()}, my probs: {my_probs.numpy()}")

# categorical with batches
probs = torch.as_tensor(np.random.random((10, 10, 5, 3, 2, 1, 2)))
dist = td.Categorical(logits=probs)
print(f'probs shape: {probs.shape}')
print(f'sample shape: {dist.sample().shape}')

# categorical loglikelihood
probabilities = torch.tensor([0.1, 0.9])
dist = td.Categorical(probs=probabilities)
logprob0 = dist.log_prob(torch.tensor([0]))
prob0 = np.exp(logprob0)
print(prob0)    # expect to match the probability 0.1