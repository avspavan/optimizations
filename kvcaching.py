import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SimpleAttentionWithCache(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleAttentionWithCache, self).__init__()
        self.feature_dim = feature_dim
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.query = nn.Linear(feature_dim, feature_dim)
        # Initializing cache as None
        self.register_buffer('key_cache', None)
        self.register_buffer('value_cache', None)

    def forward(self, inputs, use_cache=False):
        if use_cache and self.key_cache is not None and self.value_cache is not None:
            # Use cached K and V
            k = self.key_cache
            v = self.value_cache
        else:
            # Compute K and V
            k = self.key(inputs)
            v = self.value(inputs)
            # Update cache
            self.key_cache = k
            self.value_cache = v

        # Compute Q from inputs
        q = self.query(inputs)

        # Calculate attention scores and apply softmax
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attn_weights, v)

        return output

# Instantiate the model
feature_dim = 64
model = SimpleAttentionWithCache(feature_dim)

# Simulating inputs
inputs = torch.rand(1, 10, feature_dim)  # (batch_size, seq_length, feature_dim)

# Measure time without caching
start_time = time.time()
output1 = model(inputs, use_cache=False)
without_cache_time = time.time() - start_time
print(f"Time without KV caching: {without_cache_time:.6f} seconds")

# Reset cache to simulate caching in a new scenario
model.key_cache = None
model.value_cache = None

# First pass to populate the cache (not measuring this, as it's the setup cost)
model(inputs, use_cache=False)

# Now measure with cache being used
start_time = time.time()
output2 = model(inputs, use_cache=True)
with_cache_time = time.time() - start_time
print(f"Time with KV caching: {with_cache_time:.6f} seconds")

# Example run output below
#Time without KV caching: 0.000845 seconds
#Time with KV caching: 0.000073 seconds

