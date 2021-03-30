import numpy as np
import torch


# 生成随机数据
def create_random_data():
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
