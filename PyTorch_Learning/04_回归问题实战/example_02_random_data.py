import numpy as np
import torch

true_w = [2, -3.4]
true_b = 4.2
save = False


# 生成随机数据
def create_random_data():
    # 生成随机的x值，1000行，2列
    num_inputs = 2
    num_examples = 1000
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    # 计算对应的真实y值
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # 给y值增加噪声
    error = torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    labels += error
    # 保存随机生成的数据，[x1,x2,y]格式

    if save:
        with open('data_02.csv', 'w', encoding='utf-8') as f:
            for i in range(num_examples):
                print(f'{features[i, 0]},{features[i, 1]},{labels[i]}', file=f)

    print(error)


if __name__ == '__main__':
    create_random_data()
