import torch
import torch.nn as nn
import math
import torch.optim as optim
import copy
from torch.nn import functional as F, init


class ModelAdaptHeads(nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, num_head))
        self.bias = nn.Parameter(torch.ones(1, num_head) / 8)
        init.uniform_(self.weight, 0.75, 1.25)

    def forward(self, y, inverse=False):
        if inverse:
            return (y.view(-1, 1) - self.bias) / (self.weight + 1e-9)
        else:
            return (self.weight + 1e-9) * y.view(-1, 1) + self.bias


class ModelAdapter(nn.Module):
    def __init__(self, x_dim, num_head=4, temperature=4, hid_dim=32):
        super().__init__()
        self.num_head = num_head
        self.linear = nn.Linear(x_dim, hid_dim, bias=False)
        self.P = nn.Parameter(torch.empty(num_head, hid_dim))
        init.kaiming_uniform_(self.P, a=math.sqrt(5))
        # self.heads = nn.ModuleList([LabelAdaptHead() for _ in range(num_head)])
        self.heads = ModelAdaptHeads(num_head)
        self.temperature = temperature

    def forward(self, x, y, inverse=False):
        v = self.linear(x.reshape(len(x), -1))
        gate = self.cosine(v, self.P)
        gate = torch.softmax(gate / self.temperature, -1)
        # return sum([gate[:, i] * self.heads[i](y, inverse=inverse) for i in range(self.num_head)])
        return (gate * self.heads(y, inverse=inverse)).sum(-1)

    def cosine(x1, x2, eps=1e-8):
        x1 = x1 / (torch.norm(x1, p=2, dim=-1, keepdim=True) + eps)
        x2 = x2 / (torch.norm(x2, p=2, dim=-1, keepdim=True) + eps)
        return x1 @ x2.transpose(0, 1)


# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


# 定义MAML算法
class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr_outer)

    def inner_update(self, task_data, model):
        # model_copy = copy.deepcopy(model)
        inner_optimizer = optim.SGD(self.model.parameters(), lr=self.lr_inner)

        # 内循环（在任务数据上进行梯度更新）
        for x, y in task_data:
            y_pred = self.model(x)
            loss = nn.MSELoss()(y_pred, y)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        # 返回内循环后的模型参数
        return self.model.state_dict()

    def meta_update(self, tasks):
        # meta_grads = []

        # 外循环（在多个任务上进行元梯度更新）
        for task_data in tasks:
            # 备份模型当前状态
            original_state = {
                name: param.clone() for name, param in self.model.named_parameters()
            }

            # 在任务上进行内循环，获取内循环后的模型参数
            inner_params = self.inner_update(task_data, model)

            # model_copy = self.inner_update(task_data, self.model)
            # 应用内循环后的参数进行外循环损失计算
            outer_loss = 0
            for x, y in task_data:
                y_pred = self.model(x)
                outer_loss += nn.MSELoss()(y_pred, y)

            # 计算外梯度
            self.optimizer.zero_grad()
            outer_loss.backward()

            # 计算元梯度并更新模型参数
            meta_grad = {
                name: param.grad for name, param in self.model.named_parameters()
            }
            for name, param in self.model.named_parameters():
                param.data -= self.lr_outer * meta_grad[name]


if __name__ == "__main__":
    # 示例用法
    # 创建模型和元学习器
    model = SimpleModel()
    maml = MAML(model)

    # 定义多个任务数据集
    tasks = [
        [(torch.tensor([1.0]), torch.tensor([2.0]))],
        [(torch.tensor([2.0]), torch.tensor([4.0]))],
        [(torch.tensor([4.0]), torch.tensor([8.0]))],
        # 可以添加更多的任务
    ]

    # 进行元学习
    for epoch in range(1000):  # 假设进行1000个元训练周期
        maml.meta_update(tasks)

    # 使用元学习后的模型进行预测
    new_data = torch.tensor([3.0])
    prediction = model(new_data)
    print("预测结果:", prediction.item())
