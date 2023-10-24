import torch 
import torch.nn as nn
import random

torch.manual_seed(11)
random.seed(11)


# 3-dim Cross entropy
# 4 class
# bs = 4
i = torch.randn((4, 4, 3))

i_dim1 = i[:, :, 0].clone()
i_dim2 = i[:, :, 1].clone()
i_dim3 = i[:, :, 2].clone()

t_dim1 = torch.ones((4, 4, 1))
t_dim2 = torch.ones((4, 4, 1))
t_dim3 = torch.ones((4, 4, 1))

t = torch.cat([t_dim1, t_dim2, t_dim3], dim=2)

loss_fn = nn.CrossEntropyLoss(reduction='none')
loss_t = loss_fn(i, t)
print(loss_t)
print(loss_t.shape)
print('-'*20)

loss_1 = loss_fn(i_dim1.squeeze(), t_dim1.squeeze())
loss_2 = loss_fn(i_dim2.squeeze(), t_dim2.squeeze())
loss_3 = loss_fn(i_dim3.squeeze(), t_dim3.squeeze())

print(loss_1.shape)
print(loss_1)
print('+'*10)

print(loss_2.shape)
print(loss_2)
print('+'*10)

print(loss_3.shape)
print(loss_3)
print('+'*10)


