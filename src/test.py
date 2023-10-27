import torch 
import torch.nn as nn
import random

torch.manual_seed(11)
random.seed(11)


# 3-dim Cross entropy
# 4 class
# bs = 4
bs = 2
i = torch.randn((bs, 4, 3))

t1 = torch.tensor([1, 1, 0, 0], dtype=torch.int64)


loss_fn_2 = nn.NLLLoss(reduction='none')

for d in i:
    print(d)
    print(t1)
    loss_t_2 = loss_fn_2(d, t1)
    print(loss_t_2)
    print('--')






exit()

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

print((loss_1 + loss_2 + loss_3).mean())


loss_fn = nn.CrossEntropyLoss()
loss_t = loss_fn(i, t)
print(loss_t)