import torch
import torch.distributions as dist
import math

m = dist.Normal(torch.ones((64, 1, 3, 32, 32)), scale=0.0001)
print(m.scale.shape)
print(m.rsample())
# m = dist.Normal(torch.tensor([
#                     [0.0, 10.0],
#                     [0.0, -10.0]]), torch.tensor([
#                                             [1.0, 1.5],
#                                             [1.0, 1.5]
#                                         ]))
# print(m, m.loc, m.scale)
# print(m.rsample())

# l = dist.Laplace(torch.randn(128, 3, 32, 32), scale=torch.tensor(0.75))
# print(l.loc[0:5])
# print(l.scale[0:5])


print(torch.log(torch.tensor(0.75)))
print(torch.log(torch.tensor(0.10)))

# class dummy_distr():
#     def __init__(self) -> None:
#         self.scale = torch.zeros(128, 3, 32, 32).fill_(0.75)
#         self.loc = torch.randn(128, 3, 32, 32)
    
#     def log_prob(self, value):
#         # laplace log_prob
#         # return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale

#         # normal log_prob
#         var = (self.scale ** 2)
#         log_scale = math.log(self.scale) 
#         return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    
# a = dummy_distr()
# print(a.log_prob(torch.zeros(128, 3, 32, 32)).sum())

# aa = torch.ones((1, 10, 2))
# bb = torch.ones((10, 2))
# print(torch.abs(bb- aa).shape)

zss = [torch.randn(5, 2), torch.randn(5, 2)]
x = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
print(x)

# grad = {'requires_grad': True}

# print(isinstance(m, dist.Laplace))

# m = dist.Normal
# print(issubclass(m, dist.Normal))
