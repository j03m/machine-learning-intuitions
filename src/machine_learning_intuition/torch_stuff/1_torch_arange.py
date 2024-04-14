import torch

# basic
t = torch.arange(0, 10)
print(t)

# by 2
t = torch.arange(0, 10, 2)
print(t)


# backwards
t = torch.arange(10, 0, -1)
print(t)

# backwards to -5 by 2 - you need -6 because end is NOT inclusive
t = torch.arange(10, -6, -2)
print(t)

#crashes
# t = torch.arange(0, 10, -1)
# print(t)


# you only need end?
t = torch.arange(10)
print(t)