import torch

# using squeeze - visualize this as 3 groups of 1 row with 3 columns
x = torch.arange(0, 9).reshape(3, 1, 3)
print("pre squeeze: ", x, "shape: ", x.shape)

# squeeze out any dim of 1:
x1 = torch.squeeze(x)
# results in a 3x3 3 rows, 3 columns
print("post squeeze: ", x1, "shape: ", x1.shape)

# using squeeze - add more dims of size 1, they will get squeezed out:
x = torch.arange(0, 9).reshape(3, 1, 1, 1, 3)
print("pre squeeze: ", x, "shape: ", x.shape)

# squeeze out any dim of 1:
x1 = torch.squeeze(x)
# still 3x3
print("post squeeze: ", x1, "shape: ", x1.shape)

x = torch.arange(0, 9).reshape(3, 1, 3)
print("pre squeeze: ", x, "shape: ", x.shape)
x1 = torch.squeeze(x, dim=2)
# no change, dim 2 was not size 1
print("post squeeze: ", x1, "shape: ", x1.shape)

# crashes - not enough values to reshape into 3, 1, 3
# x = torch.arange(0, 7).reshape(4, 1, 3)
#print("pre squeeze: ", x, "shape: ", x.shape)

# But, 12 would be okay: 0,1,2 | 3, 4, 5 | 6,7.8 | 9, 10, 11  (aka 4x1x3)
x = torch.arange(0, 12).reshape(4, 1, 3)
print("pre squeeze: ", x, "shape: ", x.shape)
x1 = torch.squeeze(x)
print("post squeeze: ", x1, "shape: ", x1.shape)

# unsqueeze will add a dimension of 1 where instructed
# Most deep learning models expect inputs to have a batch dimension.
# For example, if your model expects inputs of shape (batch_size, channels, height, width)
# and you have a single image of shape (channels, height, width), you need to add a batch dimension to make it
# (1, channels, height, width) so the model can process it. This is done using unsqueeze.


x = torch.arange(0, 9).reshape(3, 1, 3)
x1 = torch.unsqueeze(x, 0)
print("post unsqueeze 0: ", x1, "shape: ", x1.shape)
x1 = torch.unsqueeze(x, 1)
print("post unsqueeze 1: ", x1, "shape: ", x1.shape)
x1 = torch.unsqueeze(x, 2)
print("post unsqueeze 2: ", x1, "shape: ", x1.shape)
x1 = torch.unsqueeze(x, 3)
print("post unsqueeze 3: ", x1, "shape: ", x1.shape)

x = torch.zeros(5000, 250)
x1 = x.unsqueeze(0)
x2 = x1.transpose(0, 1)
print(x.shape, x1.shape, x2.shape)

