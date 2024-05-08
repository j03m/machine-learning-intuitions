import torch
import matplotlib.pyplot as plt

time = torch.arange(0, 400, 0.1)
p1 = torch.sin(time)
p2 = torch.sin(time * 0.05)
p3 = torch.sin(time * 0.12)
linear_trend = 0.01 * time
amp = p1 + p2 + p3 + linear_trend

print("time:", time)
print("p1:", p1)
print("p2:", p2)
print("p3:", p3)
print("amp:", amp)


# Start plotting
plt.figure(figsize=(10, 8))

# Plot each sine wave
plt.subplot(4, 1, 1)  # 4 rows, 1 column, 1st subplot
plt.plot(time.numpy(), p1.numpy(), label='p1 = sin(time)')
plt.title('p1 = sin(time)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)  # 4 rows, 1 column, 2nd subplot
plt.plot(time.numpy(), p2.numpy(), label='p2 = sin(time * 0.05)')
plt.title('p2 = sin(time * 0.05)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)  # 4 rows, 1 column, 3rd subplot
plt.plot(time.numpy(), p3.numpy(), label='p3 = sin(time * 0.12)')
plt.title('p3 = sin(time * 0.12)')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 4)  # 4 rows, 1 column, 4th subplot
plt.plot(time.numpy(), amp.numpy(), label='amp = p1 + p2 + p3')
plt.title('amp = p1 + p2 + p3')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Adjust layout to not overlap graphs
plt.tight_layout()

# Show the plot
plt.show()


iw = 100
ow = 1
print("input data shape:", amp.shape)
print("seq len:", amp.size(0) - iw)

# create a sliding window of sequences each predicting the next point.
# if we have 4000 points and we want 100 point predictions, we can have 3900 of these
# thus amp.size(0) (the length) - iw
# temp is 3900 groups of 2, length iw
max_index = amp.size(0) - iw - ow + 1
temp = torch.empty(amp.size(0) - iw, 2, iw)
print("empty tensor:", temp, temp.shape)

for i in range(max_index):
    temp[i][0] = amp[i:i+iw]
    temp[i][1] = amp[i+ow, i + iw + ow]