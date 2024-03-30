import numpy as np

# Function to compute cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    # np.log takes the log of both probabilities (cat and not cat) because they
    # are logs, the closer to 0 the more negative the value
    # y_true, being binary 1,0 or 0,1 effectively removes one of these values.
    # In the case where we're wrong - we are a cat 1,0 * we think we're NOT a cat 0.1, 0.9
    # we end up with the very high negative being times 1 and the high probability being wiped out by the 0
    # np.sum then sums up the two values of the array which is effectively the non-zeroed value, leaving a very high
    # negative number, significantly punished due to use of log
    # we convert that to positive from negative for a high loss.
    return -np.sum(y_true * np.log(y_pred))

# True labels (One-Hot Encoded)
# "cat" is [1, 0], "not cat" is [0, 1]
y_true_cat = np.array([1, 0])
y_true_not_cat = np.array([0, 1])

# Model's predictions
# Example 1: Model is confident it's a cat
y_pred_confident_cat = np.array([0.9, 0.1])
# Example 2: Model is confidently wrong (thinks it's not a cat)
y_pred_confident_not_cat = np.array([0.1, 0.9])

# Calculating the loss
loss_confident_cat = cross_entropy_loss(y_true_cat, y_pred_confident_cat)
loss_confident_not_cat = cross_entropy_loss(y_true_cat, y_pred_confident_not_cat)

print(loss_confident_cat, loss_confident_not_cat)

