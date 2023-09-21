### Given Values

- Weights \( W = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{pmatrix} \) (shape \(2 \times 3\))
- Bias \( b = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} \) (shape \(3,\))
- Input \( x = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \) (shape \(1 \times 2\))
- Gradient \( g = \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix} \) (shape \(1 \times 3\))
- Learning rate \( \alpha = 0.01 \)
- Activation function is linear, so its derivative is 1

### Steps

1. **Calculate Delta**: 
   \[
   \delta = g \times \text{derivative of activation function}
   \]
   Since the activation function is linear, its derivative is 1. Therefore, \( \delta = g \).

2. **Calculate Weights Gradient**:
   \[
   \text{weights_gradient} = x^T \cdot \delta
   \]
   \( x^T \) has shape \( (2, 1) \) and \( \delta \) has shape \( (1, 3) \). The resulting weights_gradient will have a shape \( (2, 3) \).

3. **Calculate Bias Gradient**:
   \[
   \text{bias_gradient} = \text{sum}(\delta, \text{axis}=0)
   \]
   The bias_gradient will have the same shape as the bias, which is \( (3,) \).

4. **Update Weights and Bias**:
   \[
   W_{\text{new}} = W - \alpha \times \text{weights_gradient}
   \]
   \[
   b_{\text{new}} = b - \alpha \times \text{bias_gradient}
   \]

### Manual Calculation

1. \( \delta = \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix} \)
  
2. \( \text{weights_gradient} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}^T \cdot \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix} = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.1 & 0.2 & 0.3 \end{pmatrix} \)

3. \( \text{bias_gradient} = \text{sum}(\begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix}, \text{axis}=0) = \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix} \)

4. \( W_{\text{new}} = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \end{pmatrix} - 0.01 \times \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.1 & 0.2 & 0.3 \end{pmatrix} = \begin{pmatrix} 0.099 & 0.198 & 0.297 \\ 0.399 & 0.498 & 0.597 \end{pmatrix} \)

5. \( b_{\text{new}} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix} - 0.01 \times \begin{pmatrix} 0.1 \\ 0.2 \\ 0.3 \end{pmatrix} = \begin{pmatrix} -0.001 \\ -0.002 \\ -0.003 \end{pmatrix} \)

