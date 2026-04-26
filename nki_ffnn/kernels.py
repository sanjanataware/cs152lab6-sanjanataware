import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows

    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)

    pmax = nl.tile_size.pmax  

    num_row_tiles = i_rows // pmax
    num_col_tiles = i_cols // pmax

    for i in nl.affine_range(num_row_tiles):
        for j in nl.affine_range(num_col_tiles):
            tile = nl.load(in_tensor[i*pmax:(i+1)*pmax, j*pmax:(j+1)*pmax])
            t_tile = nisa.nc_transpose(tile)
            nl.store(out_tensor[j*pmax:(j+1)*pmax, i*pmax:(i+1)*pmax], t_tile)

    return out_tensor

@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)

    pmax = nl.tile_size.pmax
    num_tiles = BATCH_SIZE // pmax

    for i in nl.affine_range(num_tiles):
        a_tile = nl.load(A[i*pmax:(i+1)*pmax, :])  
        b_tile = nl.load(b[0:1, :])                

        out = nl.add(a_tile, b_tile)  

        if act == 'relu':
            out = nl.maximum(out, 0)
        elif act == 'softmax':
            out = nl.subtract(out, max_val)       
            out = nl.exp(out)
            sum_val = nl.sum(out, axis=1)         
            out = nl.divide(out, sum_val)

        nl.store(result[i*pmax:(i+1)*pmax, :], out)

    return result

@nki.jit
def nki_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
  """
  if matmul_kernel == 'tiled':
    nki_matmul = nki_matmul_tiled_
  elif matmul_kernel == 'hoist_load':
    nki_matmul = nki_matmul_hoist_load_
  elif matmul_kernel == 'block_free_dimension':
    nki_matmul = nki_matmul_block_free_dimension_
  elif matmul_kernel == 'fully_optimized':
    nki_matmul = nki_matmul_fully_optimized_
  else:
    raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")

  BATCH_SIZE, INPUT_SIZE = X.shape
  INPUT_SIZE_, HIDDEN_SIZE = W1.shape
  HIDDEN_SIZE_, OUTPUT_SIZE = W2.shape

  X_T = nki_transpose(X)  
  H_raw = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=X.dtype, buffer=nl.hbm)
  nki_matmul(X_T, W1, H_raw)

  b1_2d = b1.reshape((1, HIDDEN_SIZE))
  H = nki_bias_add_act(H_raw, b1_2d, act='relu')

  H_T = nki_transpose(H)  
  out_raw = nl.ndarray((BATCH_SIZE, OUTPUT_SIZE), dtype=X.dtype, buffer=nl.hbm)
  nki_matmul(H_T, W2, out_raw)

  b2_2d = b2.reshape((1, OUTPUT_SIZE))
  probs = nki_bias_add_act(out_raw, b2_2d, act='softmax')

  return probs


@nki.jit
def nki_predict(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel run forward pass and predict the classes of the input tensor.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  """
  probs = # YOUR CODE HERE
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel=matmul_kernel)
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  pmax = nl.tile_size.pmax
  num_tiles = BATCH_SIZE // pmax

  for i in nl.affine_range(num_tiles):
      p_tile = nl.load(probs[i*pmax:(i+1)*pmax, :]) 
      indices = nisa.tensor_reduce(np.argmax, p_tile, axis=(1,), dtype=np.int32, negate=False)
      nl.store(predictions[i*pmax:(i+1)*pmax], indices[:, 0])

  return predictions