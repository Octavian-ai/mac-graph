
from tensorflow.python.training import optimizer
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops

class PercentDeltaOptimizer(optimizer.Optimizer):

  def __init__(self,  target=0.2, use_locking=False, name="PercentDelta"):
    super(PercentDeltaOptimizer, self).__init__(use_locking, name)
    self._target = target
    
  def _prepare(self):
    self._target_t = ops.convert_to_tensor(self._target, name="target")
	
  def _apply_dense(self, grad, var):
    return self._apply_pd(grad, var)

  def _apply_grad_descent(self, grad, var):
    lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
    var_update = state_ops.assign_sub(var, grad * lr_t)
    return control_flow_ops.group(*[var_update])
  
  # Thanks to https://github.com/google/asymproj_edge_dnn/blob/master/edge_nn.py
  def _apply_pd(self, grad, var):
    
    def PlusEpsilon(x, eps=1e-6):
      """Element-wise add `eps` to `x` without changing sign of `x`."""
      return x + (tf.sign(x) * eps)

    target_t = math_ops.cast(self._target_t, var.dtype.base_dtype)
    mean_percent_grad = tf.reduce_mean(tf.abs(tf.div(grad, PlusEpsilon(var))))
    lr_t = tf.div(target_t, (mean_percent_grad + 1e-5))

    var_update = state_ops.assign_sub(var, grad * lr_t)
    return var_update

  def _apply_sparse(self, grad, var):
    raise NotImplementedError("Sparse gradient updates are not supported.")