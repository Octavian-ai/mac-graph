import tensorflow as tf

"""
From https://github.com/bparr/lars/blob/master/lars.py

Layer-wise Adaptive Rate Scaling minimizer.

When tuning, a large initial_learning_rate (e.g. 1000.0) might be needed for
good results if using the default value for lars_coefficient.

Based on Algorithm 1 in https://arxiv.org/pdf/1708.03888.pdf.

Usage Example:
  train_step = createLarsMinimizer(...)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(train_step)

This uses tf.train.get_or_create_global_step(), and increments this global step
every time the returned Operation is run.

Args:
  loss: A Tensor containing the value to minimize.
  initial_learning_rate: A float. The base LR to use (gamma_0 in paper).
  learning_rate_decay_steps: A scalar int32 or int64 Tensor or a Python number.
    See https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay
    documentation on decay_steps argument.
  momentum: A Tensor or a float. The momentum.
  weight_decay: A float. B in paper.
  lars_coefficient: A float. eta in paper.
  eps: A float. Used for numerical stability close to the optimum.
  use_decay: A boolean.
  var_list: List of tensorflow Variables. Each one will be optimized
    individually (e.g. have its own momentum). If None, then defaults to all
    trainable variables.

Returns:
  A tuple of (an Operation that minimizes using LARS,
              debug function that returns dict of values given a Session).
"""
def createLarsMinimizer(loss, initial_learning_rate, learning_rate_decay_steps,
                        momentum=0.0, weight_decay=0.0, lars_coefficient=0.0001,
                        eps=1e-9, use_decay=True, var_list=None):
  if var_list is None:
    var_list = tf.trainable_variables()
  grads = tf.gradients(loss, var_list)

  global_step = None
  learning_rate = initial_learning_rate
  if use_decay:
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.polynomial_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=learning_rate_decay_steps,
        # TODO Just use default value of 0.0001?
        end_learning_rate=0.0,
        power=2.0)

  debug_vars = {
      'global_step': global_step,
      'learning_rate': learning_rate,
  }

  apply_gradients_opts = []
  for i, (var, grad) in enumerate(zip(var_list, grads)):
    if grad is None:
      raise Exception('No gradient for Variable. Either add trainable=False ' +
                      'to its constructor, or explicitly list which vars to ' +
                      'include in the createLarsMinimizers var_list arg.', var)

    var_norm = tf.norm(var, ord=2)
    # Note: eps is used for numerical stability close to the optimum
    local_lr = lars_coefficient * var_norm / (
        eps + tf.norm(grad, ord=2) + weight_decay * var_norm)

    # When var is almost zero can't use lars.
    # TODO Figure out a better way to handle this weights close-to-zero case.
    local_lr = tf.where(var_norm < 1e-2, lars_coefficient, local_lr)
    new_grad = local_lr * (grad + weight_decay * var)
    debug_vars['local_lr' + str(i)] = local_lr
    debug_vars['new_grad' + str(i)] = new_grad

    opt = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)
    apply_gradients_opts.append(opt.apply_gradients(
        [(local_lr * (grad + weight_decay * var), var)]))

  debug_fn = _createDebugFn(debug_vars)
  if global_step is None:
    return tf.group(*apply_gradients_opts), debug_fn

  # Ensure global step incremented once only after all gradients updated.
  with tf.control_dependencies(apply_gradients_opts):
    return tf.assign_add(global_step, 1).op, debug_fn


def _createDebugFn(debug_vars):
  def debug_fn(sess):
    values = {}
    for name, var in debug_vars.items():
      value = var
      if var is not None and not isinstance(var, float):
        value = sess.run(var)
      values[name] = value
    return values

  return debug_fn