# coding: utf-8

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.rnn_cell import RNNCell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

__all__ = ['MinimalRNNCell']

tf_version = tf.__version__

def low_tf_version(detect_version, target_version='1.4.0'):
    detect_version_lst = [int(i) for i in detect_version.split('.')]
    target_version_lst = [int(i) for i in target_version.split('.')]
    if ((detect_version_lst[0] > target_version_lst[0]) or
            ((detect_version_lst[0] == target_version_lst[0]) and (detect_version_lst[1] > target_version_lst[1]))):
        print('**NOTE**: Your tensorflow version is {}. Now using MinimalRNNCell_new version.'.format(detect_version))
        return False
    else:
        print('**NOTE**: Your tensorflow version is {}. Now using MinimalRNNCell_old version.'.format(detect_version))
        return True

if low_tf_version(tf_version):
    from tensorflow.python.ops.rnn_cell_impl import _Linear

    class MinimalRNNCell_old(RNNCell):
        """MinimalRNN.
            This implementation is based on:
            Minmin Chen,
            "MinimalRNN: Toward More Interpretable and
            Trainable Recurrent Neural Networks,"
            https://arxiv.org/abs/1711.06788.
        """

        def __init__(self,
                    num_units,
                    activation=None,
                    reuse=None,
                    kernel_initializer=None,
                    bias_initializer=None):
            super(MinimalRNNCell_old, self).__init__(_reuse=reuse)
            self._num_units = num_units
            self._activation = activation or math_ops.tanh
            self._kernel_initializer = kernel_initializer
            self._bias_initializer = bias_initializer

        @property
        def state_size(self):
            return self._num_units

        @property
        def output_size(self):
            return self._num_units

        def call(self, inputs, state):
            with vs.variable_scope("map"):
                self._map_linear = _Linear(
                    [inputs],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)

            z = self._activation(self._map_linear([inputs]))
            u_inputs = array_ops.concat([z, state], 1)

            with vs.variable_scope("gate"):
                self._gate_linear = _Linear(
                    [u_inputs],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer
                    if self._bias_initializer is not None
                    else init_ops.constant_initializer(1., dtype=self.dtype),
                    kernel_initializer=self._kernel_initializer)

            u = math_ops.sigmoid(self._gate_linear([u_inputs]))

            new_state = u * state + (1. - u) * z

            return new_state, new_state
        
    MinimalRNNCell = MinimalRNNCell_old
else:
    try:
        from tensorflow.python.ops.rnn_cell_impl import _LayerRNNCell
    except Exception:
        from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell as _LayerRNNCell

    class MinimalRNNCell_new(_LayerRNNCell):
        """MinimalRNN.
        This implementation is based on:
        Minmin Chen,
        "MinimalRNN: Toward More Interpretable and
            Trainable Recurrent Neural Networks,"
        https://arxiv.org/abs/1711.06788.
        AUTHOR: wizyoung
        """

        def __init__(self,
                    num_units,
                    activation=None,
                    reuse=None,
                    kernel_initializer=None,
                    bias_initializer=None,
                    name=None):
            super(MinimalRNNCell_new, self).__init__(_reuse=reuse, name=name)

            # Inputs must be 2-dimensional.
            self.input_spec = base_layer.InputSpec(ndim=2)

            self._num_units = num_units
            self._activation = activation or math_ops.tanh
            self._kernel_initializer = kernel_initializer
            self._bias_initializer = bias_initializer

        @property
        def state_size(self):
            return self._num_units

        @property
        def output_size(self):
            return self._num_units

        def build(self, inputs_shape):
            if inputs_shape[1].value is None:
                raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                                % inputs_shape)

            # input_depth: input dim
            input_depth = inputs_shape[1].value
            self._map_kernel = self.add_variable(
                "map/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth, self._num_units],
                initializer=self._kernel_initializer)
            self._map_bias = self.add_variable(
                "map/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(
                    self._bias_initializer
                    if self._bias_initializer is not None
                    else init_ops.constant_initializer(0., dtype=self.dtype)))
            self._gate_kernel = self.add_variable(
                "gate/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[2 * self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._gate_bias = self.add_variable(
                "gate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(
                    self._bias_initializer
                    if self._bias_initializer is not None
                    else init_ops.constant_initializer(1., dtype=self.dtype)))

            self.built = True

        def call(self, inputs, state):

            z = self._activation(
                nn_ops.bias_add(math_ops.matmul(inputs, self._map_kernel), self._map_bias))

            u_inputs = array_ops.concat([z, state], 1)
            u = math_ops.sigmoid(
                nn_ops.bias_add(math_ops.matmul(
                    u_inputs, self._gate_kernel), self._gate_bias)
            )

            new_state = u * state + (1. - u) * z

            return new_state, new_state


    MinimalRNNCell = MinimalRNNCell_new
