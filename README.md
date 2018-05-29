# Minimal-RNN-TensorFlow
This is the TensorFlow implementation of the paper: [MinimalRNN: Toward More Interpretable and Trainable Recurrent Neural Networks](https://arxiv.org/abs/1711.06788) by Minmin Chen in NIPS 2017.

![](https://github.com/wizyoung/Minimal-RNN-TensorFlow/blob/master/rnn_img.png)

### Usage

The usage is quite simple as the API of the Minimal RNN layer is totally the same with other RNN layers (like LSTM, GRU): Just `from rnn_cell import MinimalRNNCell` and use the standard TensorFlow RNN layer API.

An example code (Multi RNN example):

```python
import tensorflow as tf
from rnn_cell import MinimalRNNCell

# input_shape: [batch_size, seq_length, feat_dim]
input = tf.placeholder(tf.float32, [160, 100, 1024], name='inputs')

def get_rnn_cell():
    return MinimalRNNCell(num_units=128, kernel_initializer=tf.orthogonal_initializer())

multi_rnn_cell_video = tf.contrib.rnn.MultiRNNCell([get_rnn_cell() for _ in range(2)], state_is_tuple=True)
initial_state = multi_rnn_cell_video.zero_state(batch_size=160, dtype=tf.float32)

rnn_outputs, state = tf.nn.dynamic_rnn(
    cell=multi_rnn_cell_video,
    inputs=input,
    initial_state=initial_state,
    dtype=tf.float32
)

print(rnn_outputs)
print(state)
```

output:

```
Tensor("rnn/transpose_1:0", shape=(160, 100, 128), dtype=float32)
(<tf.Tensor 'rnn/while/Exit_3:0' shape=(160, 128) dtype=float32>, <tf.Tensor 'rnn/while/Exit_4:0' shape=(160, 128) dtype=float32>)
```

So the usage is totally the same with other RNN layers like GRU!

### NOTE

The RNN layer cells (including LSTM, GRU) in TensorFlow are defined in [tensorflow/python/ops/rnn_cell_impl.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py). The Minimal RNN layer in this repo is inherited from the RNNCell in that file to have the consistent API. Note that the API of the RNN layer cells in TensorFlow has changed a lot after version 1.4, so I implement two versions of Minimal RNN layers corresponding to TensorFlow version <=1.4 and TensorFlow version > 1.4 for compatibility. And the version switch is performed automatically so you don't need to worry about that.

