from rllab.spaces.box import Box as TheanoBox
import tensorflow as tf


class Box(TheanoBox):
    def new_tensor_variable(self, name, extra_dims, flatten=True, n_additional_inputs=0):
        if flatten:
            return tf.placeholder(tf.float32, shape=[None] * extra_dims + [self.flat_dim + n_additional_inputs], name=name)
        return tf.placeholder(tf.float32, shape=[None] * extra_dims + list(self.shape), name=name)

    @property
    def dtype(self):
        return tf.float32
