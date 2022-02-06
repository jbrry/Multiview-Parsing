from unittest import TestCase

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model

from typing import List

from tensorflow.keras.initializers import RandomUniform


class CrossStitchBlock(Layer):
    """
    Cross-stitch block
    References
    ----------
    [1] Cross-stitch Networks for Multi-task Learning, (2017)
    Ishan Misra et al
    """

    def build(self, batch_input_shape: List[tf.TensorShape]) -> None:

        stitch_shape = len(batch_input_shape)
        print("\n\n")
        print(f"batch input shape {batch_input_shape}")
        # batch input shape [TensorShape([2, 3]), TensorShape([2, 3]), TensorShape([2, 3]), TensorShape([2, 3])]
        print(f"stitch shape {stitch_shape}")
        # stitch shape 4

        # initialize using random uniform distribution as suggested in ection 5.1
        self.cross_stitch_kernel = self.add_weight(
            shape=(stitch_shape, stitch_shape),
            # initializer=RandomUniform(0., 1.),
            initializer=RandomUniform(0, 10),
            dtype=tf.int64,
            trainable=True,
            name="cross_stitch_kernel",
        )

        print()
        print(f"the cross stitch kernel: \n {self.cross_stitch_kernel}")
        print()
        # the cross stitch kernel:
        # <tf.Variable 'cross_stitch_block/cross_stitch_kernel:0' shape=(4, 4) dtype=float32, numpy=
        # array([[0.98532534, 0.4047928 , 0.6820288 , 0.88104355],
        #     [0.08658648, 0.6606133 , 0.01084447, 0.7565317 ],
        #     [0.11862683, 0.5820112 , 0.44571018, 0.5615753 ],
        #     [0.16160917, 0.66141176, 0.9784386 , 0.5447322 ]], dtype=float32)>

        # the cross stitch kernel:
        #  <tf.Variable 'cross_stitch_block/cross_stitch_kernel:0' shape=(3, 3) dtype=int64, numpy=
        # array([[4, 9, 9],
        #        [4, 3, 2],
        #        [7, 0, 7]])>

        # normalize, so that each row will be convex linear combination,
        # here we follow recommendation in paper ( see section 5.1 )
        # normalizer = tf.reduce_sum(self.cross_stitch_kernel,
        #                           keepdims=True,
        #                           axis=0)

        # self.cross_stitch_kernel.assign(self.cross_stitch_kernel / normalizer)

        # print(f"the cross stitch kernel after normalization: \n {self.cross_stitch_kernel}")
        # print()
        # the cross stitch kernel after normalization:
        #  <tf.Variable 'cross_stitch_block/cross_stitch_kernel:0' shape=(4, 4) dtype=float32, numpy=
        # array([[0.72871125, 0.17532384, 0.32216424, 0.32109374],
        #        [0.06403625, 0.28612483, 0.00512251, 0.27571577],
        #        [0.08773215, 0.25208068, 0.21053639, 0.20466447],
        #        [0.11952034, 0.28647065, 0.46217686, 0.19852605]], dtype=float32)>

    def call(self, inputs):
        """
        Forward pass through cross-stitch block
        Parameters
        ----------
        inputs: np.array or tf.Tensor
            List of task specific tensors
        """
        # vectorized cross-stitch unit operation

        print("----- The call function-----\n")
        # print(f"inputs: {inputs}, {tf.shape(inputs)}")

        x = tf.concat([tf.expand_dims(e, axis=-1) for e in inputs], axis=-1)
        print(f"x, the concatenated inputs: {x}, shape x: {tf.shape(x)}")
        # x, the concatenated inputs: [[[1 5 9]
        #   [7 9 1]
        #   [3 0 4]]

        #  [[6 7 2]
        #   [8 5 2]
        #   [9 2 9]]], shape x: [2 3 3]

        stitched_output = tf.matmul(x, self.cross_stitch_kernel)
        print(
            f"stitched_output with shape: {tf.shape(stitched_output)}:\n{stitched_output}"
        )
        # stitched_output with shape: [2 3 3]:
        # [[[ 87  24  82]
        #   [ 71  90  88]
        #   [ 40  27  55]]

        #  [[ 66  75  82]
        #   [ 66  87  96]
        #   [107  87 148]]]

        # split result into tensors corresponding to specific tasks and return
        # Note on implementation: applying tf.squeeze(*) on tensor of shape (None,x,1)
        # produces tensor of shape unknown, so we just extract 0-th element on last axis
        # through gather function

        print("**********")
        for e in tf.split(stitched_output, len(inputs), axis=-1):
            print(e)
        outputs = [
            tf.gather(e, 0, axis=-1)
            for e in tf.split(stitched_output, len(inputs), axis=-1)
        ]
        print(f"outputs with shape: {tf.shape(outputs)}:\n{outputs}")
        # outputs with shape: [3 2 3]:
        # [<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
        # array([[ 87,  71,  40],
        #        [ 66,  66, 107]])>, <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
        # array([[24, 90, 27],
        #        [75, 87, 87]])>, <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
        # array([[ 82,  88,  55],
        #        [ 82,  96, 148]])>]

        return outputs


print("~~~ RUNNING TEST ~~~")
# test_input = [tf.ones([2, 3], dtype=tf.float32) for _ in range(4)]
test_input = [
    tf.random.uniform([2, 3], minval=0, maxval=10, dtype=tf.int64) for _ in range(3)
]

cross_stitch = CrossStitchBlock()
output = cross_stitch(test_input)

# col_sum = tf.reduce_sum(cross_stitch.cross_stitch_kernel, 0)
# expected_col_sum = tf.ones(4, dtype=tf.float32)
# abs_diff = tf.abs(col_sum - expected_col_sum)

print("Outside function\n")
# print((tf.reduce_sum(abs_diff) < 1e-4))
# check shape
# for i in range(4):
#    print("-->", (tf.reduce_all(output[i].shape == tf.TensorShape([2, 3]))))
