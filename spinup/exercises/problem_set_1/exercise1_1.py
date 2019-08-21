import tensorflow as tf
import numpy as np
from tensorflow import square, exp, divide, log, scalar_mul, to_float, cast
from tensorflow.python import reduce_sum

"""

Exercise 1.1: Diagonal Gaussian Likelihood

Write a function which takes in Tensorflow symbols for the means and 
log stds of a batch of diagonal Gaussian distributions, along with a 
Tensorflow placeholder for (previously-generated) samples from those 
distributions, and returns a Tensorflow symbol for computing the log 
likelihoods of those samples.

"""


def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    #######################
    #                     #
    #   YOUR CODE HERE    #
    #                     #
    #######################
    global sum
    global factor1
    global result
    shape = x.get_shape().as_list()[-1]
    tf.print(shape)
    sum = divide(square(x - mu), square(exp(log_std))) + scalar_mul(2, log_std)
    tf.print(sum)
    factor1 = reduce_sum(sum, 1)
    factor2 = cast(shape * np.log(2.0 * np.pi), dtype=tf.float32)
    result = (factor1 + factor2) / -2.0
    return result
    # return tf.constant(0)


if __name__ == '__main__':
    """
    Run this file to verify your solution.
    """
    from spinup.exercises.problem_set_1_solutions import exercise1_1_soln
    from spinup.exercises.common import print_result

    sess = tf.Session()

    dim = 10
    x = tf.placeholder(tf.float32, shape=(None, dim))
    mu = tf.placeholder(tf.float32, shape=(None, dim))
    log_std = tf.placeholder(tf.float32, shape=(dim,))

    your_gaussian_likelihood = gaussian_likelihood(x, mu, log_std)
    true_gaussian_likelihood = exercise1_1_soln.gaussian_likelihood(x, mu, log_std)

    batch_size = 32
    feed_dict = {x: np.random.rand(batch_size, dim),
                 mu: np.random.rand(batch_size, dim),
                 log_std: np.random.rand(dim)}

    test, test2, test3, your_result, true_result = sess.run([sum, factor1, result, your_gaussian_likelihood, true_gaussian_likelihood],
            feed_dict=feed_dict)
    print(test)
    print(test2)
    print(test3)
    correct = np.allclose(your_result, true_result)
    print_result(correct)
