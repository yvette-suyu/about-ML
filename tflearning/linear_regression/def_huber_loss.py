"""
huber loss:
L_sigma(y,f(x)) = 1/2(y-f(x))^2    for|y-f(x)|<=sigma,
L_sigma(y,f(x)) = sigma|y-f(x)|-1/2sigma^2    otherwise.
"""


def huber_loss(labels, predictions, delta = 1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.select(condition, small_res, large_res)
