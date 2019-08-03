import tensorflow as tf


def exact_match_metric(y_true, y_pred):

    # get indices from vectors
    y_pred_argmax = tf.argmax(y_pred, axis=-1)
    y_true_argmax = tf.argmax(y_true, axis=-1)

    # get mask of rows with no entry
    mask = tf.equal(tf.reduce_sum(y_true, axis=-1), 0)

    pred_match = tf.equal(y_pred_argmax, y_true_argmax)

    # if no label in y_true, then actual match doesn't matter --> equal=True
    pred_match_fixed = tf.where(
        mask, tf.ones_like(pred_match, dtype=tf.bool), pred_match
    )

    exact_match = tf.reduce_min(tf.cast(pred_match_fixed, tf.float32), axis=[1])
    return tf.reduce_mean(exact_match)


def exact_match_metric_index(y_true, y_pred):

    # get mask of rows with no entry
    mask = tf.equal(y_true, 0)

    pred_match = tf.equal(y_pred, y_true)

    # if no label in y_true, then actual match doesn't matter --> equal=True
    pred_match_fixed = tf.where(
        mask, tf.ones_like(pred_match, dtype=tf.bool), pred_match
    )

    exact_match = tf.reduce_min(tf.cast(pred_match_fixed, tf.float32), axis=[1])
    return tf.reduce_mean(exact_match)
