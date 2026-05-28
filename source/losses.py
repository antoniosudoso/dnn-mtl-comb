import tensorflow as tf
import numpy as np


def comb_loss_wrapper_old(f_matrix):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    """

    def comb_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """
        se_num = tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=1)
        f_average = tf.math.reduce_mean(f_matrix, axis=1)
        se_den = tf.math.reduce_sum(tf.math.square(f_average - y_true), axis=1)
        comb = tf.math.reduce_mean(se_num / se_den)
        return comb

    return comb_loss


def comb_loss_wrapper(f_matrix):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    """

    def comb_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """
        se_num = tf.math.reduce_sum(tf.abs(y_pred - y_true), axis=1)
        f_average = tf.math.reduce_mean(f_matrix, axis=1)
        se_den = tf.math.reduce_sum(tf.abs(f_average - y_true), axis=1)
        comb = tf.math.reduce_mean(se_num / se_den)
        return comb

    return comb_loss


def comb_loss_wrapper_no_scale(f_matrix):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    """

    def comb_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """
        comb = tf.math.reduce_mean(tf.abs(y_pred - y_true))
        return comb

    return comb_loss


def cls_loss_wrapper(y_cls_true, y_cls_pred):
    """
    :param y_cls_true: tf.Tensor (n_samples, f_methods)
    :param y_cls_pred: tf.Tensor (n_samples, f_methods)
    """

    def cls_loss(y_true, y_pred):
        bce = tf.keras.losses.BinaryCrossentropy()
        cls = bce(y_cls_true, y_cls_pred)
        return cls

    return cls_loss


# add weights in the BCE loss function
def cls_loss_wrapper_class_weights(y_cls_true, y_cls_pred, weights):
    """
    :param weights:
    :param y_cls_true: tf.Tensor (n_samples, f_methods)
    :param y_cls_pred: tf.Tensor (n_samples, f_methods)
    """

    weights_tf = tf.convert_to_tensor(weights, dtype=np.float32)

    def cls_loss(y_true, y_pred):
        #bce = tf.keras.losses.BinaryCrossentropy()
        #cls = bce(y_cls_true, y_cls_pred)
        out = -(y_cls_true * tf.math.log(y_cls_pred)*weights_tf[1] + (1.0 - y_cls_true) * tf.math.log(1.0 - y_cls_pred)*weights_tf[0])
        return tf.reduce_mean(out)

    return cls_loss


def ort_loss_wrapper(x_reg_feat, x_cls_feat):
    """
    :param x_reg_feat: extracted features from the regression branch
    :param x_cls_feat: extracted features from the classification branch
    """

    def ort_loss(y_true, y_pred):
        ort_mul = tf.linalg.matmul(x_reg_feat, x_cls_feat, transpose_a=False, transpose_b=True)
        # ort = tf.math.square(tf.linalg.norm(ort_mul, ord='fro', axis=(0, 1)))
        ort = tf.reduce_mean(tf.math.square(ort_mul))
        return ort

    return ort_loss


def overall_loss_wrapper_old(f_matrix, lambda_hyper, y_cls_true, y_cls_pred, mu_hyper, x_reg_feat, x_cls_feat):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    :param x_cls_feat: tf.Tensor (n_samples, n_features)
    :param x_reg_feat: tf.Tensor (n_samples, n_features)
    :param y_cls_true: tf.Tensor (n_samples, f_methods)
    :param y_cls_pred: tf.Tensor (n_samples, f_methods)
    :param mu_hyper: penalty for orthogonality
    :param lambda_hyper: penalty for classification
    """
    cls = cls_loss_wrapper(y_cls_true, y_cls_pred)
    ort = ort_loss_wrapper(x_reg_feat, x_cls_feat)

    def custom_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """
        se_num = tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=1)
        f_average = tf.math.reduce_mean(f_matrix, axis=1)
        se_den = tf.math.reduce_sum(tf.math.square(f_average - y_true), axis=1)
        comb = tf.math.reduce_mean(se_num / se_den)
        return comb + lambda_hyper * cls(0, 0) + mu_hyper * ort(0, 0)

    return custom_loss


def overall_loss_wrapper(f_matrix, lambda_hyper, y_cls_true, y_cls_pred, mu_hyper, x_reg_feat, x_cls_feat):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    :param x_cls_feat: tf.Tensor (n_samples, n_features)
    :param x_reg_feat: tf.Tensor (n_samples, n_features)
    :param y_cls_true: tf.Tensor (n_samples, f_methods)
    :param y_cls_pred: tf.Tensor (n_samples, f_methods)
    :param mu_hyper: penalty for orthogonality
    :param lambda_hyper: penalty for classification
    """
    cls = cls_loss_wrapper(y_cls_true, y_cls_pred)
    ort = ort_loss_wrapper(x_reg_feat, x_cls_feat)

    def custom_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """

        se_num = tf.math.reduce_sum(tf.abs(y_pred - y_true), axis=1)
        f_average = tf.math.reduce_mean(f_matrix, axis=1)
        se_den = tf.math.reduce_sum(tf.abs(f_average - y_true), axis=1)
        comb = tf.math.reduce_mean(se_num / se_den)
        return comb + lambda_hyper * cls(0, 0) + mu_hyper * ort(0, 0)

    return custom_loss


def overall_loss_wrapper_class_weights(f_matrix, lambda_hyper, y_cls_true, y_cls_pred, mu_hyper, x_reg_feat, x_cls_feat,
                                       weights):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    :param x_cls_feat: tf.Tensor (n_samples, n_features)
    :param x_reg_feat: tf.Tensor (n_samples, n_features)
    :param y_cls_true: tf.Tensor (n_samples, f_methods)
    :param y_cls_pred: tf.Tensor (n_samples, f_methods)
    :param mu_hyper: penalty for orthogonality
    :param lambda_hyper: penalty for classification
    """
    cls = cls_loss_wrapper_class_weights(y_cls_true, y_cls_pred, weights)
    ort = ort_loss_wrapper(x_reg_feat, x_cls_feat)

    def custom_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """

        se_num = tf.math.reduce_sum(tf.abs(y_pred - y_true), axis=1)
        f_average = tf.math.reduce_mean(f_matrix, axis=1)
        se_den = tf.math.reduce_sum(tf.abs(f_average - y_true), axis=1)
        comb = tf.math.reduce_mean(se_num / se_den)
        return comb + lambda_hyper * cls(0, 0) + mu_hyper * ort(0, 0)

    return custom_loss


def overall_loss_wrapper_no_scale(f_matrix, lambda_hyper, y_cls_true, y_cls_pred, mu_hyper, x_reg_feat, x_cls_feat):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    :param x_cls_feat: tf.Tensor (n_samples, n_features)
    :param x_reg_feat: tf.Tensor (n_samples, n_features)
    :param y_cls_true: tf.Tensor (n_samples, f_methods)
    :param y_cls_pred: tf.Tensor (n_samples, f_methods)
    :param mu_hyper: penalty for orthogonality
    :param lambda_hyper: penalty for classification
    """
    cls = cls_loss_wrapper(y_cls_true, y_cls_pred)
    ort = ort_loss_wrapper(x_reg_feat, x_cls_feat)

    def custom_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """
        comb = tf.math.reduce_mean(tf.abs(y_pred - y_true))
        return comb + lambda_hyper * cls(0, 0) + mu_hyper * ort(0, 0)

    return custom_loss

def overall_loss_wrapper_no_scale_class_weights(f_matrix, lambda_hyper, y_cls_true, y_cls_pred, mu_hyper, x_reg_feat, x_cls_feat, weights):
    """
    :param f_matrix: tf.Tensor (n_samples, f_methods, f_horizon)
    :param x_cls_feat: tf.Tensor (n_samples, n_features)
    :param x_reg_feat: tf.Tensor (n_samples, n_features)
    :param y_cls_true: tf.Tensor (n_samples, f_methods)
    :param y_cls_pred: tf.Tensor (n_samples, f_methods)
    :param mu_hyper: penalty for orthogonality
    :param lambda_hyper: penalty for classification
    """
    cls = cls_loss_wrapper_class_weights(y_cls_true, y_cls_pred, weights)
    ort = ort_loss_wrapper(x_reg_feat, x_cls_feat)

    def custom_loss(y_true, y_pred):
        """
        :param y_true: tf.Tensor (n_samples, f_horizon)
        :param y_pred: tf.Tensor (n_samples, f_horizon)
        """
        comb = tf.math.reduce_mean(tf.abs(y_pred - y_true))
        return comb + lambda_hyper * cls(0, 0) + mu_hyper * ort(0, 0)

    return custom_loss