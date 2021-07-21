# Author: Ruben Rodriguez
# License: MIT

from functools import partial

import numpy as np

import jax.numpy as jnp
import jax


__ALL__ = ['predict_fn']


def _predict_proba(prob_a, prob_b, decision_value):
    """[summary]

    Args:
        prob_a ([type]): [description]
        prob_b ([type]): [description]
        decision_value ([type]): [description]

    Returns:
        [type]: [description]
    """

    fApB = prob_a * decision_value + prob_b
    prob = jnp.where(fApB >= 0, jnp.exp(-fApB) /
                     (1.0+jnp.exp(-fApB)), 1.0/(1+jnp.exp(fApB)))
    return 1 - prob


def _prediction_class(decision_function):
    return jnp.where(decision_function >= 0, 1, 0)


def _chain_pred_transformation(decision_function, transformation_function):
    """[summary]

    Args:
        decision_function ([type]): [description]
        transformation_function ([type]): [description]

    Returns:
        [type]: [description]
    """
    def apply(X):
        return transformation_function(decision_function(X))

    return apply


def _fast_predict_rbf_impl(alphas, svs, b, gamma, X):
    """[summary]

    Args:
        alphas ([type]): [description]
        svs ([type]): [description]
        b ([type]): [description]
        gamma ([type]): [description]
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    dist = jnp.linalg.norm(X - svs, axis=1)
    rbf = jnp.exp(gamma * (dist * dist))
    decision = jnp.sum(alphas * rbf) + b
    return decision


def _fast_predict_linear_impl(alphas, svs, b, X):
    """[summary]

    Args:
        alphas ([type]): [description]
        svs ([type]): [description]
        b ([type]): [description]
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    return jnp.sum(alphas * jnp.dot(X, svs)) + b


def _fast_predict_poly_impl(alphas, svs, b, degree, X):
    pass


def predict_fn(clf, output_type='class', jit=True, batch=True):
    """[summary]

    Args:
        clf (sklearn.svm.SVC): A SVM instance (previously trained)
        output_type (str): Options are: 
            class: 0 or 1 predictions, 
            decision_function: svm prediction without the sign function,
            proba: probability using plat scaling (The SVM must be trained using probability=True).

    Raises:
        NotImplemented: [description]
        ValueError: [description]
        NotImplemented: [description]

    Returns:
        A function that takes data as input and returns the predictions (see output_type).
    """
    alphas = jnp.array(np.ravel(clf.dual_coef_))
    svs = jnp.array(clf.support_vectors_)
    b = clf.intercept_

    if clf.kernel == 'rbf':
        gamma = -clf._gamma
        predict = partial(_fast_predict_rbf_impl, alphas, svs, b, gamma)
    elif clf.kernel == 'lineal':
        predict = partial(_fast_predict_linear_impl, alphas, svs, b)
    elif clf.kernel == 'poly':
        degree = clf.degree
        predict = partial(_fast_predict_poly_impl, alphas, svs, b, degree)
    else:
        raise NotImplemented(
            f"Kernel type {clf.kernel} not supported. Only rbf, lineal and poly are currently implemented")

    if output_type == 'proba':
        if not clf.probability:
            raise ValueError(f"SVM was not trained using probability=True")

        prob_a = np.ravel(clf.probA_)[0]
        prob_b = np.ravel(clf.probB_)[0]

        predict = _chain_pred_transformation(
            predict, partial(_predict_proba, prob_a, prob_b))
    elif output_type == 'decision_function':
        pass
    elif output_type == 'class':
        predict = _chain_pred_transformation(predict, _prediction_class)
    else:
        raise NotImplemented(
            f"Output type {output_type} not supported. Only proba, decision_function and class are currently implemented")

    predict_fn = _chain_pred_transformation(predict, jnp.ravel)

    if batch:
        predict_fn = jax.vmap(predict_fn)

    if jit:
        predict_fn = jax.jit(predict_fn)

    return predict_fn
