"""Supporting functions for arbitrary order Factorization Machines."""

import math
import numpy as np
import tensorflow as tf
import itertools
from scipy import sparse
from itertools import combinations_with_replacement, takewhile, count
from collections import defaultdict


def get_shorter_decompositions(basic_decomposition):
    """Returns all arrays simpler than basic_decomposition.

    Returns all arrays that can be constructed from basic_decomposition
    via joining (summing) its elements.

    Parameters
    ----------
    basic_decomposition : list or np.array
        The array from which to build subsequent ones.

    Returns
    -------
    decompositions : list of tuples
        All possible arrays that can be constructed from basic_decomposition.
    counts : np.array
        counts[i] equals to the number of ways to build decompositions[i] from
        basic_decomposition.

    Example
    -------
    decompositions, counts = get_shorter_decompositions([1, 2, 3])
        decompositions == [(1, 5), (2, 4), (3, 3), (6,)]
        counts == [ 2.,  1.,  1.,  2.]
    """
    order = int(np.sum(basic_decomposition))
    decompositions = []
    variations = defaultdict(lambda: [])
    for curr_len in range(1, len(basic_decomposition)):
        for sum_rule in combinations_with_replacement(range(curr_len), order):
            sum_rule = np.array(sum_rule)
            curr_pows = np.array([np.sum(sum_rule == i) for i in range(curr_len)])
            curr_pows = curr_pows[curr_pows != 0]
            sorted_pow = tuple(np.sort(curr_pows))
            variations[sorted_pow].append(tuple(curr_pows))
            decompositions.append(sorted_pow)
    if len(decompositions) > 1:
        decompositions = np.unique(decompositions)
        counts = np.zeros(decompositions.shape[0])
        for i, dec in enumerate(decompositions):
            counts[i] = len(np.unique(variations[dec]))
    else:
        counts = np.ones(1)
    return decompositions, counts

def sort_topologically(children_by_node, node_list):
    """Topological sort of a graph.

    Parameters
    ----------
    children_by_node : dict
        Children for any node.
    node_list : list
        All nodes (some nodes may not have children and thus a separate
        parameter is needed).

    Returns
    -------
    list, nodes in the topological order
    """
    levels_by_node = {}
    nodes_by_level = defaultdict(set)

    def walk_depth_first(node):
        if node in levels_by_node:
            return levels_by_node[node]
        children = children_by_node[node]
        level = 0 if not children else (1 + max(walk_depth_first(lname) for lname, _ in children))
        levels_by_node[node] = level
        nodes_by_level[level].add(node)
        return level

    for node in node_list:
        walk_depth_first(node)

    nodes_by_level = list(takewhile(lambda x: x != [],
                                    (list(nodes_by_level[i]) for i in count())))
    return list(itertools.chain.from_iterable(nodes_by_level))

def initial_coefficient(decomposition):
    """Compute initial coefficient of the decomposition."""
    order = np.sum(decomposition)
    coef = math.factorial(order)
    coef /= np.prod([math.factorial(x) for x in decomposition])
    _, counts = np.unique(decomposition, return_counts=True)
    coef /= np.prod([math.factorial(c) for c in counts])
    return coef

def powers_and_coefs(order):
    """For a `order`-way FM returns the powers and their coefficients needed to
    compute model equation efficiently
    """
    decompositions, _ = get_shorter_decompositions(np.ones(order))
    graph = defaultdict(lambda: list())
    graph_reversed = defaultdict(lambda: list())
    for dec in decompositions:
        parents, weights = get_shorter_decompositions(dec)
        for i in range(len(parents)):
            graph[parents[i]].append((dec, weights[i]))
            graph_reversed[dec].append((parents[i], weights[i]))

    topo_order = sort_topologically(graph, decompositions)

    final_coefs = defaultdict(lambda: 0)
    for node in topo_order:
        final_coefs[node] += initial_coefficient(node)
        for p, w in graph_reversed[node]:
            final_coefs[p] -= w * final_coefs[node]
    powers_and_coefs_list = []
    # for dec, c in final_coefs.iteritems():
    for dec, c in final_coefs.items():
        in_pows, out_pows = np.unique(dec, return_counts=True)
        powers_and_coefs_list.append((in_pows, out_pows, c))

    return powers_and_coefs_list


def matmul_wrapper(A, B, optype):
    """Wrapper for handling sparse and dense versions of `tf.matmul` operation.

    Parameters
    ----------
    A : tf.Tensor
    B : tf.Tensor
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope('matmul_wrapper') as scope:
        if optype == 'dense':
            return tf.matmul(A, B)
        elif optype == 'sparse':
            return tf.sparse_tensor_dense_matmul(A, B)
        else:
            raise NameError('Unknown input type in matmul_wrapper')


def pow_wrapper(X, p, optype):
    """Wrapper for handling sparse and dense versions of `tf.pow` operation.

    Parameters
    ----------
    X : tf.Tensor
    p : int
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor
    """
    with tf.name_scope('pow_wrapper') as scope:
        if optype == 'dense':
            return tf.pow(X, p)
        elif optype == 'sparse':
            return tf.SparseTensor(X.indices, tf.pow(X.values, p), X.dense_shape)
        else:
            raise NameError('Unknown input type in pow_wrapper')


def count_nonzero_wrapper(X, optype):
    """Wrapper for handling sparse and dense versions of `tf.count_nonzero`.

    Parameters
    ----------
    X : tf.Tensor (N, K)
    optype : str, {'dense', 'sparse'}

    Returns
    -------
    tf.Tensor (1,K)
    """
    with tf.name_scope('count_nonzero_wrapper') as scope:
        if optype == 'dense':
            return tf.count_nonzero(X, axis=0, keep_dims=True)
        elif optype == 'sparse':
            indicator_X = tf.SparseTensor(X.indices, tf.ones_like(X.values), X.dense_shape)
            return tf.sparse_reduce_sum(indicator_X, axis=0, keep_dims=True)
        else:
            raise NameError('Unknown input type in count_nonzero_wrapper')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Predefined loss functions
# Should take 2 tf.Ops: outputs, targets and should return tf.Op of element-wise losses
# Be careful about dimensionality -- maybe tf.transpose(outputs) is needed

def loss_logistic(outputs, y):
    margins = -y * tf.transpose(outputs)
    raw_loss = tf.log(tf.add(1.0, tf.exp(margins)))
    return tf.minimum(raw_loss, 100, name='truncated_log_loss')


def loss_mse(outputs, y):
    return tf.pow(y -  tf.transpose(outputs), 2, name='mse_loss')


def loss_ranknet(outputs, y):
    y_true = tf.cast(y, tf.int32)

    # Sigma is a shape parameter for the sigmoid function
    sigma = tf.constant(1.0, dtype='float32')

    S = tf.sign((tf.expand_dims(y_true, 0) - tf.expand_dims(y_true, -1))) + tf.constant(1)
    s_diff = tf.expand_dims(outputs, 0) - tf.expand_dims(outputs, -1)

    parts = tf.dynamic_partition(s_diff, S, 3)

    # Loss function for records that can be compared
    loss_neg = tf.reduce_mean(
        tf.log(tf.constant(1.0) + tf.exp(sigma * parts[0]))
    )
    loss_pos = tf.reduce_mean(
        tf.log(tf.constant(1.0) + tf.exp(tf.constant(-1.0) * sigma * parts[2]))
    )

    # Loss functions for ties. We use the fact that the linear term
    # cancels out over all pairs.
    loss_tie = tf.reduce_mean(
        tf.constant(0.5) * sigma * parts[1] +
        tf.log(tf.constant(1.0) + tf.exp(tf.constant(-1.0) * sigma * parts[1]))
    )

    return loss_neg + loss_pos


def ranknet_batcher(X_, y_=None, w_=None):
    """Split data into batches. Each batch corresponds to a single
    query.

    Parameters
    ----------
    X_ : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features + 1)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features. The first column should contain
        the group indicators.

    y_ : np.array or None, shape (n_samples,)
        Target vector relative to X.

    w_ : np.array or None, shape (n_samples,)
        Vector of sample weights.

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (?, n_features)
        Same type as input with the group indicators removed.

    ret_y : np.array or None, shape (?,)

    ret_w : np.array or None, shape (?,)
    """
    if sparse.issparse(X_):
        groups = X_[:,0].toarray()[:,0]
    else:
        groups = X_[:,0]

    X_feat = X_[:, 1:]

    X_blocks = [X_feat[groups == k] for k in np.unique(groups)]

    if y_ is not None:
        y_blocks = [y_[groups == k] for k in np.unique(groups)]

    if w_ is not None:
        w_blocks = [w_[groups == k] for k in np.unique(groups)]

    for i in np.random.permutation(len(X_blocks)):
        ret_x = X_blocks[i]
        ret_y = None
        ret_w = None
        if y_ is not None:
            ret_y = y_blocks[i]
        if w_ is not None:
            ret_w = w_blocks[i]
        yield (ret_x, ret_y, ret_w)
    

