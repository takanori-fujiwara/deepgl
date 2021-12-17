import math

import graph_tool.all as gt
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity


class FeatDefUtil():
    def __init__(self):
        None

    @classmethod
    def to_base_feat(cls, feat_def):
        return feat_def.split('-')[-1]


class NeighborOp():
    def __init__(self):
        None

    @classmethod
    def out_nbr(cls, g, v):
        return g.get_out_neighbors(v)

    @classmethod
    def in_nbr(cls, g, v):
        return g.get_in_neighbors(v)

    @classmethod
    def all_nbr(cls, g, v):
        return np.unique(
            np.concatenate((g.get_out_neighbors(v), g.get_in_neighbors(v)),
                           axis=None))


class RelFeatOp():
    def __init__(self):
        None

    @classmethod
    def mean(cls, S, x, na_fill=0.0):
        result = 0.0
        for v in S:
            result += x[v]

        if len(S) == 0:
            result = na_fill
        else:
            result /= len(S)

        return result

    @classmethod
    def sum(cls, S, x):
        result = 0.0
        for v in S:
            result += x[v]

        return result

    @classmethod
    def maximum(cls, S, x, init=0.0):
        result = init
        for v in S:
            result = max(result, x[v])

        return result

    @classmethod
    def hadamard(cls, S, x, init=1.0):
        result = init
        for v in S:
            result *= x[v]

        return result

    @classmethod
    def lp_norm(cls, S, x, p=1, init=0.0):
        if p == 0:
            print("p must not be = 0")

        result = init
        for v in S:
            result += x[v]**p

        return result**(1 / p)

    @classmethod
    def rbf(cls, S, x, init=0.0, na_fill=0.0):
        result = init

        mean = 0.0
        sq_mean = 0.0
        for v in S:
            sq = x[v] * x[v]
            result += sq
            mean += x[v]
            sq_mean += sq
        if len(S) == 0:
            result = na_fill
        else:
            mean /= len(S)
            sq_mean /= len(S)
            var = sq_mean - mean**2
            if var == 0:
                result = na_fill
            else:
                try:
                    result = math.exp(-1 * result / var)
                except OverflowError:
                    result = 0.0

        return result


class Processing():
    def __init__(self):
        None

    @classmethod
    def log_binning(cls, X, alpha=0.5):
        if alpha > 1.0 or alpha < 0.0:
            print('alpha must between 0.0 and 1.0')

        n, d = X.shape
        ranks = rankdata(X, method='average', axis=0)

        bin_start = 0
        bin_width = math.ceil(alpha * n)
        bin_val = 0

        while bin_start <= n:
            bin_end = bin_start + bin_width

            X[(ranks >= bin_start) * (ranks < bin_end)] = bin_val

            bin_start = bin_end
            bin_width = math.ceil(alpha * bin_width)
            bin_val += 1

        return X

    @classmethod
    def feat_diffusion(cls, X, g=None, D_inv=None, A=None, iter=10):
        if iter != 0:
            if g is None and D is None and A is None:
                print('input at least either g or D & A')
                return None

            if A is None:
                A = gt.adjacency(g)
            if D_inv is None:
                # TODO: maybe need to change here when using undirected graph
                D_inv = np.diag(1.0 / g.get_in_degrees(g.get_vertices()))

            for i in range(iter):
                X = D_inv.dot(A.dot(X))

    @classmethod
    def prune_feats(cls,
                    X,
                    feat_defs,
                    lambda_value=0.9,
                    measure='cosine_similarity'):
        n, d = X.shape
        n_last_feat_defs = len(feat_defs[-1])

        ug = gt.Graph(directed=False)
        [ug.add_vertex() for i in range(d)]

        ug.edge_properties['weight'] = ug.new_edge_property("double")
        sim_mat = eval(measure + '(X.transpose())')

        for i in range(d - n_last_feat_defs, d):
            for j in range(d - n_last_feat_defs):
                if sim_mat[i, j] > lambda_value:
                    e = ug.add_edge(i, j)
                    ug.edge_properties['weight'][e] = sim_mat[i, j]
        comp_labels, _ = gt.label_components(ug)
        uniq_comp_labels = np.unique(comp_labels.a)

        repr_feat_defs = []
        remove_X_cols = []
        for comp_label in uniq_comp_labels:
            comp = np.where(comp_labels.a == comp_label)[0]
            # only take last layer's ones
            comp = comp[comp >= d - n_last_feat_defs]
            if len(comp) > 0:
                # only take first one as a representative feature
                repr_feat_defs.append(feat_defs[-1][comp[0] -
                                                    (d - n_last_feat_defs)])
                remove_X_cols += list(comp[1:])

        # note: repr_feat_defs might have different order from original
        # so, we need to handle this way (but probably can be simplified)
        remove_feat_idices = []
        for i in range(len(feat_defs[-1])):
            if not feat_defs[-1][i] in repr_feat_defs:
                remove_feat_idices.append(i)
        for index in sorted(remove_feat_idices, reverse=True):
            del feat_defs[-1][index]

        X = np.delete(X, remove_X_cols, axis=1)

        return X, feat_defs
