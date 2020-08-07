import graph_tool.all as gt
import numpy as np

from deepgl_utils import NeighborOp, Processing, RelFeatOp

# TODO: support local graphlet
# TODO: use more matrix calc (right now following the psuedocode in the paper)
# TODO: use more sparse matrix to save memory usage
# TODO: add arg to select eracing info of the pruned features from graph object
# TODO: do sanity check for input of base_feats
# TODO: support summation and multiplication of relational functions later


class DeepGL():
    def __init__(
            self,
            base_feat_defs=[
                'in_degree', 'out_degree', 'total_degree', 'pagerank',
                'kcore_decomposition'
            ],
            rel_feat_ops=['mean', 'sum', 'maximum'],
            nbr_types=['in', 'out', 'all'],
            ego_dist=3,
            lambda_value=0.9,  # bigger lambda makes more features
            diffusion_iter=0,
            transform_method='log_binning',
            log_binning_alpha=0.5):
        # Note: removed 'hadamard' from rel_feat_ops (it's problemetic now)
        self.base_feat_defs = base_feat_defs
        self.rel_feat_ops = rel_feat_ops
        self.nbr_types = nbr_types
        self.ego_dist = ego_dist
        self.lambda_value = lambda_value
        self.diffusion_iter = diffusion_iter
        self.transform_method = transform_method
        self.log_binning_alpha = log_binning_alpha

        self.X = None
        self.feat_defs = None

    def fit_transform(self, g):
        self._prepare_base_feats(g, transform=self.transform_method)
        self._search_rel_func_space(g,
                                    diffusion_iter=self.diffusion_iter,
                                    transform=self.transform_method)

        return self.X

    def fit(self, g):
        self._prepare_base_feats(g, transform=self.transform_method)
        self._search_rel_func_space(g,
                                    diffusion_iter=self.diffusion_iter,
                                    transform=self.transform_method)
        return self

    def transform(self,
                  g,
                  feat_defs=None,
                  diffusion_iter=None,
                  transform_method=None,
                  log_binning_alpha=None):
        if feat_defs is None:
            feat_defs = self.get_feat_defs(flatten=True)
        if diffusion_iter is None:
            diffusion_iter = self.diffusion_iter
        if transform_method is None:
            transform_method = self.transform_method
        if log_binning_alpha is None:
            log_binning_alpha = self.log_binning_alpha

        feat_defs_computed = []
        for feat_def in feat_defs:
            feat_comps = self._feat_def_to_feat_comps(feat_def)

            tmp_feat_def = ''
            for feat_comp in reversed(feat_comps):
                prev_tmp_feat_def = tmp_feat_def
                feat_op = feat_comp[0]
                nbr_type = None
                if len(feat_comp) == 2:
                    nbr_type = feat_comp[1]

                # base feature
                if nbr_type is None:
                    tmp_feat_def = feat_op
                    if not tmp_feat_def in feat_defs_computed:
                        self._comp_base_feat(g, feat_op)
                # rel operators
                else:
                    tmp_feat_def = self._gen_feat_def(feat_op, nbr_type,
                                                      prev_tmp_feat_def)
                    if not tmp_feat_def in feat_defs_computed:
                        self._comp_rel_op_feat(g, feat_op, nbr_type,
                                               prev_tmp_feat_def)

                feat_defs_computed.append(tmp_feat_def)

        X = np.zeros((g.num_vertices(), len(feat_defs)))
        for i, feat_def in enumerate(feat_defs):
            X[:, i] = g.vertex_properties[feat_def].a

        Processing.feat_diffusion(X, g, iter=diffusion_iter)
        if transform_method == 'log_binning':
            Processing().log_binning(self.X, alpha=log_binning_alpha)

        return X

    def get_feat_defs(self, flatten=True):
        result = []
        if flatten:
            for ith_feat_defs in self.feat_defs:
                result += ith_feat_defs
        else:
            result = self.feat_defs
        return result

    def get_related_subgraph(self, g, v, feat_def):
        filter_prop = g.new_vertex_property('bool')
        filter_prop.a = np.zeros(g.num_vertices(), dtype=bool)

        feat_comps = self._feat_def_to_feat_comps(feat_def)
        related_nodes = [v]
        prev_nbrs = [v]
        for feat_comp in feat_comps:
            if len(feat_comp) >= 2:
                for u in prev_nbrs:
                    nbr_type = feat_comp[1]
                    nbrs = list(eval('NeighborOp().' + nbr_type +
                                     '_nbr(g, u)'))
                    related_nodes += nbrs
                    prev_nbrs = nbrs
        for u in set(related_nodes):
            filter_prop[u] = True

        gv = gt.GraphView(g, vfilt=filter_prop.a == True)

        return gv

    def _comp_base_feat(self, g, base_feat_def):
        '''
        Compute and store base feature by using a method provided by graph-tool
        '''
        gt_measures = [
            # from graph-tool's centrarity
            'pagerank',
            'betweenness',
            'closeness',
            'eigenvector',
            'katz',
            'hits',
            # from graph-tool's topology
            'kcore_decomposition',
            'sequential_vertex_coloring',
            'max_independent_vertex_set',
            'label_components',
            'label_out_component',
            'label_largest_component'
        ]

        # judge whether weighted feature or not
        eweight = None
        b_feat = base_feat_def
        if len(base_feat_def) > 2 and base_feat_def[:2] == 'w_':
            eweight = g.edge_properties['weight']
            b_feat = base_feat_def[2:]

        if b_feat == 'in_degree' or b_feat == 'out_degree':
            g.vertex_properties[base_feat_def] = g.new_vertex_property(
                'double')
            for v in g.vertices():
                g.vertex_properties[base_feat_def][v] = eval("v." + b_feat)(
                    weight=eweight)
        elif b_feat == 'total_degree':
            g.vertex_properties[base_feat_def] = g.new_vertex_property(
                'double')
            for v in g.vertices():
                g.vertex_properties[base_feat_def][v] = v.in_degree(
                    weight=eweight) + v.out_degree(weight=eweight)
        elif b_feat in gt_measures and eweight is None:
            vals = eval('gt.' + b_feat)(g)
            if b_feat == "betweenness":
                vals = vals[0]
            elif b_feat == "eigenvector":
                vals = vals[1]
            elif b_feat == "hits":
                vals = vals[1]  # authority value TODO: hub value
            elif b_feat == "label_components":
                vals = vals[0]
            g.vertex_properties[base_feat_def] = vals
        elif b_feat in gt_measures and eweight is not None:
            vals = eval('gt.' + b_feat)(g, weight=eweight)
            if b_feat == "betweenness":
                vals = vals[0]
            elif b_feat == "eigenvector":
                vals = vals[1]
            elif b_feat == "hits":
                vals = vals[1]  # authority value TODO: hub value
            g.vertex_properties[base_feat_def] = vals
        else:
            try:
                g.vertex_properties[base_feat_def]
            except:
                print('base feature, ' + base_feat_def +
                      ', is set which is not supported in graph-tool. Load ' +
                      base_feat_def +
                      'as a vertex_properties before using DeepGL')

        return self

    def _prepare_base_feats(self, g, transform='log_binning'):
        '''
        Compute and store all base features
        '''
        self.X = np.zeros((g.num_vertices(), len(self.base_feat_defs)))
        self.feat_defs = [self.base_feat_defs]
        for i, feat_def in enumerate(self.base_feat_defs):
            self._comp_base_feat(g, feat_def)
            self.X[:, i] = g.vertex_properties[feat_def].a

        if transform == 'log_binning':
            Processing().log_binning(self.X, alpha=self.log_binning_alpha)

        return self

    def _gen_feat_def(self, rel_op, nbr_type, prev_feat_def):
        '''
        Generate feature definition string from inputs
        '''
        return rel_op + '^' + nbr_type + '-' + prev_feat_def

    def _comp_rel_op_feat(self, g, rel_op, nbr_type, prev_feat_def):
        '''
        Compute and store new features by applying relational operators
        '''
        new_feat_def = self._gen_feat_def(rel_op, nbr_type, prev_feat_def)

        g.vertex_properties[new_feat_def] = g.new_vertex_property('double')
        x = g.vertex_properties[prev_feat_def]

        for v in g.vertices():
            nbrs = eval('NeighborOp().' + nbr_type + '_nbr(g, v)')
            feat_val = eval('RelFeatOp().' + rel_op + '(nbrs, x)')
            # to avoid the result > inf (this happens when using hadamard)
            feat_val = min(feat_val, np.finfo(np.float64).max)
            g.vertex_properties[new_feat_def][v] = feat_val

        return new_feat_def

    def _search_rel_func_space(self,
                               g,
                               diffusion_iter=0,
                               transform='log_binning'):
        '''
        Searching the relational function space (Sec. 2.3, Rossi et al., 2018)
        '''
        n_rel_feat_ops = len(self.rel_feat_ops)
        n_nbr_types = len(self.nbr_types)

        for l in range(1, self.ego_dist):
            prev_feat_defs = self.feat_defs[l - 1]
            new_feat_defs = []

            for i, op in enumerate(self.rel_feat_ops):
                for j, nbr_type in enumerate(self.nbr_types):
                    for k, prev_feat_def in enumerate(prev_feat_defs):
                        new_feat_def = self._comp_rel_op_feat(
                            g, op, nbr_type, prev_feat_def)

                        new_feat = np.expand_dims(
                            g.vertex_properties[new_feat_def].a, axis=1)
                        self.X = np.concatenate((self.X, new_feat), axis=1)
                        new_feat_defs.append(new_feat_def)

            self.feat_defs.append(new_feat_defs)

            Processing.feat_diffusion(self.X, g, iter=diffusion_iter)

            if transform == 'log_binning':
                Processing().log_binning(self.X, alpha=self.log_binning_alpha)

            # feature pruning
            self.X, self.feat_defs = Processing().prune_feats(
                self.X, self.feat_defs, lambda_value=self.lambda_value)

        return self

    def _feat_def_to_feat_comps(self, feat_def):
        ''' From feature definition string, generate a list of pairs of
        a relational feature operator and a neigbor type.
        e.g., 'mean^in-mean^in-in_degree' => [['mean', 'in'], ['mean', 'in'], ['in_degree']]
        '''
        feat_comps = feat_def.split('-')
        for i, feat_comp in enumerate(feat_comps):
            feat_comps[i] = feat_comp.split('^')
        return feat_comps
