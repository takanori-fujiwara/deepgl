import graph_tool.all as gt

from deepgl import DeepGL

# Data from http://www.sociopatterns.org/datasets/primary-school-cumulative-networks/
g1 = gt.load_graph('./data/school_day2.xml.gz')

# DeepGL setting
deepgl = DeepGL(base_feat_defs=[
    'total_degree', 'eigenvector', 'katz', 'pagerank', 'closeness',
    'betweenness', 'gender'
],
                ego_dist=3,
                nbr_types=['all'],
                lambda_value=0.7,
                transform_method='log_binning')

# network representation learing with DeepGL
X1 = deepgl.fit_transform(g1)

# results
print(X1)
print(X1.shape)
for nth_layer_feat_def in deepgl.feat_defs:
    print(nth_layer_feat_def)

# transfer/inductive learning example
g2 = gt.load_graph('./data/school_day1.xml.gz')
X2 = deepgl.transform(g2)

# results
print(X2)
print(X2.shape)
