import pydotplus
import plotly.express as px
from pydotplus import graph_from_dot_data
from IPython.display import Image

dot_data = 'tree.dot'
export_graphviz(model_dtree, out_file=dot_data, feature_names=data_df.columns[:-1], filled=True, rounded=True, impurity=True, special_characters=True)
graph = pydotplus.graph_from_dot_file(dot_data)
Image(graph.create_png(), unconfined=True)
