import pandas as pd
import networkx as nx

def list_to_graph(edges):
    g = nx.MultiDiGraph()
    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.MultiDiGraph.add_edges_from.html#networkx.MultiDiGraph.add_edges_from
    g.add_edges_from(edges)
    # print(g.in_edges(0, keys=True))
    # print(g.out_edges(0, keys=True))
    return g

def df_to_graph(df):
    # list of (u, v, k) tuples
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.itertuples.html
    edges = list(df.itertuples(index=False, name=None))
    return list_to_graph(edges)