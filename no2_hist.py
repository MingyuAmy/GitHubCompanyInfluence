import networkx as nxl

import numpy as np
import matplotlib.pyplot as plt

# 486 company, only use features from matrix

common_contributor_threshold = 100

# config to display graph

show_graph = [True, True, True, True, True]
# show_graph = [False, True, False, False, False]

# --------------graph part--------------
def draw_graph(G, title):
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_title(title)
    ax.axis("off")
    plot_options = {"node_size": 10, "with_labels": True, "width": 0.15}
    nx.draw_networkx(G, pos=nx.random_layout(G), ax=ax, **plot_options)
    plt.show()


def plot_hist(feature, title, xlabel, bins=25):
    plt.figure(figsize=(15, 8))
    sorted_map = dict(sorted(feature.items(), key=lambda item: item[1]))
    print(sorted_map)

    max_x = max(feature.values())
    plt.hist(feature.values(), bins=bins)
    # plt.xticks(ticks=[0, 0.025, 0.05, 0.1, 0.15, 0.2])  # set the x axis ticks
    plt.xticks(ticks=list(np.arange(0, max_x, 0.025)))  # set the x axis ticks
    plt.title(title, fontdict={"size": 35}, loc="center")
    plt.xlabel(xlabel, fontdict={"size": 20})
    plt.ylabel("Counts", fontdict={"size": 20})
    plt.show()


def main():
    from common import read_matrix
    names, data = read_matrix("data/matrix_22-12-10_23_18.csv")
    print('\n---------------------------')
    print('company count:', len(names))
    print(names)
    print('---------------------------')

    # build the graph
    G = nx.Graph()
    G.add_nodes_from(names)
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i != j and data[i][j] >= common_contributor_threshold:
                G.add_edge(names[i], names[j])

    np.mean([d for _, d in G.degree()])
    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())
    print('neighbors of the node/average degree of a node:', np.mean([d for _, d in G.degree()]))

    if show_graph[0]:
        draw_graph(G, 'Common Contributor Threshold: {}'.format(common_contributor_threshold))

    # input('\nPress any key to continue...\n')

    if show_graph[1]:
        degree_centrality = nx.centrality.degree_centrality(G)
        print('degree_centrality:')
        plot_hist(degree_centrality, 'Degree Centrality Histogram', 'Degree Centrality')

    input('\nPress any key to continue...\n')

    if show_graph[2]:
        betweenness_centrality = nx.centrality.betweenness_centrality(G)
        print('betweenness_centrality:')
        plot_hist(betweenness_centrality, 'Betweenness Centrality Histogram', 'Betweenness Centrality')

    input('\nPress any key to continue...\n')

    if show_graph[3]:
        closeness_centrality = nx.centrality.closeness_centrality(G)
        print('closeness_centrality :')
        plot_hist(closeness_centrality, 'Closeness Centrality Histogram', 'Closeness Centrality')

    input('\nPress any key to continue...\n')

    if show_graph[4]:
        eigenvector_centrality = nx.centrality.eigenvector_centrality(G)
        print('eigenvector_centrality')
        plot_hist(eigenvector_centrality, 'Eigenvector Centrality Histogram', 'Eigenvector Centrality')


if __name__ == '__main__':
    main()
