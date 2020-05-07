import networkx as nx
G = nx.Graph()

# Nodes
G.add_node(1) # This function is to add one node
G.add_nodes_from([4,5]) # This function takes in a list and adds multiple nodes. It can also be used to add just one node
G.add_nodes_from([3])

# Edges
G.add_edge(1,3) # This function is to add one edge
G.add_edges_from([(1,4),(1,5)]) # This function takes in a list and adds multiple edges. It can also be used to add just one edge
G.add_edges_from([(1,8)]) # This function is adding an edge but also is implicitly adding nodes as node 8 does not exist

# Removing nodes and edges
G.remove_nodes_from([5]) #This removes node 5. This also removes any edges associated with that node
G.remove_edges_from([(1,4)]) #This removes the edge but not the nodes themselves

# Number of nodes and edges
G.number_of_nodes()
G.number_of_edges

# Working with karate club data

G = nx.karate_club_graph()
nx.draw(G, with_labels = True, node_color="yellow", edge_color = "black")
G.degree() # Returns a dictionary with the degree of each node
	# G.degree()[10] or G.degree(10) can be used to subset the degree of node 10

# Generating a random network (Erdos-Renyi Graphs):

def er_graph(N,p):
	import networkx as nx
	from scipy.stats import bernoulli
	
	G = nx.Graph()
	G.add_nodes_from(range(N))

	for node1 in G.nodes():
		for node2 in range(node1+1, len(G.nodes()), 1):
			if node2>node1 and bernoulli.rvs(p=p) == 1:
				G.add_edge(node1, node2)
	return G		

nx.draw(er_graph(10,0.5), with_labels = True)

# Plotting degree distribution

def plot_degree_distribution(G):
	degree_sequence = [d for n,d in G.degree()]
	plt.hist(degree_sequence, histtype = "step")
	plt.xlabel("Degree $k$")
	plt.ylabel("$P(k)")
	plt.title("Degree Distribution")


plot_degree_distribution(er_graph(100,0.2))

# Plotting microfinance network data from villages in India

import numpy as np
A1 = np.loadtxt("adj_allVillageRelationships_vilno_1.csv", delimiter = ",")
A2 = np.loadtxt("adj_allVillageRelationships_vilno_2.csv", delimiter = ",")

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
	print("Number of nodes: %d" % G.number_of_nodes())
	print("Number of edges: %d" % G.number_of_edges())
	print("Mean degree: %.2f" % np.mean([d for n,d in G.degree()]))

basic_net_stats(G1)
basic_net_stats(G2)


# Creating a modified ER network where the probability of a new edge reduces as number of degrees of nodes increases
# Concept is that for a village of size 1000 or 5000 the number of connections will still be about 10-20. The original ER plots ~100. This new function is built to combat that effect of too many friends
def mod_er_graph(N):
	import networkx as nx
	from scipy.stats import bernoulli
	
	G = nx.Graph()
	G.add_nodes_from(range(N))

	for node1 in G.nodes():
		for node2 in range(node1+1, len(G.nodes()), 1):
			deg_n1 = G.degree(node1)+1
			deg_n2 = G.degree(node2)+1
			p = (deg_n1 + deg_n2) / (2 * deg_n1 * deg_n1 * deg_n2 * deg_n2)
			if node2>node1 and bernoulli.rvs(p=p) == 1:
				G.add_edge(node1, node2)
	return G




# Figuring out the largest connected component in a network

def connected_component_subgraphs(G): #Ideally we need to directly use the nx.connected_component_subgraph but that function is deprecated. Hence, we are have defined this function in place
    for c in nx.connected_components(G):
        yield G.subgraph(c)


gen = connected_component_subgraphs(G1) # Generates a sequence of objects which can be accessed with the 'next method'
g = gen.__next__() # Accesses an element in the generated sequence. Ordering is arbitrary i.e. not largest to smallest etc.
len(gen.__next__()) # Gives the number of nodes in the next connected component. You can keep running this until we run out of components

G1_LCC = max(connected_component_subgraphs(G1), key = len) # This gives us the largest component of G1 where 'large' is defined by the length function i.e. maximum number of nodes
G2_LCC = max(connected_component_subgraphs(G2), key = len)

len(G1_LCC)
G2_LCC.number_of_nodes()