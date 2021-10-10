"""
@author: Yuan Guo, Andrey Gritsenko
         SPIRAL Group
         Electrical & Computer Engineering
         Northeastern University
"""

import random
import numpy as np
import networkx as nx


def read_graph(Amatrix):
	"""
	Reads the input network in networkx
	"""
	G = nx.from_numpy_matrix(Amatrix)
	G = G.to_undirected()
	return G


class Graph:
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		for walk_iter in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed

		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)

		alias_edges = {}

		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q



def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]


def frequency(walks):
	""" for a random walk, calculate the 3/4 occurrence probability for
	negative sampling"""
	P_m = {}
	for walk in walks:
		for item in walk:
			try:
				P_m[item] += 1
			except:
				P_m[item] = 1
	for key, value in P_m.items():
		P_m[key] = value**0.75
	return P_m


def negative_frequency(P_m):
	"""get the negative probability"""
	sample_num = []
	sample_prob = []
	for key, value in P_m.items():
		sample_num.append(key)
		sample_prob.append(value)
	return sample_num, np.array(sample_prob)/sum(sample_prob)


def get_negative_sample(context, num, prob, Gn):
	"""sample negative nodes for each context node"""
	negative_list = []
	while len(negative_list) < Gn:
		negative_sample = np.random.choice(num, p=prob.ravel())
		if negative_sample != context:
			negative_list.append(negative_sample)
		else:
			pass
	return np.array([negative_list])


def skip_train(walks, window_size, negative_size):
	"""
	use the wallks to generate negative samples for neural network
	generate train input under the skip-gram formula
	"""
	P_m = frequency(walks)
	Num, Prob = negative_frequency(P_m)
	targets = []
	contexts = []
	similarity = []
	negative_samples = []
	for walk in walks:
		for source_id, source in enumerate(walk):
			reduced_window = np.random.randint(window_size)
			start = max(0, source_id - window_size + reduced_window)
			for target_id in range(start, source_id + window_size + 1 - reduced_window):
				if target_id != source_id:
					try:
						target = walk[target_id]
						targets.append(target)
						contexts.append(source)
						negative_samples.append(get_negative_sample(target, Num, Prob, negative_size))
						similarity.append(np.concatenate((np.ones(1), np.zeros(negative_size))))
					except:
						pass
				else:
					pass
	return map(np.array, (targets, contexts, similarity, negative_samples))


def getRandomWalks(adj_mat, p, q, n_walks, walk_length):
	nx_G = read_graph(adj_mat)
	G = Graph(nx_G, False, p, q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(n_walks, walk_length)
	return walks


def getRWSimilarity(adj_mat, n_walks, walk_length, window_size, p, q, n_negative):
	"""
	Implements random walk similarity in analogy to node2vec

	Inputs:
		ADJ_MAT Adjacency matrix of a graph
		U, V Indeces of graph nodes pair, between which similarity is computed

	Outputs:
		Measure of similarity between two nodes
	"""
	walks = getRandomWalks(adj_mat, p, q, n_walks, walk_length)
	target, context, similarity, negative_samples = skip_train(walks, window_size, n_negative)
	return target, context, similarity, negative_samples




