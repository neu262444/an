import random
from collections import defaultdict



class OneBatch_:
    """
    Efficient 1-hop batch constructor for hypergraphs.

    hypergraph: dict
        key   = hyperedge id / author
        value = iterable of nodes in that hyperedge
    """

    def __init__(self, dataset, args):
        self.hypergraph = dataset['hypergraph']
       
        self.seed = args.seed

        # -------- build node → incident hyperedges index -------- #
        self.node2edges = defaultdict(list)

        for edge in self.hypergraph.values():
            edge = tuple(edge)
            for v in edge:
                self.node2edges[v].append(edge)

    def batch_for_targets(self, target_nodes):
        """
        Returns:
            batch_dict: dict[target_node] -> list of incident hyperedges (lists)
        """

        batch_dict = {}
        all_nodes = set(target_nodes)

        for v in target_nodes:
            edges = self.node2edges.get(v, [])
            sampled_edges = []

            for edge in edges:
                    
                # if len(edge) <= self.M:
                    sampled_edges.append(list(edge))
                    all_nodes.update(edge)
                
                # else:
                #     edge_lst = list(edge)
                #     edge_lst.remove(v)

                #     random.seed(self.seed)
                #     sampled = random.sample(edge_lst, self.M - 1) + [v]
                #     sampled_edges.append(sampled)
                #     all_nodes.update(sampled)

            batch_dict[v] = sampled_edges
            
        return batch_dict
