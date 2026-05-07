import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer
import torch_geometric
from torch_geometric.data import InMemoryDataset, Data
from torch_sparse import coalesce


""" Adapted from https://github.com/Graph-COM/ED-HNN """

class AddHypergraphSelfLoops(torch_geometric.transforms.BaseTransform):
    def __init__(self, ignore_repeat=True):
        super().__init__()
        # whether to detect existing self loops
        self.ignore_repeat = ignore_repeat
    
    def __call__(self, data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_hyperedges = data.num_hyperedges

        node_added = torch.arange(num_nodes, device=edge_index.device, dtype=torch.int64)
        if self.ignore_repeat:
            # 1. compute hyperedge degree
            hyperedge_deg = torch.zeros(num_hyperedges, device=edge_index.device, dtype=torch.int64)
            hyperedge_deg = hyperedge_deg.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
            hyperedge_deg = hyperedge_deg[edge_index[1]]

            # 2. if a node has a hyperedge with degree 1, then this node already has a self-loop
            has_self_loop = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.int64)
            has_self_loop = has_self_loop.scatter_add(0, edge_index[0], (hyperedge_deg == 1).long())
            node_added = node_added[has_self_loop == 0]

        # 3. create dummy hyperedges for other nodes who have no self-loop
        hyperedge_added = torch.arange(num_hyperedges, num_hyperedges + node_added.shape[0])
        edge_indx_added = torch.stack([node_added, hyperedge_added], 0)
        edge_index = torch.cat([edge_index, edge_indx_added], -1)

        # 4. sort along w.r.t. nodes
        _, sorted_idx = torch.sort(edge_index[0])
        data.edge_index = edge_index[:, sorted_idx].long()

        return data

class HypergraphDataset(InMemoryDataset):


    cocitation_list = ['cora', 'citeseer', 'pubmed']
    coauthor_list = ['coauthor_cora', 'coauthor_dblp']
    LE_list = ['20newsW100', 'ModelNet40', 'zoo', 'NTU2012', 'Mushroom']
    yelp_list = ['yelp']
    cornell_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'congress-bills', 'senate-committees']

    existing_dataset = cocitation_list + coauthor_list + LE_list + yelp_list + cornell_list

    @staticmethod
    def parse_dataset_name(name):
        name_cornell = '-'.join(name.split('-')[:-1])
        extras = {}
        if name_cornell in HypergraphDataset.cornell_list:
            extras['feature_dim'] = int(name.split('-')[-1])
            name = name_cornell

        return name, extras

    @staticmethod
    def dataset_exists(name):
        name, _ = HypergraphDataset.parse_dataset_name(name)
        return (name in HypergraphDataset.existing_dataset)

    def __init__(self, root, name, feature_noise=None, transform=None, args=None, pre_transform=None):
 
        assert self.dataset_exists(name), f'Dataset {name} is not defined'
        self.name = name
        self.feature_noise = feature_noise
        self.root = root
        self.dname = args.dname

        processed_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found at {processed_path}. Please make sure the dataset is in the 'data/{self.name}/processed/' directory.")
       
        self._data, self.slices = torch.load(processed_path, weights_only=False)
        device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

        edge_index = self._data.edge_index
        _, sorted_idx = torch.sort(edge_index[0])#returns the sorted values and their original indices
        edge_index = edge_index[:, sorted_idx].long()#returns the sorted edge_index based on the nodes 
        self._data.num_hyperedges  = int(self._data.num_hyperedges)
        num_nodes, num_hyperedges = self._data.num_nodes, self._data.num_hyperedges
        
        assert ((num_nodes + num_hyperedges - 1) == self._data.edge_index.max().item())

        cidx = torch.where(edge_index[0] == num_nodes)[0].min()#find boundary where hyperedges start
        self._data.edge_index = edge_index[:, :cidx].long()#keep only node → hyperedge incidences
        self._data.edge_index[1] -= num_nodes#reindex hyperedges to [0, E-1]

        # Calculate node degrees
        node_degrees = torch.zeros(num_nodes, dtype=torch.int64, device=edge_index.device)
        for node_idx in self._data.edge_index[0]:
            node_degrees[node_idx] += 1
       
        # Calculate hyperedge degrees
        hyperedge_degrees = torch.zeros(num_hyperedges, dtype=torch.int64, device=edge_index.device)
        for hyperedge_idx in self._data.edge_index[1]:
            hyperedge_degrees[hyperedge_idx] += 1
        
        # Print degree statistics
        print(f"Node degrees - Mean: {node_degrees.float().mean():.4f}, Min: {node_degrees.min()}, Max: {node_degrees.max()}")
        print(f"Number of unique node degrees: {torch.unique(node_degrees).numel()}")
        print(f"Number of unique hyperedge degrees: {torch.unique(hyperedge_degrees).numel()}")
        print(f"Hyperedge degrees - Mean: {hyperedge_degrees.float().mean():.4f}, Min: {hyperedge_degrees.min()}, Max: {hyperedge_degrees.max()}")
        
        # Find and print isolated nodes and hyperedges
        isolated_nodes = torch.where(node_degrees == 0)[0]
        isolated_hyperedges = torch.where(hyperedge_degrees == 0)[0] 
        isolated_hyperedges_1 = torch.where(hyperedge_degrees == 1)[0]
        
        if isolated_nodes.numel() > 0:
            print(f"Isolated nodes (degree 0): {isolated_nodes.numel()} nodes")
            #print(f"Isolated nodes (degree 0): {isolated_nodes.numel()} nodes - {isolated_nodes.tolist()}")
        else:
            print(f"Isolated nodes (degree 0): None")
        
        if isolated_hyperedges.numel() > 0:
            print(f"Isolated hyperedges (degree 0): {isolated_hyperedges.numel()} hyperedges - {isolated_hyperedges.tolist()}")
        else:
            print(f"Isolated hyperedges (degree 0): None")
        if isolated_hyperedges_1.numel() > 0:
            print(f"Isolated hyperedges (degree 1): {isolated_hyperedges_1.numel()} hyperedges")
        print(f"Number of hyperedges: {self._data.num_hyperedges}")
        
        
        
        if transform is not None:
            self._data = transform(self._data)

        if args.extract_subgraph:
            sub_E, sub_root, sub_node_idx = extract_sub_hypergraphs(self._data.edge_index, self._data.num_nodes, k_hop=4)
            self._data.sub_E = sub_E
            self._data.sub_root = sub_root
            self._data.sub_node_index = sub_node_idx
    
        
      

        num_hyperedges= self._data.edge_index[1].max().item() + 1

        node_degrees = torch.zeros(num_nodes, dtype=torch.int64, device=edge_index.device)
        for node_idx in self._data.edge_index[0]:
            node_degrees[node_idx] += 1
       
        # Calculate hyperedge degrees
        hyperedge_degrees = torch.zeros(num_hyperedges, dtype=torch.int64, device=edge_index.device)
        for hyperedge_idx in self._data.edge_index[1]:
            hyperedge_degrees[hyperedge_idx] += 1
        
        # Print degree statistics
        print(f"Node degrees - Mean: {node_degrees.float().mean():.4f}, Min: {node_degrees.min()}, Max: {node_degrees.max()}")
        print(f"Number of unique node degrees: {torch.unique(node_degrees).numel()}")
        print(f"Number of unique hyperedge degrees: {torch.unique(hyperedge_degrees).numel()}")
        print(f"Hyperedge degrees - Mean: {hyperedge_degrees.float().mean():.4f}, Min: {hyperedge_degrees.min()}, Max: {hyperedge_degrees.max()}")
        
        # Find and print isolated nodes and hyperedges
        isolated_nodes = torch.where(node_degrees == 0)[0]
        isolated_hyperedges = torch.where(hyperedge_degrees == 0)[0] 
        isolated_hyperedges_1 = torch.where(hyperedge_degrees == 1)[0]
        
        if isolated_nodes.numel() > 0:
            print(f"Isolated nodes (degree 0): {isolated_nodes.numel()} nodes")
            #print(f"Isolated nodes (degree 0): {isolated_nodes.numel()} nodes - {isolated_nodes.tolist()}")
        else:
            print(f"Isolated nodes (degree 0): None")
        
        if isolated_hyperedges.numel() > 0:
            print(f"Isolated hyperedges (degree 0): {isolated_hyperedges.numel()} hyperedges - {isolated_hyperedges.tolist()}")
        else:
            print(f"Isolated hyperedges (degree 0): None")
        if isolated_hyperedges_1.numel() > 0:
            print(f"Isolated hyperedges (degree 1): {isolated_hyperedges_1.numel()} hyperedges")
        print(f"Number of hyperedges: {num_hyperedges}")


        

        # self._data.adj = torch.empty(num_nodes, num_nodes)

    @property# decorator to define a property, the 
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        if self.feature_noise is not None and self.dname in HypergraphDataset.cornell_list:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self._data.num_node_features

    @property
    def num_classes(self):
        return self._data.num_classes

    def __repr__(self):
        return '{}(feature_noise={})'.format(self.name, self.feature_noise)

    
def extract_sub_hypergraphs(edge_index, num_nodes, k_hop):
    """
    For each root node, extract the k-hop sub-hypergraph.
    
    Definition: collect all hyperedges traversed during k-hop BFS 
    from the root. Each traversed hyperedge is kept WHOLE — all its 
    members are included in the sub-hypergraph node set. No truncation.
    
    This is the correct semantics for hypergraphs: a hyperedge encodes
    a joint relationship among ALL its members simultaneously. Clipping
    it would destroy this higher-order interaction.
    """
    V, E = edge_index[0], edge_index[1]
    num_hedges = E.max().item() + 1

    node_to_edges = [[] for _ in range(num_nodes)]
    edge_to_nodes = [[] for _ in range(num_hedges)]
    for idx in range(V.shape[0]):
        v, e = V[idx].item(), E[idx].item()
        node_to_edges[v].append(e)
        edge_to_nodes[e].append(v)

   
    all_sub_E = []
    all_sub_root = []
    all_sub_node_index = []

    for root in range(num_nodes):

        if k_hop == 0:
            # 0-hop: just the root node, no hyperedges
            all_sub_root.append(root)
            all_sub_node_index.append(root)
            # sub_V and sub_E are empty for this root
            continue

        visited_nodes = {root}
        visited_edges = set()
        frontier = {root}

        for _ in range(k_hop):
            new_frontier = set()
            for node in frontier:
                for e in node_to_edges[node]:
                    if e not in visited_edges:
                        visited_edges.add(e)
                        # Keep hyperedge whole — pull in ALL its members
                        for u in edge_to_nodes[e]:
                            if u not in visited_nodes:
                                visited_nodes.add(u)
                                new_frontier.add(u)
            frontier = new_frontier
            if not frontier:
                break

        # Build incidence pairs for the sub-hypergraph.
        # Every member of every visited hyperedge is in visited_nodes
        # by construction — no membership check needed.
        edge_local_id = {e: i for i, e in enumerate(visited_edges)}

        for e in visited_edges:
            local_e = edge_local_id[e]
            for u in edge_to_nodes[e]:
            
                all_sub_E.append(local_e)
                all_sub_root.append(root)
                all_sub_node_index.append(u)

  
    sub_E          = torch.tensor(all_sub_E,          dtype=torch.long)
    sub_root       = torch.tensor(all_sub_root,       dtype=torch.long)
    sub_node_index = torch.tensor(all_sub_node_index, dtype=torch.long)
   
    return sub_E, sub_root, sub_node_index