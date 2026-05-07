import torch
import numpy as np
import scipy.sparse as sp
import itertools
from scipy.sparse.linalg import eigsh
from itertools import chain
from collections import defaultdict

import random
import copy
import numpy as np
import os
import dask
from dask.diagnostics import ProgressBar
from dask import delayed

def edge_index_to_dict(data):

    edge_index = data.edge_index.cpu().numpy()
    dataset = {}

    hg_dict = {}
    for node, edge in zip(edge_index[0], edge_index[1]):
        if edge not in hg_dict:
            hg_dict[edge] = []
        hg_dict[edge].append(node)
    hg_dict = {edge: set(nodes) for edge, nodes in hg_dict.items()}

    dataset['hypergraph'] = hg_dict
    dataset['n'] = data.x.shape[0]

    return dataset

def clique_expansion_edge_index(hyperedge_index):
    """
    Compute the clique expansion edge index for a hypergraph.
    
    Given a hyperedge index matrix of shape (2, M) where:
      - The first row contains node indices.
      - The second row contains hyperedge indices.
      
    For each hyperedge with two or more nodes, every distinct pair of nodes 
    is connected (undirected edge). Duplicate edges (resulting from overlapping hyperedges)
    are removed. The function returns the edge index of the resulting graph, where
    each column represents an edge and the two rows are the node indices forming that edge.
    
    Args:
        hyperedge_index (np.ndarray): A 2 x M numpy array containing the hypergraph's edges.
        
    Returns:
        np.ndarray: A 2 x num_edges numpy array representing the clique-expanded graph's edge index.
    """
    # Build a dictionary mapping hyperedge id to a list of nodes in that hyperedge.
    
    hyperedge_dict = {}
    nodes = hyperedge_index[0].cpu().numpy()
    hedges = hyperedge_index[1].cpu().numpy()

    for node, hedge in zip(nodes, hedges):
        hyperedge_dict.setdefault(int(hedge), []).append(int(node))

    # Use a set to collect unique undirected edges.
    edges_set = set()
    
    for nodes in hyperedge_dict.values():
        # Skip hyperedges with fewer than 2 nodes.
        if len(nodes) < 2:
            continue
        # Generate all unique pairs using combinations.
        for edge in itertools.combinations(nodes, 2):
            # Sort the tuple so that (i, j) and (j, i) are considered the same.
            edge = tuple(sorted(edge))
            edges_set.add(edge)
    
    # Convert the set of edges to a sorted list (for reproducibility, optional).
    edges = sorted(list(edges_set))
    
    # If no edges are present, return an empty array of shape (2, 0)
    if not edges:
        return np.empty((2, 0), dtype=int)
    
    # Convert the list of edges to a 2 x num_edges numpy array.
    edge_index = np.array(edges).T
    
   
    return edge_index


def clique_expansion_edge_index_zh(hyperedge_index):
    """
    Zhou-style weighted clique expansion.
    Returns:
        edge_index: (2, E)
        edge_weight: (E,)
    """
    hyperedge_dict = defaultdict(list)
    for v, e in zip(hyperedge_index[0], hyperedge_index[1]):
        hyperedge_dict[int(e)].append(int(v))

    edge_weight = defaultdict(float)

    for nodes in hyperedge_dict.values():
        k = len(nodes)
        if k < 2:
            continue
        w = 1.0 / (k - 1)   # normalization term
        for i, j in itertools.combinations(nodes, 2):
            a, b = min(i, j), max(i, j)
            edge_weight[(a, b)] += w

    if len(edge_weight) == 0:
        return np.empty((2, 0), dtype=int), np.empty((0,), dtype=float)

    edges = np.array(list(edge_weight.keys()))
    weights = np.array(list(edge_weight.values()))

    edge_index = edges.T
    return edge_index, weights






def remove_isolated_nodes(A):
    deg = np.array(A.sum(1)).flatten()
    mask = deg > 0
    A2 = A[mask][:, mask]
    return A2, mask


def normalized_laplacian(A, eps=1e-5):
    deg = np.array(A.sum(1)).flatten()
    deg[deg == 0] = 1.0

    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    I = sp.eye(A.shape[0])
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    L = L + eps * I
    return L.tocsr()


def laplacian_positional_encoding(L, k):
    try:
        eigvals, eigvecs = eigsh(
            L, k=k+1, sigma=0.0, which='LM', tol=1e-3
        )
    except:
        eigvals, eigvecs = eigsh(
            L, k=k+1, which='SM', maxiter=5000, tol=1e-2
        )

    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    return eigvecs[:, 1:k+1]


def edge_index_to_adj(edge_index):
    num_nodes = edge_index.max() + 1
   
    row, col = edge_index
    data = np.ones(len(row), dtype=np.float32)

    # undirected graph: add (i,j) and (j,i)
    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()

def weighted_edge_index_to_adj(edge_index, edge_weight, num_nodes):
    row, col = edge_index
    data = edge_weight

    A = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A + A.T          # undirected
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()

def zhou_laplacian(A, eps=1e-6):
    deg = np.array(A.sum(1)).flatten()
    deg[deg == 0] = 1.0
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    I = sp.eye(A.shape[0])
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    L = L + eps * I
    return L.tocsr()

def hyperedge_index_to_incidence(hyperedge_index):
    nodes = hyperedge_index[0].cpu().numpy()
    edges = hyperedge_index[1].cpu().numpy()

    num_nodes = nodes.max() + 1
    num_edges = edges.max() + 1

    data = np.ones(len(nodes), dtype=np.float32)
    H = sp.coo_matrix((data, (nodes, edges)),
                      shape=(num_nodes, num_edges))
    return H.tocsr()


def node_node_rw_prob_pe(H, t=1): 
    """
    Multi-step random-walk transition matrix for hypergraphs.

    Args:
        H: torch.sparse_coo_tensor [num_nodes, num_hyperedges]
        t: diffusion steps (walk length)

    Returns:
        P_final: torch.Tensor [num_nodes, num_nodes]
                 The transition probability matrix after t steps (P^t).
    """

    device = H.device
    num_nodes = H.size(0)

    H = H.coalesce()

    # -------------------
    # node degrees
    # -------------------
    deg_v = torch.sparse.sum(H, dim=1).to_dense()

    # -------------------
    # remove isolated nodes
    # -------------------
    mask = deg_v > 0
    keep_idx = mask.nonzero(as_tuple=False).view(-1)

    H2 = H.index_select(0, keep_idx).coalesce()
    N = H2.size(0)

    # -------------------
    # Build node adjacency A = H H^T
    # -------------------
    A = torch.sparse.mm(H2, H2.t()).to_dense()

    # remove self loops
    A.fill_diagonal_(0)

    # -------------------
    # Row-normalize → transition matrix P
    # -------------------
    row_sum = A.sum(dim=1)
    row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))

    P = A / row_sum.unsqueeze(1)

    # -------------------
    # Multi-step Diffusion (Compute P^t)
    # -------------------
    P_t = torch.linalg.matrix_power(P, t)

    # -------------------
    # Zero pad back to original nodes (CRITICAL FIX)
    # -------------------
    P_final = torch.zeros((num_nodes, num_nodes), device=device)
    
    idx_mesh_row, idx_mesh_col = torch.meshgrid(keep_idx, keep_idx, indexing='ij')
    P_final[idx_mesh_row, idx_mesh_col] = P_t

    return P_final




def hypergraph_rw_pe_multiple(H, t=16, rw_type="WE"):
    """
    Hypergraph random walk positional encodings.

    Args:
        H : sparse incidence matrix [num_nodes, num_hyperedges]
        t : walk length
        rw_type : "EN", "EE", "WE"

    Returns:
        node_pe : [num_nodes, t]
    """

    device = H.device
    num_nodes = H.size(0)

    H = H.coalesce()
    row, col = H.indices()

    # --------------------------------------------------
    # node degrees
    # --------------------------------------------------
    deg_v = torch.sparse.sum(H, dim=1).to_dense()

    # remove isolated nodes
    mask = deg_v > 0
    keep = mask.nonzero(as_tuple=False).view(-1)

    H = H.index_select(0, keep).coalesce()
    row, col = H.indices()
    deg_v = deg_v[keep]

    N = H.size(0)
    M = H.size(1)

    # hyperedge sizes
    deg_e = torch.sparse.sum(H, dim=0).to_dense()

    # --------------------------------------------------
    # Build adjacency according to walk type
    # --------------------------------------------------
    A = torch.zeros((N, N), device=device)

    # group nodes per hyperedge
    edge_nodes = [[] for _ in range(M)]
    for r, c in zip(row.tolist(), col.tolist()):
        edge_nodes[c].append(r)

    if rw_type == "EN":
        # equal probability among neighbors
        for nodes in edge_nodes:
            for i in nodes:
                for j in nodes:
                    if i != j:
                        A[i,j] = 1

        # normalize rows later

    elif rw_type == "EE":
        # weight = 1/(|e|-1)
        for e, nodes in enumerate(edge_nodes):
            s = len(nodes)
            if s <= 1:
                continue
            w = 1.0 / (s - 1)
            for i in nodes:
                for j in nodes:
                    if i != j:
                        A[i,j] += w

    elif rw_type == "WE":
        # counts normalized by total weight of incident edges
        for i in range(N):
            weight_sum = 0.0
            neighbor_counts = {}

            for e, nodes in enumerate(edge_nodes):
                if i in nodes:
                    s = len(nodes)
                    weight_sum += (s - 1)

                    for j in nodes:
                        if j == i:
                            continue
                        neighbor_counts[j] = neighbor_counts.get(j,0) + 1

            if weight_sum == 0:
                continue

            for j, c in neighbor_counts.items():
                A[i,j] = c / weight_sum

    else:
        raise ValueError("rw_type must be EN / EE / WE")

    # --------------------------------------------------
    # Convert adjacency → transition matrix
    # --------------------------------------------------
    if rw_type in ["EN","EE"]:
        row_sum = A.sum(dim=1)
        row_sum = torch.where(row_sum>0, row_sum, torch.ones_like(row_sum))
        P = A / row_sum.unsqueeze(1)
    else:
        P = A   # already normalized in WE

    # --------------------------------------------------
    # diffusion powers (cumulative)
    # --------------------------------------------------
    X = torch.eye(N, device=device)

    max_t = t

    diag_list = []

    for t in range(0, max_t):
        X = X @ P
    
        diag_list.append(torch.diag(X))

    pe_small = torch.stack(diag_list, dim=1)

    # --------------------------------------------------
    # pad back removed nodes
    # --------------------------------------------------
    node_pe = torch.zeros(num_nodes, pe_small.size(1), device=device)
    node_pe[mask] = pe_small

    return node_pe


def hypergraph_diffusion_pe(H, t_list=[1, 2, 3]):
    """
    H: torch.sparse_coo_tensor [num_nodes, num_hyperedges]
    Returns:
        node_pe: torch.Tensor [num_nodes, num_nodes * len(t_list)]
    """

    device = H.device
    num_nodes = H.size(0)

    H = H.coalesce()
    idx = H.indices()
    val = H.values()

    # -------------------
    # degrees
    # -------------------
    deg_v = torch.sparse.sum(H, dim=1).to_dense()  # [N]
    deg_e = torch.sparse.sum(H, dim=0).to_dense()  # [M]

    # -------------------
    # remove isolated nodes
    # -------------------
    mask = deg_v > 0
    keep_idx = mask.nonzero(as_tuple=False).view(-1)

    H2 = H.index_select(0, keep_idx).coalesce()   # keep rows
    deg_v = deg_v[keep_idx]

    deg_e = torch.where(deg_e > 0, deg_e, torch.ones_like(deg_e))

    # -------------------
    # Dv^{-1} H
    # -------------------
    dv_inv = 1.0 / deg_v
    row_idx, col_idx = H2.indices()
    new_vals = H2.values() * dv_inv[row_idx]

    DvH = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]),
        new_vals,
        size=H2.size(),
        device=device
    ).coalesce()

    # -------------------
    # Dv^{-1} H De^{-1}
    # -------------------
    de_inv = 1.0 / deg_e
    row_idx, col_idx = DvH.indices()
    new_vals = DvH.values() * de_inv[col_idx]

    P_half = torch.sparse_coo_tensor(
        torch.stack([row_idx, col_idx]),
        new_vals,
        size=DvH.size(),
        device=device
    ).coalesce()

    # -------------------
    # P = Dv^{-1} H De^{-1} H^T
    # -------------------
    P = torch.sparse.mm(P_half, H2.t())   # [N', N']

    # -------------------
    # diffusion powers
    # -------------------
    X = P

    pe_list = []
    
    # for _ in t_list:
    #     X = torch.sparse.mm(X, P)
    #     pe_list.append(X)

    # pe_small = torch.cat(pe_list, dim=1)

    # boolean connectivity
    mask = X.to_dense() > 0

    # allow self attention
    mask.fill_diagonal_(True)

    # convert to additive mask
    attn_mask = torch.where(mask, 0.0, float('-inf'))
    

    
    # ---------------------------------
    # SVD compression (low-rank PE)
    # ---------------------------------
    k = 32  # desired PE dimension

    # # full SVD (stable)
    # U, S, Vh = torch.linalg.svd(pe_small, full_matrices=False)

    # # take top-k components
    # U_k = U[:, :k]
    # S_k = S[:k]

    # V_k = Vh[:k, :].t()

    # S_k_root = torch.sqrt(S_k)

  

    # # final compressed PE
    # pe_small = U_k

    # eigvals, eigvecs = torch.linalg.eigh(pe_small)
    # eigvals,eigvecs = eigvals[:k+1], eigvecs[:, :k+1]
    # eigvals_, eigvecs_ = torch.lobpcg(pe_small, k=k+1, largest=False)
    # #check whether the two methods give similar results
    # print("Eigensolver consistency check (max abs diff):", (eigvals - eigvals_).abs().max().item())
    # print("Eigensolver consistency check (max abs diff):", (eigvecs - eigvecs_).abs().max().item())
    # breakpoint()

    # idx = torch.argsort(eigvals, descending=True)
    # idx = idx[1:k+1]                          # skip top trivial mode
    # pe_small = eigvecs[:, idx]
    

    # U, S, Vh = torch.svd_lowrank(pe_small, q=k)
    # pe_small = U


    # # Proper plotting
    # import matplotlib.pyplot as plt

    # eigvals_np = eigvals.detach().cpu().numpy()
    # eigvecs_np = eigvecs.detach().cpu().numpy()
    # print((eigvals_np < 1e-6).sum())
    # eigvals_np = np.sort(eigvals_np)
    # print("Eigenvalues:", eigvals_np[:10])

    # print("zero eigvals:", (eigvals < 1e-6).sum())
    # print("min:", eigvals.min(), "max:", eigvals.max())


    # # 1) Laplacian spectrum
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(len(eigvals_np)), eigvals_np, marker='o', linewidth=1)
    # plt.xlabel("Eigenvalue index")
    # plt.ylabel("Eigenvalue")
    # plt.title("Hypergraph Laplacian Spectrum")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig("spectrum.png", dpi=200); plt.close()

    # # 2) Optional: embedding view using 2nd and 3rd eigenvectors
    # if eigvecs_np.shape[1] >= 3:
    #     plt.figure(figsize=(5, 5))
    #     plt.scatter(eigvecs_np[:, 1], eigvecs_np[:, 2], s=10, alpha=0.8)
    #     plt.xlabel("eigenvector 1")
    #     plt.ylabel("eigenvector 2")
    #     plt.title("Spectral embedding")
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig("embedding.png", dpi=200); plt.close()

    # -------------------
    # zero pad back to original nodes
    # -------------------
    # node_pe = torch.zeros(num_nodes, pe_small.size(1), device=device)
    # node_pe[mask] = pe_small

    return attn_mask

def attn_mask(H):
    """
    H: torch.sparse_coo_tensor [num_nodes, num_hyperedges]
    Returns:
        the masked attention
    """

    device = H.device

    # Ensure the sparse tensor is in canonical form
    H = H.coalesce()

    # Compute node-to-node connectivity matrix
    P = torch.sparse.mm(H, H.t())  # [num_nodes, num_nodes]

    # Boolean connectivity
    mask = P.to_dense() > 0

    # Allow self-attention
    mask.fill_diagonal_(True)

    # Convert to additive mask
    attn_mask = torch.where(mask, 0.0, float('-inf'))

    return attn_mask



def hypergraph_norm_laplacian_pe(H, k=32):
    """
    Symmetric normalized hypergraph Laplacian positional encoding

    L = I - Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}

    Args:
        H : torch.sparse_coo_tensor  [N, M]
        k : number of eigenvectors

    Returns:
        pe : [N, k] positional encodings
    """

    device = H.device
    N = H.size(0)

    # ----------------------------------
    # coalesce
    # ----------------------------------
    H = H.coalesce()

    # ----------------------------------
    # degrees
    # ----------------------------------
    deg_v = torch.sparse.sum(H, dim=1).to_dense()  # [N]
    deg_e = torch.sparse.sum(H, dim=0).to_dense()  # [M]

    # ----------------------------------
    # remove isolated nodes
    # ----------------------------------
    mask = deg_v > 0
    keep = mask.nonzero(as_tuple=False).view(-1)

    H = H.index_select(0, keep).coalesce()
    deg_v = deg_v[keep]

    # avoid division by zero
    deg_e = torch.where(deg_e > 0, deg_e, torch.ones_like(deg_e))

    # ----------------------------------
    # build Dv^{-1/2} H De^{-1}
    # ----------------------------------
    dv_inv_sqrt = deg_v.pow(-0.5)
    de_inv = deg_e.pow(-1)

    row, col = H.indices()
    val = H.values()

    val = val * dv_inv_sqrt[row]
    val = val * de_inv[col]

    B = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        val,
        size=H.size(),
        device=device
    ).coalesce()

    # ----------------------------------
    # compute P = B H^T Dv^{-1/2}
    # ----------------------------------
    row, col = H.indices()
    val = H.values() * dv_inv_sqrt[row]

    H_norm = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        val,
        size=H.size(),
        device=device
    ).coalesce()

    P = torch.sparse.mm(B, H_norm.t())  # [N', N']

    # ----------------------------------
    # Laplacian
    # ----------------------------------
    n = P.size(0)
    I = torch.eye(n, device=device)
    L = I - P.to_dense()   # eigensolver requires dense
    # L =P.to_dense()   # for regular random walk Laplacian, use P directly

    # ----------------------------------
    # eigenvectors (smallest k)
    # ----------------------------------
    k = min(k, n-1)  # safety
    #check whether L is symmetric
   

    if not torch.allclose(L, L.t(), atol=1e-6):
        print("Warning: L is not symmetric. Using SVD instead of eigendecomposition.")
        U, S, Vh = torch.linalg.svd(L, full_matrices=False)
        pe_small = U[:, :k]
        breakpoint()
    eigvals, eigvecs = torch.linalg.eigh(L)

    if eigvals.is_complex():
        print("Warning: complex eigenvalues detected. Taking real part.")
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        breakpoint()

    # skip first eigenvector (constant)
    # pe_small = eigvecs[:, 1:k+1]
    # idx= torch.argsort(eigvals, descending=True)
    # idx = idx[1:k+1]                          # skip top trivial mode
    # pe_small = eigvecs[:, idx]

    U, S, Vh = torch.linalg.svd(L, full_matrices=False)

    # take top-k components
    U_k = U[:, :k]
    S_k = S[:k]
    pe_small = U_k

    # ----------------------------------
    # pad back removed nodes
    # ----------------------------------
    pe = torch.zeros(N, k, device=device)
    pe[mask] = pe_small

    # # Proper plotting
    # import matplotlib.pyplot as plt

    # eigvals_np = eigvals.detach().cpu().numpy()
    # eigvecs_np = eigvecs.detach().cpu().numpy()
    # print((eigvals_np < 1e-6).sum())
    # eigvals_np = np.sort(eigvals_np)
    # print("Eigenvalues:", eigvals_np[:10])

    # print("zero eigvals:", (eigvals < 1e-6).sum())
    # print("min:", eigvals.min(), "max:", eigvals.max())


    # # 1) Laplacian spectrum
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(len(eigvals_np)), eigvals_np, marker='o', linewidth=1)
    # plt.xlabel("Eigenvalue index")
    # plt.ylabel("Eigenvalue")
    # plt.title("Hypergraph Laplacian Spectrum")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig("spectrum.png", dpi=200); plt.close()

    # # 2) Optional: embedding view using 2nd and 3rd eigenvectors
    # if eigvecs_np.shape[1] >= 3:
    #     plt.figure(figsize=(5, 5))
    #     plt.scatter(eigvecs_np[:, 1], eigvecs_np[:, 2], s=10, alpha=0.8)
    #     plt.xlabel("eigenvector 1")
    #     plt.ylabel("eigenvector 2")
    #     plt.title("Spectral embedding")
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig("embedding.png", dpi=200); plt.close()
    
    return pe

def hypergraph_clique_expansion_laplacian(H, k=32):

    H = H.coalesce()
    device = H.device
    num_nodes = H.size(0)

    # Hyperedge sizes
    edge_sizes = torch.sparse.sum(H, dim=0).to_dense()
    edge_sizes = torch.clamp(edge_sizes, min=1)

    # D_e^{-1}
    De_inv = torch.diag(1.0 / edge_sizes).to(device)

    # Clique expansion adjacency
    A = torch.sparse.mm(H, torch.sparse.mm(De_inv, H.t())).to_dense()

    # Remove self loops
    A.fill_diagonal_(0)

    # Degree
    degrees = A.sum(dim=1)

    # Laplacian
    D = torch.diag(degrees)
    L = D - A

    # Eigenvectors
    k = min(k, num_nodes - 1)
    eigvals, eigvecs = torch.linalg.eigh(L)

    pe = eigvecs[:, 1:k+1]

    return pe



def hypergraph_laplacian_pe(H, k=32):
    """
    Symmetric normalized hypergraph Laplacian positional encoding

    L = I - Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}

    Args:
        H : torch.sparse_coo_tensor  [N, M]
        k : number of eigenvectors

    Returns:
        pe : [N, k] positional encodings
    """

    device = H.device
    N = H.size(0)

    # ----------------------------------
    # coalesce
    # ----------------------------------
    H = H.coalesce()

    # ----------------------------------
    # degrees
    # ----------------------------------
    deg_v = torch.sparse.sum(H, dim=1).to_dense()  # [N]
    deg_e = torch.sparse.sum(H, dim=0).to_dense()  # [M]

    # ----------------------------------
    # remove isolated nodes
    # ----------------------------------
    mask = deg_v > 0
    keep = mask.nonzero(as_tuple=False).view(-1)

    H = H.index_select(0, keep).coalesce()
    deg_v = deg_v[keep]

    # avoid division by zero
    deg_e = torch.where(deg_e > 0, deg_e, torch.ones_like(deg_e))

    # ----------------------------------
    # build Dv^{-1/2} H De^{-1}
    # ----------------------------------
    dv_inv_sqrt = deg_v.pow(-0.5)
    de_inv = deg_e.pow(-1)

    row, col = H.indices()
    val = H.values()

    # val = val * dv_inv_sqrt[row]
    # val = val * de_inv[col]

    B = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        val,
        size=H.size(),
        device=device
    ).coalesce()

    # ----------------------------------
    # compute P = B H^T Dv^{-1/2}
    # ----------------------------------
    row, col = H.indices()
    # val = H.values() * dv_inv_sqrt[row]

    H_norm = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        val,
        size=H.size(),
        device=device
    ).coalesce()

    P = torch.sparse.mm(B, H_norm.t())  # [N', N']

    # ----------------------------------
    # Laplacian
    # ----------------------------------
    n = P.size(0)
    I = torch.eye(n, device=device)
    # L = I - P.to_dense()   # eigensolver requires dense
    L =P.to_dense()   # for regular random walk Laplacian, use P directly

    # ----------------------------------
    # eigenvectors (smallest k)
    # ----------------------------------
    k = min(k, n-1)  # safety
    #check whether L is symmetric
   

    if not torch.allclose(L, L.t(), atol=1e-6):
        print("Warning: L is not symmetric. Using SVD instead of eigendecomposition.")
        U, S, Vh = torch.linalg.svd(L, full_matrices=False)
        pe_small = U[:, :k]
        breakpoint()
    eigvals, eigvecs = torch.linalg.eigh(L)

    if eigvals.is_complex():
        print("Warning: complex eigenvalues detected. Taking real part.")
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        breakpoint()

    # skip first eigenvector (constant)
    pe_small = eigvecs[:, 1:k+1]
    idx= torch.argsort(eigvals, descending=True)
    idx = idx[1:k+1]                          # skip top trivial mode
    pe_small = eigvecs[:, idx]

    # U, S, Vh = torch.linalg.svd(L, full_matrices=False)

    # # take top-k components
    # U_k = U[:, :k]
    # S_k = S[:k]
    # pe_small = U_k

    # ----------------------------------
    # pad back removed nodes
    # ----------------------------------
    pe = torch.zeros(N, k, device=device)
    pe[mask] = pe_small

    # # Proper plotting
    # import matplotlib.pyplot as plt

    # eigvals_np = eigvals.detach().cpu().numpy()
    # eigvecs_np = eigvecs.detach().cpu().numpy()
    # print((eigvals_np < 1e-6).sum())
    # eigvals_np = np.sort(eigvals_np)
    # print("Eigenvalues:", eigvals_np[:10])

    # print("zero eigvals:", (eigvals < 1e-6).sum())
    # print("min:", eigvals.min(), "max:", eigvals.max())


    # # 1) Laplacian spectrum
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(len(eigvals_np)), eigvals_np, marker='o', linewidth=1)
    # plt.xlabel("Eigenvalue index")
    # plt.ylabel("Eigenvalue")
    # plt.title("Hypergraph Laplacian Spectrum")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig("spectrum.png", dpi=200); plt.close()

    # # 2) Optional: embedding view using 2nd and 3rd eigenvectors
    # if eigvecs_np.shape[1] >= 3:
    #     plt.figure(figsize=(5, 5))
    #     plt.scatter(eigvecs_np[:, 1], eigvecs_np[:, 2], s=10, alpha=0.8)
    #     plt.xlabel("eigenvector 1")
    #     plt.ylabel("eigenvector 2")
    #     plt.title("Spectral embedding")
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig("embedding.png", dpi=200); plt.close()
    
    return pe

# def hypergraph_diffusion_pe(H, t_list=[1,2,3], k=64):
#     """
#     k = number of random probes (PE dimension)
#     """
#     n = H.shape[0]

#     deg_v = np.array(H.sum(1)).flatten()
#     deg_e = np.array(H.sum(0)).flatten()
#     deg_e[deg_e == 0] = 1

#     Dv_inv = sp.diags(1.0 / deg_v)
#     De_inv = sp.diags(1.0 / deg_e)

#     P = (Dv_inv @ H @ De_inv @ H.T).tocsr()

#     # top-k eigenvectors (largest magnitude)
#     vals, vecs = eigsh(P, k=k, which='LM')

#     pe_list = []
#     for t in t_list:
#         pe_list.append(vecs * (vals ** t))

#     return np.concatenate(pe_list, axis=1)


# def hypergraph_diffusion_pe(
#     H,
#     k=128,
#     t_list=(1, 2, 4, 8),
#     normalize=True,
#     reorthogonalize=False,
#     seed=0
# ):
#     """
#     Scalable hypergraph random-walk diffusion positional encoding.

#     Args:
#         H: scipy.sparse incidence matrix [num_nodes, num_hyperedges]
#         k: probe dimension
#         t_list: diffusion steps (multi-scale)
#         normalize: row-normalize after each step
#         reorthogonalize: QR on final PE (optional, expensive)
#         seed: random seed for reproducibility

#     Returns:
#         pe: [num_nodes, k * len(t_list)] numpy array
#     """

#     np.random.seed(seed)
#     n, m = H.shape

#     # --- degrees ---
#     deg_v = np.array(H.sum(1)).flatten()
#     deg_e = np.array(H.sum(0)).flatten()
#     deg_e[deg_e == 0] = 1.0
#     deg_v[deg_v == 0] = 1.0

#     # --- random probes ---
#     X = np.random.randn(n, k).astype(np.float32)

#     pe_list = []
#     t_prev = 0

#     for t in t_list:
#         # apply diffusion (t - t_prev) times
#         for _ in range(t - t_prev):
#             # node → hyperedge
#             X = H.T @ X
#             X = X / deg_e[:, None]

#             # hyperedge → node
#             X = H @ X
#             X = X / deg_v[:, None]

#             if normalize:
#                 X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

#         pe_list.append(X.copy())
#         t_prev = t

#     pe = np.concatenate(pe_list, axis=1)

#     # --- optional re-orthogonalization ---
#     if reorthogonalize:
#         pe, _ = np.linalg.qr(pe)

#     return pe

from scipy.sparse.linalg import eigsh

# def hypergraph_diffusion_pe(H, t_list=[1,2,3], k=64):
#     n = H.shape[0]

#     deg_v = np.array(H.sum(1)).flatten()
#     deg_e = np.array(H.sum(0)).flatten()
#     deg_e[deg_e == 0] = 1

#     Dv_inv = sp.diags(1.0 / deg_v)
#     De_inv = sp.diags(1.0 / deg_e)

#     P = (Dv_inv @ H @ De_inv @ H.T).tocsr()

#     # top-k eigenvectors (largest magnitude)
#     vals, vecs = eigsh(P, k=k, which='LM')

#     pe_list = []
#     for t in t_list:
#         pe_list.append(vecs * (vals ** t))

#     return np.concatenate(pe_list, axis=1)


def hypergraph_co_membership(edge_index):
    src, dst = edge_index
    num_nodes = int(src.max()) + 1 
    E = int(dst.max()) + 1
   
    H = torch.zeros(num_nodes, E, device=src.device)
    H[src, dst] = 1

    C = (H @ H.T) > 0     # [N, N] boolean
    C.fill_diagonal_(0)
    return C

def sample_pairs(C, num_samples=4000):
    pos = torch.nonzero(C)
    idx = torch.randint(0, pos.size(0), (num_samples,))
    pos = pos[idx]

    N = C.size(0)
    neg_u = torch.randint(0, N, (num_samples,))
    neg_v = torch.randint(0, N, (num_samples,))
    neg = torch.stack([neg_u, neg_v], dim=1)
    

    return pos, neg


def hg_ortho_pe(edge_index, pe_dim=1024):
    "creates a random orthogonal Gaussian matrix and then does QR decomposition and returns Q"
    src, dst = edge_index
    num_nodes = int(src.max()) + 1  
    E = int(dst.max()) + 1
    H = torch.zeros(num_nodes, E, device=src.device)
    H[src, dst] = 1
    G = torch.randn(E, pe_dim, device=src.device)
    Q, R = torch.linalg.qr(G)
    pe = H @ Q
    return pe
