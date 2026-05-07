# Hypergraph Neural Networks with Spectral Methods

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of hypergraph neural network models with advanced spectral and diffusion-based techniques.

## Introduction

This repository provides a comprehensive implementation of modern hypergraph neural network architectures designed to capture complex higher-order relationships in data. Our framework combines several advanced techniques:

- **Spectral Methods**: Normalized Laplacian positional encodings, clique expansion techniques
- **Diffusion Processes**: Hypergraph random walk probability matrices and multi-step diffusion
- **Equivariant Operators**: Equivariant set operations and structure-aware message passing
- **Transformer Architectures**: Attention-based mechanisms for hypergraph processing

The codebase supports multiple model variants and datasets, enabling researchers to benchmark and develop new hypergraph learning methods.

## Getting Started

### Dependency

To run our code, the following Python libraries are required:

```
pytorch 1.8.0+
torch-geometric
torch-scatter
torch-sparse
torch-cluster
numpy
scipy
dask
matplotlib
seaborn
pandas
configargparse
```

### Data Preparation

Download our preprocessed datasets. We provide multiple download options:


- Download from [Google Drive Folder]([https://drive.google.com/drive/folders/YOUR_FOLDER_ID](https://drive.google.com/drive/folders/1vv-KmOUNuGqotZLtvEaX_Mck1xb--sqE?usp=sharing))
- Extract the datasets to the `raw_data/` directory

The directory structure should look like:
```
an/
  <source code files>
  ...
  raw_data/
    cocitation/
      cora/
      citeseer/
      pubmed/
    coauthorship/
      coauthor_cora/
      coauthor_dblp/
    senate-committees-100/
    house-committees-100/
    ...
```

**Supported Datasets:**
- Cora, Citeseer, Pubmed (Cocitation networks)
- Coauthor-Cora, Coauthor-DBLP (Coauthorship networks)
- Senate Committees, House Committees (Congressional networks)
- Congress Bills (Legislative data)
- ModelNet40, NTU 2012 (3D/Action data)

## Training

To train a hypergraph neural network model, use the command below:

```
python train.py --method <model_name> --dname <dataset_name> --All_num_layers <num_layers> \
  --MLP_num_layers <mlp_depth> --MLP_hidden <mlp_hidden> \
  --Classifier_num_layers <classifier_depth> --Classifier_hidden <classifier_hidden> \
  --lr <learning_rate> --wd <weight_decay> \
  --epochs <num_epochs> --cuda <cuda_id> \
  --data_dir <data_path> --raw_data_dir <raw_data_path>
```

### Hyperparameter Configuration

- `--MLP_num_layers`: Number of layers in internal MLPs
- `--MLP_hidden`: Hidden dimension for internal MLPs
- `--MLP2_num_layers`, `--MLP3_num_layers`: Separate MLP depths (optional)
- `--Classifier_num_layers`: Number of classifier layers
- `--Classifier_hidden`: Classifier hidden dimension
- `--n_layers`: Number of transformer layers
- `--hops`: Number of neighborhood hops
- `--heads`: Number of attention heads
- `--dropout`: Dropout rate
- `--activate`: Activation function (relu, prelu, Id)
- `--normalization`: Normalization type (bn, ln, None)
- `--aggregate`: Aggregation type (sum, mean)

### Supported Models

The framework supports the following model architectures:

<details>

<summary>EDGNN (Equivariant)</summary>

```
python train.py --method EDGNN --dname cora --All_num_layers 1 --MLP_num_layers 0 \
  --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 1 \
  --MLP_hidden 256 --Classifier_hidden 256 --aggregate mean \
  --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 500 --runs 10 \
  --cuda 0 --data_dir ./data/cora --raw_data_dir ./raw_data/cocitation/cora
```

</details>

<details>

<summary>AllSetTransformer</summary>

```
python train.py --method AllSetTransformer --dname cora --All_num_layers 2 \
  --MLP_num_layers 1 --MLP_hidden 128 --Classifier_num_layers 1 \
  --Classifier_hidden 128 --lr 0.001 --wd 0 --epochs 500 --runs 10 \
  --cuda 0 --data_dir ./data/cora --raw_data_dir ./raw_data/cocitation/cora
```

</details>

<details>

<summary>HSAT (Hypergraph Structure-Aware Transformer)</summary>

```
python train.py --method HSAT --dname cora --All_num_layers 1 \
  --n_layers 1 --hops 3 --heads 8 --hidden_dim 256 --Classifier_hidden 256 \
  --lr 0.0005 --wd 0.00001 --epochs 600 --runs 15 \
  --cuda 0 --data_dir ./data/cora --raw_data_dir ./raw_data/cocitation/cora
```

</details>

<details>

<summary>HNHN</summary>

```
python train.py --method HNHN --dname cora --All_num_layers 2 \
  --MLP_num_layers 1 --MLP_hidden 128 --Classifier_num_layers 1 \
  --HNHN_alpha -1.5 --HNHN_beta -0.5 --lr 0.001 --wd 0 --epochs 500 \
  --cuda 0 --data_dir ./data/cora --raw_data_dir ./raw_data/cocitation/cora
```

</details>

<details>

<summary>HyperGCN</summary>

```
python train.py --method HyperGCN --dname cora --All_num_layers 2 \
  --MLP_num_layers 1 --MLP_hidden 128 --Classifier_num_layers 1 \
  --HyperGCN_mediators --HyperGCN_fast --lr 0.001 --wd 0 --epochs 500 \
  --cuda 0 --data_dir ./data/cora --raw_data_dir ./raw_data/cocitation/cora
```

</details>

<details>

<summary>HyperSAGE</summary>

```
python train.py --method HyperSAGE --dname cora --All_num_layers 2 \
  --MLP_num_layers 1 --MLP_hidden 128 --HyperSAGE_power 1.0 \
  --HyperSAGE_num_sample 100 --lr 0.001 --wd 0 --epochs 500 \
  --cuda 0 --data_dir ./data/cora --raw_data_dir ./raw_data/cocitation/cora
```

</details>

## Core Modules

### Hypergraph Processing (`hypergraph.py`)

Provides comprehensive hypergraph manipulation utilities:

- **Spectral Methods**:
  - `hypergraph_norm_laplacian_pe()`: Symmetric normalized Laplacian positional encodings
  - `hypergraph_clique_expansion_laplacian()`: Clique-expanded Laplacian embeddings
  - `hypergraph_laplacian_pe()`: Standard Laplacian PE

- **Diffusion Operators**:
  - `hypergraph_rw_pe_multiple()`: Multi-type random walk positional encodings (EN, EE, WE)
  - `node_node_rw_prob_pe()`: Node-level random walk transition matrices
  - `hypergraph_diffusion_pe()`: Multi-step diffusion with attention masking

- **Graph Conversions**:
  - `clique_expansion_edge_index()`: Convert hypergraph to clique-expanded graphs
  - `clique_expansion_edge_index_zh()`: Zhou-style weighted clique expansion
  - `hyperedge_index_to_incidence()`: Build incidence matrices

- **Utilities**:
  - `edge_index_to_dict()`: Convert PyG format to hypergraph dictionaries
  - `attn_mask()`: Compute connectivity-based attention masks
  - `hg_ortho_pe()`: Orthogonal random positional encodings

### Dataset Management (`datasets.py`)

- Multi-format dataset loading (cocitation, coauthorship, heterogeneous)
- Automatic feature augmentation with noise injection
- Train/validation/test splitting
- Built-in transforms for data preprocessing

### Training Framework (`training.py`, `train.py`)

- Flexible training pipeline with early stopping
- Multi-run evaluation with statistical averaging
- Learning rate scheduling and optimizer setup
- Gate visualization (feature and structure gates)
- Support for 11+ model architectures

### Model Implementation (`models/`)

Implements state-of-the-art architectures:
- AllSetTransformer, AllDeepSets
- HNHN, HCHA
- HyperGCN, HyperSAGE
- LEGCN, UniGCNII
- HyperND, EquivSetGNN
- GlobalHyperNodeTransformer (HSAT)

## Configuration Files

Create YAML configs for reproducible experiments:

```yaml
# config.yaml
method: HSAT
dname: cora
All_num_layers: 1
n_layers: 1
hops: 3
heads: 8
hidden_dim: 256
Classifier_hidden: 256
lr: 0.0005
wd: 0.00001
epochs: 600
runs: 15
cuda: 0
data_dir: ./data/cora
raw_data_dir: ./raw_data/cocitation/cora
```

Then run:
```
python train.py --config config.yaml
```

## Citation

If you use this work in your research, please cite:

```
@inproceedings{wang2022equivariant,
  title={Equivariant Hypergraph Diffusion Neural Operators},
  author={Wang, Peihao and Yang, Shenghao and Liu, Yunyu and Wang, Zhangyang and Li, Pan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation builds upon foundational work in hypergraph learning and is compatible with PyTorch Geometric ecosystem. Key references include:

- Ding et al. (2021) - HyperGCN
- Feng et al. (2019) - Hypergraph Neural Networks
- Wang et al. (2023) - Equivariant Hypergraph Diffusion Neural Operators
