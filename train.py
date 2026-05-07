import gc
import os
from multiprocessing import dummy
import random
import time
import warnings

from sympy import deg

import configargparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from models import SetGNN, HCHA, HNHN, HyperGCN, HyperSAGE, \
    LEGCN, UniGCNII, HyperND, EquivSetGNN, GlobalHyperNodeTransformer

import data_processing, datasets, hypergraph, training
from utils import OneBatch_






@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, loss_fn, return_out=False):
    model.eval()
    
    out = model(data)
    out = F.log_softmax(out, dim=1)

    train_acc = evaluator.eval(data.y[split_idx['train']], out[split_idx['train']])['acc']
    valid_acc = evaluator.eval(data.y[split_idx['valid']], out[split_idx['valid']])['acc']
    test_acc = evaluator.eval(data.y[split_idx['test']], out[split_idx['test']])['acc']
    ret_list = [train_acc, valid_acc, test_acc]

    train_loss = loss_fn(out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = loss_fn(out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = loss_fn(out[split_idx['test']], data.y[split_idx['test']])
    ret_list += [train_loss, valid_loss, test_loss]

    if return_out:
        ret_list.append(out)

    return ret_list

def main(args):
    torch.cuda.empty_cache()#release unoccupied memory
    gc.collect()# garbage collection

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')   
    args.device = device

    prep = []

    begin_prep = time.time()

    if args.method not in ['HyperGCN', 'HyperSAGE']:
        transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
    else:
        transform = None
    # transform = torch_geometric.transforms.Compose([datasets.AddHypergraphSelfLoops()])
    data = datasets.HypergraphDataset(root=args.data_dir, name=args.dname,
        feature_noise='1', transform=transform, args=args)._data
    
    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = SetGNN.norm_contruction(data, option=args.normtype)
    elif args.method == 'HNHN':
        data = HNHN.generate_norm(data, args)
    elif args.method == 'HyperSAGE':
        data = HyperSAGE.generate_hyperedge_dict(data)
    elif args.method == 'LEGCN':
        data = LEGCN.line_expansion(data)
    data = data.to(device)
   
    
    data.H =torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.size(1), device=device),   size=(data.num_nodes, data.num_hyperedges)
).coalesce().to(device)
    edge_ind = data.edge_index
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = data_processing.rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
   
   
    dataset = hypergraph.edge_index_to_dict(data)
    hg_dict = dataset['hypergraph']
    nodes = list(np.arange(0, dataset['n'], 1))
    hg= OneBatch_(dataset, args).batch_for_targets(nodes) 
    
    if args.method == 'AllSetTransformer':
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method == 'AllDeepSets':
        args.AllSet_PMA = False
        args.aggregate = 'add'
        if args.AllSet_LearnMask:
            model = SetGNN(data.num_features, data.num_classes, args, data.norm)
        else:
            model = SetGNN(data.num_features, data.num_classes, args)
    elif args.method in ['HGNN', 'HCHA']:
        model = HCHA(data.num_features, data.num_classes, args)
    elif args.method in 'HNHN':
        model = HNHN(data.num_features, data.num_classes, args)
    elif args.method in 'HyperGCN':
        model = HyperGCN(data.num_features, data.num_classes, args)
    elif args.method == 'HyperSAGE':
        model = HyperSAGE(data.num_features, data.num_classes, args)
    elif args.method == 'LEGCN':
        model = LEGCN(data.num_features, data.num_classes, args)
    elif args.method == 'UniGCNII':
        model = UniGCNII(data.num_features, data.num_classes, args)
    elif args.method == 'HyperND':
        model = HyperND(data.num_features, data.num_classes, args)
    elif args.method == 'EDGNN':
        model = EquivSetGNN(data.num_features, data.num_classes, args)
    elif args.method == 'HSAT':
        model = GlobalHyperNodeTransformer(
       in_channels=data.x.shape[1],
       hidden_channels=args.hidden_dim,
       out_channels=data.num_classes,
       n_layers=args.n_layers,
       heads=args.heads,
        dropout_rate=args.dropout,
        edge_index=edge_ind,
        hops=args.hops,
        sparse_h = data.H,
        hg = hg,
        args = args
    )
    else:
        raise ValueError(f'Undefined model name: {args.method}')


    viz_degrees = []
    viz_g_proc = []
    viz_g_se = []

   
   
    model = model.to(device)
    
    # model = torch.compile(model, mode="max-autotune")
    logger = training.Logger(args.runs, args)
    print("# Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    loss_fn = nn.NLLLoss()
    evaluator = training.NodeClsEvaluator()
    runtime_list = []
    run_time_list = []
    for run in range(args.runs):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)

        
        torch.cuda.empty_cache()
        gc.collect()
        
        total_train = []
        eval_time= []
        model.reset_parameters()
        optimizer, lr_scheduler, early_stopping = training.setup_training_components(model, args)


        best_val = float('-inf')
        for epoch in range(args.epochs):
            # Training loop
            start_epoch = time.time()
           
            model.train()
            optimizer.zero_grad()
            out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = loss_fn(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            end_epoch = time.time()
            total_train.append(end_epoch - start_epoch)
            ev_start = time.time()
            # Evaluation and logging
            result = evaluate(model, data, split_idx, evaluator, loss_fn)
            logger.add_result(run, *result[:3])
            ev_end = time.time()
            eval_time.append(ev_end - ev_start)

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Run: {run:02d}, '
                    f'Epoch: {epoch:02d}, '
                    f'Train Loss: {loss:.4f}, '
                    f'Valid Loss: {result[4]:.4f}, '
                    f'Test Loss: {result[5]:.4f}, '
                    f'Train Acc: {100 * result[0]:.2f}%, '
                    f'Valid Acc: {100 * result[1]:.2f}%, '
                    f'Test Acc: {100 * result[2]:.2f}%')
            if early_stopping.check([result[1].item(), result[4].item()], epoch):
                break
        
        model.load_state_dict(early_stopping.best_state)
        evaluate(model, data, split_idx, evaluator, loss_fn)

        model.eval()
        
        k = args.hops
        if args.extract_subgraph == False and k > 1:

            
            with torch.no_grad():
               
                _ = model(data) 
                
                run_degrees = model.saved_node_degree
                run_g_proc = model.saved_g_proc
                run_g_se = model.saved_g_se
                
            
        end_time = time.time()
        run_time_list.append(np.mean(total_train))
        runtime_list.extend(total_train)

  
   
    print(f"Average evaluation time per epoch: {np.mean(eval_time):.4f} seconds")
    print(f"Average training time per epoch: {np.mean(run_time_list):.4f} seconds")
    print(f"Standard deviation of training time per epoch: {np.std(run_time_list):.4f} seconds")
    print(f"Average runtime across all runs and epochs: {np.mean(runtime_list):.4f} seconds")
    print(f"Standard deviation of runtime across all runs and epochs: {np.std(runtime_list):.4f} seconds")
    
    logger.print_statistics()
    if args.extract_subgraph == False and k> 1:
        print("Generating gate visualization...")
       
        print(f"Average g_proc across nodes: {np.mean(run_g_proc):.4f}")
        print(f"Average g_se across nodes: {np.mean(run_g_se):.4f}")
        
        
        plot_gates_by_split(run_degrees, run_g_proc, run_g_se, split_idx, data.num_nodes, data=args.dname)




# def plot_gates_by_split(degrees, g_proc, g_se, split_idx, num_nodes, data= None):
#     # 1. Create a label array for all nodes
#     node_splits = np.array(['Unassigned'] * num_nodes, dtype=object)
    
#     # Map indices to their respective splits
#     # Note: Ensure split_idx tensors are moved to CPU for numpy indexing
#     node_splits[split_idx['train'].cpu().numpy()] = 'Train'
#     node_splits[split_idx['valid'].cpu().numpy()] = 'Validation'
#     node_splits[split_idx['test'].cpu().numpy()] = 'Test'

#     # 2. Build a DataFrame
#     df = pd.DataFrame({
#         'Node Degree': degrees,
#         'g_proc': g_proc,
#         'g_se': g_se,
#         'Split': node_splits
#     })

#     # Remove unassigned nodes (if your dataset doesn't use all nodes)
#     df = df[df['Split'] != 'Unassigned']

#     # 3. Plotting
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#     # Plot g_proc
#     sns.scatterplot(data=df, x='Node Degree', y='g_proc', hue='Split', 
#                     alpha=0.6, ax=axes[0], palette='Set1')
#     axes[0].set_title('g_proc (Process Gate) vs Node Degree')
#     axes[0].set_xscale('log') # Use log scale if degrees vary wildly

#     # Plot g_se
#     sns.scatterplot(data=df, x='Node Degree', y='g_se', hue='Split', 
#                     alpha=0.6, ax=axes[1], palette='Set1')
#     axes[1].set_title('g_se (Structure Gate) vs Node Degree')
#     axes[1].set_xscale('log')

#     plt.tight_layout()
#     if data is None:
#         filename = "img.png"
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Successfully saved plot to {filename}")
#     else:
#         filename = f"img_{data}.png"
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Successfully saved plot to {filename}")    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_gates_by_split(degrees, g_fe, g_se, split_idx, num_nodes, data=None):
    # ── 0. Global style (NeurIPS Caption Matching) ───────────────────────────
    mpl.rcParams.update({
        'font.family':           'serif',
        'font.serif':            ['Times New Roman', 'Times', 'serif'], # Force Times-like font
        'font.size':              9,  # Base size matches NeurIPS caption
        'axes.labelsize':         9,  # Exact match to caption size
        'axes.labelweight':      'normal',
        'axes.titlesize':         9,
        'xtick.labelsize':        8,  # Slightly smaller than labels
        'ytick.labelsize':        8,
        'xtick.major.size':       4,
        'ytick.major.size':       4,
        'xtick.minor.size':       2.5,
        'ytick.minor.size':       2.5,
        'xtick.major.width':      0.8,
        'ytick.major.width':      0.8,
        'xtick.minor.width':      0.5,
        'ytick.minor.width':      0.5,
        'legend.fontsize':        9,  # Match caption size
        'legend.title_fontsize':  9,
        'axes.linewidth':         0.8,
        'pdf.fonttype':          42,  # Strict vector font rendering
        'ps.fonttype':           42,
    })

    # ── 1. Build split label array ────────────────────────────────────────────
    node_splits = np.array(['Unassigned'] * num_nodes, dtype=object)
    node_splits[split_idx['train'].cpu().numpy()] = 'Train'
    node_splits[split_idx['valid'].cpu().numpy()] = 'Validation'
    node_splits[split_idx['test'].cpu().numpy()]  = 'Test'

    # ── 2. DataFrame ──────────────────────────────────────────────────────────
    df = pd.DataFrame({
        'Node Degree': degrees,
        'g_fe':      g_fe,
        'g_se':        g_se,
        'Split':       node_splits,
    })
    df = df[df['Split'] != 'Unassigned'].copy()

    # ── 3. Log-space jitter ───────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    jitter_scale = 0.025  
    df['Degree_jittered'] = df['Node Degree'] * np.exp(
        rng.normal(0, jitter_scale, size=len(df))
    )

    # ── 4. Palette ────────────────────────────────────────────────────────────
    # High-contrast, colorblind-friendly Okabe-Ito palette
    palette = {
        'Train':      '#0072B2',  # Blue
        'Validation': '#D55E00',  # Vermillion 
        'Test':       '#009E73',  # Bluish Green
    }
    split_order = ['Train', 'Validation', 'Test']   # back → front
    zorder_map  = {'Train': 1, 'Validation': 3, 'Test': 2}

    # ── 5. Figure — NeurIPS single-column, 2 rows ─────────────────────────────
    # Single column width is exactly 3.25 inches to 3.5 inches in NeurIPS
    fig, axes = plt.subplots(
        2, 1,
        figsize=(3.25, 4.8), # Tighter figure to save vertical space
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.08, bottom=0.22)

    plots = [
        (axes[0], 'g_fe', r'Feature Gate ($g_{fe}$)'),
        (axes[1], 'g_se',   r'Structure Gate ($g_{se}$)'),
    ]

    # ── 6. Draw splits back → front ───────────────────────────────────────────
    for split in split_order:
        sub = df[df['Split'] == split]
        for ax, ycol, _ in plots:
            ax.scatter(
                sub['Degree_jittered'],
                sub[ycol],
                color=palette[split],
                alpha=0.5,  # Bumped slightly to 0.5 to offset smaller marker size
                s=8,        
                linewidths=0,
                label=split,
                zorder=zorder_map[split],
            )

    # ── 7. Axes formatting ────────────────────────────────────────────────────
    for i, (ax, ycol, ylabel) in enumerate(plots):
        ax.set_xscale('log')
        ax.set_ylabel(ylabel)
        
        # Only set the x-label on the bottom plot
        if i == 1:
            ax.set_xlabel('Node Degree')
            
        ax.grid(axis='y', linewidth=0.3, alpha=0.5, linestyle='--', color='grey')
        ax.grid(axis='x', linewidth=0.3, alpha=0.3, linestyle=':', color='grey')
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(False, which='minor')

    # ── 8. Shared legend — bottom center, single row ──────────────────────────
    legend_handles = [
        mpl.lines.Line2D(
            [], [],
            marker='o',
            color='w',
            markerfacecolor=palette[s],
            markersize=7,
            label=s,
            linewidth=0,
        )
        for s in split_order
    ]
    fig.legend(
        handles=legend_handles,
        title='Split',
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        edgecolor='#dddddd',
        columnspacing=1.2,
        handletextpad=0.3,
    )

    # ── 9. Save ───────────────────────────────────────────────────────────────
    suffix   = f"_{data}" if data is not None else ""
    pdf_path = f"img{suffix}.pdf"
    
    # Save purely as PDF to guarantee vector graphics in LaTeX
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved → {pdf_path}")
    plt.close(fig)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--config', is_config_file=True)

    # Dataset specific arguments
    parser.add_argument('--dname', default='walmart-trips-100')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--extract_subgraph', action='store_true')

    # Training specific hyperparameters
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--runs', default=15, type=int)
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--input_dropout', default=0.2, type=float)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--wd', default=0.00001, type=float)
    parser.add_argument('--display_step', type=int, default=20)


    parser.add_argument('--n_layers', default = 1, type=int, help='number of transformer layers')
    parser.add_argument('--hops', default = 3, type=int, help='number of hops')
    parser.add_argument('--heads', default = 8, type=int, help='number of heads for transformer')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dimension of mlps')
    parser.add_argument('--Classifier_hidden', default=256,type=int)
     # Model common hyperparameters
    parser.add_argument('--method', default='HSAT', help='model type')
    parser.add_argument('--All_num_layers', default=2, type=int, help='number of basic blocks')
    parser.add_argument('--MLP_num_layers', default=2, type=int, help='layer number of mlps')
    parser.add_argument('--MLP_hidden', default=64, type=int, help='hidden dimension of mlps')
    parser.add_argument('--Classifier_num_layers', default=2,
                        type=int)  # How many layers of decoder
    
    parser.add_argument('--aggregate', default='mean', choices=['sum', 'mean'])
    parser.add_argument('--normalization', default='ln', choices=['bn','ln','None'])
    parser.add_argument('--activation', default='relu', choices=['Id','relu', 'prelu'])
    
    # Args for EDGNN
    parser.add_argument('--MLP2_num_layers', default=-1, type=int, help='layer number of mlp2')
    parser.add_argument('--MLP3_num_layers', default=-1, type=int, help='layer number of mlp3')
    parser.add_argument('--edconv_type', default='EquivSet', type=str, choices=['EquivSet', 'JumpLink', 'MeanDeg', 'Attn', 'TwoSets'])
    parser.add_argument('--restart_alpha', default=0.5, type=float)

    # Args for AllSet
    parser.add_argument('--AllSet_input_norm', default=True)
    parser.add_argument('--AllSet_GPR', action='store_false')  # skip all but last dec
    parser.add_argument('--AllSet_LearnMask', action='store_false')
    parser.add_argument('--AllSet_PMA', action='store_true')
    parser.add_argument('--AllSet_num_heads', default=1, type=int)
    # Args for CEGAT
    parser.add_argument('--output_heads', default=1, type=int)  # Placeholder
    # Args for HyperGCN
    parser.add_argument('--HyperGCN_mediators', action='store_true')
    parser.add_argument('--HyperGCN_fast', action='store_true')
    # Args for HyperSAGE
    parser.add_argument('--HyperSAGE_power', default=1., type=float)
    parser.add_argument('--HyperSAGE_num_sample', default=100, type=int)
    # Args for HNHN
    parser.add_argument('--HNHN_alpha', default=-1.5, type=float)
    parser.add_argument('--HNHN_beta', default=-0.5, type=float)
    parser.add_argument('--HNHN_nonlinear_inbetween', default=True, type=bool)
    # Args for HCHA
    parser.add_argument('--HCHA_symdegnorm', action='store_true')
    # Args for UniGNN
    parser.add_argument('--UniGNN_use_norm', action="store_true", help='use norm in the final layer')
    parser.add_argument('--UniGNN_degV', default = 0)
    parser.add_argument('--UniGNN_degE', default = 0)
    # Args for HyperND
    parser.add_argument('--HyperND_ord', default = 1., type=float)
    parser.add_argument('--HyperND_tol', default = 1e-4, type=float)
    parser.add_argument('--HyperND_steps', default = 100, type=int)

    parser.set_defaults(add_self_loop=True)
    parser.set_defaults(exclude_self=False)
    parser.set_defaults(struct_data=False)
    parser.set_defaults(extract_subgraph=False)
    parser.set_defaults(AllSet_GPR=False)
    parser.set_defaults(AllSet_LearnMask=False)
    parser.set_defaults(AllSet_PMA=True)  # True: Use PMA. False: Use Deepsets.
    parser.set_defaults(HyperGCN_mediators=True)
    parser.set_defaults(HyperGCN_fast=True)
    parser.set_defaults(HCHA_symdegnorm=False)
  
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    main(args)
