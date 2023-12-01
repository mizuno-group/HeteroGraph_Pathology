# -*- coding: utf-8 -*-
"""
Created on 2023-11-01 (Wed) 17:11:07

heterograph builders

@author: I.Azuma
"""
# %%
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD

from PIL import Image

import dgl
import torch

import sys
sys.path.append('/workspace/home/azuma/github/HeteroGraph_Pathology')
from _utils import graph_builders,cell_feature_extractor,heterograph_builders,visualizers

# %%------------------------------------------------------
# for cell-tissue heterogeneous graph
def cg_from_hovernet(image_path = '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Test/Images/test_10.png',
                     mat_path = '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/mat/test_10.mat',
                     json_path = '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/json/test_10.json',neighbor_k=5,thresh=50):
    # 0. image
    image = np.array(Image.open(image_path))
    # 1. instance map
    inst_map = sio.loadmat(mat_path)['inst_map']
    # 2. node feature
    cfe = cell_feature_extractor.CellFeatureExtractor(mat_path=mat_path,json_path=json_path)
    cfe.load_data()
    node_feature = cfe.conduct()
    node_feature = node_feature[1::] # avoid background
    # 3. centroids
    with open(json_path) as json_file:
        info = json.load(json_file)
    info = info['nuc']
    centroids = np.empty((len(info), 2))
    for i,k in enumerate(info):
        cent = info[k]['centroid']
        centroids[i,0] = int(round(cent[0]))
        centroids[i,1] = int(round(cent[1]))
    # cell type label
    type_list = []
    for i,k in enumerate(info):
        type_list.append(info[k]['type'])

    dat = graph_builders.CentroidsKNNGraphBuilder(k=neighbor_k, thresh=thresh, add_loc_feats=False)
    cell_graph = dat.process(instance_map=inst_map,features=node_feature,centroids=centroids)

    return cell_graph, type_list

def multiimage_tissue_cell_heterograph(image_path_list,mat_path_list,json_path_list,tissue_feature_list,superpixel_list,true_label_list,feature_dim=32):
    """_summary_

    Args:
        image_path_list (_type_):
            image_path_list = [
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_6.png',
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_10.png',
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Images/train_20.png',
                '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Test/Images/test_10.png'
            ]
        mat_path_list (_type_):
            mat_path_list = [
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/mat/train_6.mat',
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/mat/train_10.mat',
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/mat/train_20.mat',
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/test/mat/test_10.mat'
            ]
        json_path_list (_type_):
            json_path_list = [
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/json/train_6.json',
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/json/train_10.json',
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/train/json/train_20.json',
                '/workspace/mnt/data1/Azuma/Pathology/results/HoverNet_on_ConSeP/pannuke_new_feature/test/json/test_10.json'
            ]
        tissue_feature_list (_type_):
            tissue_feature_list = [
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/train_6_tissue_features.pkl',
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/train_10_tissue_features.pkl',
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/train_20_tissue_features.pkl',
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/test_10_tissue_features.pkl'
        ]
        superpixel_list (_type_):
            superpixel_list = [
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/trian_6_superpixel.pkl',
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/trian_10_superpixel.pkl',
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/trian_20_superpixel.pkl',
            '/workspace/home/azuma/HeteroGraph_Pathology/231031_models_trial/231114_tissue_feature/results/test_10_superpixel.pkl',
        ]
        true_label_list (_type_):
            true_label_list = [
            '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Labels/train_6.mat',
            '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Labels/train_10.mat',
            '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Train/Labels/train_20.mat',
            '/workspace/mnt/data1/Azuma/Pathology/datasource/consep/CoNSeP/Test/Labels/test_10.mat',
        ]
        feature_dim (int, optional): _description_. Defaults to 32.

    """
    t_max = 0
    c_max = 0

    final_tissue_labels = []
    final_cell_labels = []
    final_cci_s = []
    final_cci_d = []
    final_estimated_type = [] # estimated type list
    final_true_type = []
    for idx in range(4):
        # load cell graph and estimated type list
        cell_graph, type_list = heterograph_builders.cg_from_hovernet(image_path=image_path_list[idx],mat_path=mat_path_list[idx],json_path=json_path_list[idx])
        # obtain initial cell and tissue feature
        cell_feature = cell_graph.ndata['feat']
        tissue_feature = pd.read_pickle(tissue_feature_list[idx])
        superpixel = pd.read_pickle(superpixel_list[idx])
        final_estimated_type.extend(type_list)

        # load true label
        true_map = sio.loadmat(true_label_list[idx])['type_map']

        tissue_labels = []
        true_type = []
        for centroids in cell_graph.ndata['centroid'].tolist():
            x = int(centroids[0])
            y = int(centroids[1])
            l = superpixel[x][y]
            tissue_labels.append(l)
            true_type.append(int(true_map[x][y]))
        final_true_type.extend(true_type)


        unique_idx = [k-1 for k in sorted(list(set(tissue_labels)))]
        target_feature = tissue_feature[[unique_idx]]
        tissue_labels = relabel(tissue_labels) # relabel on the serial number.

        # cell labels of cell-cell graph
        s = cell_graph.edges()[0].tolist()
        d = cell_graph.edges()[1].tolist()

        us = [es+c_max for es in s]
        ud = [ed+c_max for ed in d]
        cci_s = us+ud
        cci_d = ud+us

        final_cci_s.extend(cci_s)
        final_cci_d.extend(cci_d)

        # cell labels of tissue-cell graph
        tc_dest = [i for i in range(cell_graph.num_nodes())] # tissue-cell destination (0,1,...,649)
        utc_dest = [tc+c_max for tc in tc_dest]
        final_cell_labels.extend(utc_dest)
        c_max = max(utc_dest)+1

        # update tissue labels
        ut = [tl+t_max for tl in tissue_labels]
        final_tissue_labels.extend(ut)
        t_max = max(ut)+1

        # concat each cell and tissue feature
        if idx == 0:
            merge_tissue_feature = target_feature
            merge_cell_feature = cell_feature
        else:
            merge_tissue_feature = torch.concat([merge_tissue_feature,target_feature])
            merge_cell_feature = torch.concat([merge_cell_feature,cell_feature])
        
        print('cell size: ',len(cell_graph.ndata['centroid']))

    # tissue-tissue interaction
    cor_adj = pd.DataFrame(merge_tissue_feature).T.corr()
    threshold = 0.7
    fxn = lambda x : x if (threshold < x)&(x<1) else 0
    cor_adj = cor_adj.applymap(fxn) # update
    adj_t = torch.tensor(np.array(cor_adj))
    edge_index = adj_t.nonzero().t().contiguous()
    ts = edge_index[0].tolist()
    td = edge_index[1].tolist()

    # process feature
    svd = TruncatedSVD(n_components=feature_dim, random_state=1) # tissue feature
    merge_tissue_feature = svd.fit_transform(merge_tissue_feature)
    svd = TruncatedSVD(n_components=feature_dim, random_state=1) # cell feature
    merge_cell_feature = svd.fit_transform(merge_cell_feature)

    # construct graph
    graph_data = {}
    graph_data[('tissue','tissue2cell','cell')] = (final_tissue_labels, final_cell_labels)
    graph_data[('cell','cell2tissue','tissue')] = (final_cell_labels, final_tissue_labels)
    graph_data[('cell','cci','cell')] = (final_cci_s, final_cci_d)
    graph_data[('tissue','tti','tissue')] = (ts+td, td+ts)
    graph = dgl.heterograph(graph_data)
    edges = ['tissue2cell','cell2tissue','cci','tti']

    graph.nodes['tissue'].data['id'] = torch.ones(graph.num_nodes('tissue')).long()
    graph.nodes['cell'].data['id'] = torch.arange(graph.num_nodes('cell')).long()
    graph.nodes['tissue'].data['feat'] = torch.tensor(merge_tissue_feature)
    graph.nodes['cell'].data['feat'] = torch.tensor(merge_cell_feature)
    graph.edges['cell2tissue'].data['weight'] = torch.ones(graph['cell2tissue'].num_edges())
    graph.edges['tissue2cell'].data['weight'] = torch.ones(graph['tissue2cell'].num_edges())
    graph.edges['cci'].data['weight'] = torch.ones(graph['cci'].num_edges())
    graph.edges['tti'].data['weight'] = torch.ones(graph['tti'].num_edges()) # torch.cat((adj_t[ts,td],adj_t[ts,td]))

    return graph, edges, final_estimated_type, relabel(final_true_type)

def tissue_cell_heterograph(superpixel, cell_graph, tissue_feature, cell_feature, feature_dim=32, bilayer=False, **kwargs):
    # assign cells to tissue
    tissue_labels = []
    for centroids in cell_graph.ndata['centroid'].tolist():
        x = int(centroids[0])
        y = int(centroids[1])
        l = superpixel[x][y]
        tissue_labels.append(l)
    # cell-cell interaction
    s = cell_graph.edges()[0].tolist()
    d = cell_graph.edges()[1].tolist()

    unique_idx = [k-1 for k in sorted(list(set(tissue_labels)))]
    target_feature = tissue_feature[[unique_idx]]
    tissue_labels = relabel(tissue_labels) # Relabel on the serial number.

    # feature dim compression
    svd = TruncatedSVD(n_components=feature_dim, random_state=1) # cell feature
    cell_feature = svd.fit_transform(cell_feature)
    svd = TruncatedSVD(n_components=feature_dim, random_state=1) # tissue feature
    target_feature = svd.fit_transform(target_feature)

    # tissue-tissue, tissue-cell, cell-cell
    if bilayer:
        # prepare tissue graph
        cor_adj = pd.DataFrame(target_feature).T.corr()
        threshold = 0.5
        fxn = lambda x : x if (threshold < x)&(x<1) else 0
        cor_adj = cor_adj.applymap(fxn) # update
        adj_t = torch.tensor(np.array(cor_adj))
        edge_index = adj_t.nonzero().t().contiguous()
        ts = edge_index[0].tolist()
        td = edge_index[1].tolist()

        graph_data = {}
        graph_data[('tissue','tissue2cell','cell')] = (tissue_labels, [i for i in range(cell_graph.num_nodes())])
        graph_data[('cell','cell2tissue','tissue')] = ([i for i in range(cell_graph.num_nodes())], tissue_labels)
        graph_data[('cell','cci','cell')] = (s+d, d+s)
        graph_data[('tissue','tti','tissue')] = (ts+td, td+ts)
        graph = dgl.heterograph(graph_data)
        edges = ['tissue2cell','cell2tissue','cci','tti']

        # add info to the graph
        graph.nodes['tissue'].data['id'] = torch.ones(graph.num_nodes('tissue')).long()
        graph.nodes['cell'].data['id'] = torch.arange(graph.num_nodes('cell')).long()
        graph.nodes['tissue'].data['feat'] = torch.tensor(target_feature)
        graph.nodes['cell'].data['feat'] = torch.tensor(cell_feature)
        graph.edges['cell2tissue'].data['weight'] = torch.ones(graph['cell2tissue'].num_edges())
        graph.edges['tissue2cell'].data['weight'] = torch.ones(graph['tissue2cell'].num_edges())
        graph.edges['cci'].data['weight'] = torch.ones(graph['cci'].num_edges())
        graph.edges['tti'].data['weight'] = torch.ones(graph['tti'].num_edges()) # torch.cat((adj_t[ts,td],adj_t[ts,td]))
    # tissue-cell, cell-cell
    else:
        graph_data = {}
        graph_data[('tissue','tissue2cell','cell')] = (tissue_labels, [i for i in range(cell_graph.num_nodes())])
        graph_data[('cell','cell2tissue','tissue')] = ([i for i in range(cell_graph.num_nodes())], tissue_labels)
        graph_data[('cell','cci','cell')] = (s+d, d+s)
        graph = dgl.heterograph(graph_data)
        edges = ['tissue2cell','cell2tissue','cci']

        # add info to the graph
        graph.nodes['tissue'].data['id'] = torch.ones(graph.num_nodes('tissue')).long()
        graph.nodes['cell'].data['id'] = torch.arange(graph.num_nodes('cell')).long()
        graph.nodes['tissue'].data['feat'] = torch.tensor(target_feature)
        graph.nodes['cell'].data['feat'] = torch.tensor(cell_feature)
        graph.edges['cell2tissue'].data['weight'] = torch.ones(graph['cell2tissue'].num_edges())
        graph.edges['tissue2cell'].data['weight'] = torch.ones(graph['tissue2cell'].num_edges())
        graph.edges['cci'].data['weight'] = torch.ones(graph['cci'].num_edges())

    return graph, edges

def relabel(label_list=[3,10,21,5]):
    """_summary_

    Args:
        label_list (list, optional): _description_. Defaults to [3,10,21,5].

    Returns:
        list: [0,2,3,1]
    """
    unique_s = sorted(list(set(label_list)))
    relabel_dic = dict(zip(unique_s,[i for i in range(len(unique_s))]))
    relabel_l = [relabel_dic.get(k) for k in label_list]
    return relabel_l

# %%------------------------------------------------------
# for heterogeneous cell types graph
def cell_type_uv(whole_graph,cell_type:int=1,num_types:int=5,type_list:list=[],target_labels=[1,2,3,4,5],relabel=True):
    """ Generate the components of heterogeneous cell type graph.
    Args:
        whole_graph (_type_): dgl.heterograph.DGLGraph. Original homogeneous graph. Returned from KNN graph builders.
        cell_type (int, optional): _description_. Defaults to 1.
        num_types (int, optional): _description_. Defaults to 5.
        type_list (list, optional): A list containing cell types. The index corresponds to the cell ID. Defaults to []. Generate like this:
            type_list = []
            for i,k in enumerate(data):
                type_list.append(data[k]['type']).
        target_labels (list, optional): _description_. Defaults to [1,2,3,4,5].
        relabel (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    whole_edges = whole_graph.edges()
    whole_sources = whole_edges[0].tolist()
    whole_destinations = whole_edges[1].tolist()
    id2type = dict(zip([i for i in range(len(type_list))], type_list))

    reid_dict_list = []
    for tmp in range(1,num_types+1):
        cell_selection = sorted([i for i, x in enumerate(type_list) if x == tmp])
        id2reid = dict(zip(cell_selection, [i for i in range(len(cell_selection))]))
        reid_dict_list.append(id2reid)

    cell_selection = [i for i, x in enumerate(type_list) if x == cell_type]
    uu_list = [[] for _ in range(num_types)]
    vv_list = [[] for _ in range(num_types)]
    for s_id in (cell_selection):
        indexes = [i for i, x in enumerate(whole_sources) if x == s_id]
        d_ids = [whole_destinations[idx] for idx in indexes]
        s_type = id2type.get(s_id)
        for d_id in d_ids:
            d_type = id2type.get(d_id)
            if d_type in target_labels:
                vv_list[d_type-1].append(d_id)
                uu_list[d_type-1].append(s_id)
            else:
                pass
    # inner (bidirected)
    i_uu = uu_list[cell_type-1]+vv_list[cell_type-1]
    i_vv = vv_list[cell_type-1]+uu_list[cell_type-1]
    uu_list[cell_type-1] = i_uu
    vv_list[cell_type-1] = i_vv

    # relabel
    if relabel:
        for j in range(len(uu_list)):
            update_uu = [reid_dict_list[cell_type-1].get(k) for k in uu_list[j]]
            uu_list[j] = update_uu
        
        for j in range(len(vv_list)):
            update_vv = [reid_dict_list[j].get(k) for k in vv_list[j]]
            if None in update_vv:
                print(cell_type,j)
            vv_list[j] = update_vv

    return uu_list, vv_list

def build_celltype_hetero(whole_graph,num_types=5,type_list=[]):
    graph_data = {}
    for i in range(1,num_types+1):
        uu_list, vv_list = cell_type_uv(whole_graph=whole_graph,cell_type=i,num_types=5,type_list=type_list,relabel=True)
        for k,uu in enumerate(uu_list):
            if i-1 ==k:
                graph_data[('cell_{}'.format(i-1),'inner','cell_{}'.format(k))] = (uu_list[i-1],vv_list[k])
            else:
                graph_data[('cell_{}'.format(i-1),'outer','cell_{}'.format(k))] = (uu_list[k],vv_list[k])
        
    graph = dgl.heterograph(graph_data)
    return graph

def cell_type_uv_legacy(whole_graph=None,cell_type:int=1,type_list:list=[]):
    # cell type selection
    indexes = [i for i, x in enumerate(type_list) if x == cell_type]
    inner_uu = []
    inner_vv = []
    outer_uu = []
    outer_vv = []
    for idx in indexes:
        source_idxs = [i for i, x in enumerate(whole_graph.edges()[0].tolist()) if x == idx]
        destinations = [whole_graph.edges()[1].tolist()[s] for s in source_idxs]
        inner_dest = sorted(list(set(destinations) & set(indexes)))
        outer_dest = sorted(list(set(destinations) - set(indexes)))
        # inner (bidirected)
        i_uu = [idx]*len(inner_dest) + inner_dest
        i_vv = inner_dest + [idx]*len(inner_dest)
        """
        e.g. idx=895
        i_uu = [895, 895, 895, 873, 877, 885]
        i_vv = [873, 877, 885, 895, 895, 895]
        """
        # outer (directed)
        o_uu = [idx]*len(outer_dest)
        o_vv = outer_dest

        # update
        inner_uu.extend(i_uu)
        inner_vv.extend(i_vv)
        outer_uu.extend(o_uu)
        outer_vv.extend(o_vv)

    return inner_uu,inner_vv,outer_uu,outer_vv

