# -*- coding: utf-8 -*-
"""
Created on 2023-11-01 (Wed) 17:11:07

heterograph builders

@author: I.Azuma
"""
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


