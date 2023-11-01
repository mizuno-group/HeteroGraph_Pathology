# -*- coding: utf-8 -*-
"""
Created on 2023-07-11 (Tue) 15:12:04

Handles graph building.

reference: 
https://github.com/BiomedSciAI/histocartography/blob/main/histocartography/preprocessing/graph_builders.py

@author: I.Azuma
"""
#%%
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2
import dgl
import json
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph

from _utils.pipelines import PipelineStep

#%%
LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"

#%%
class BaseGraphBuilder(PipelineStep):
    """
    Base interface class for graph building.
    """

    def __init__(
            self,
            nr_annotation_classes: int = 5,
            annotation_background_class: Optional[int] = None,
            add_loc_feats: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Base Graph Builder constructor.
        Args:
            nr_annotation_classes (int): Number of classes in annotation. Used only if setting node labels.
            annotation_background_class (int): Background class label in annotation. Used only if setting node labels.
            add_loc_feats (bool): Flag to include location-based features (ie normalized centroids)
                                  in node feature representation.
                                  Defaults to False.
        """
        self.nr_annotation_classes = nr_annotation_classes
        self.annotation_background_class = annotation_background_class
        self.add_loc_feats = add_loc_feats
        super().__init__(**kwargs)
    
    def _fromjson(self,json_data=None,json_path='/workspace/github/TopoPathology/preprocessing/results/BRACS_291_IC_20_res.json'):
        if json_data is None:
            with open(json_path) as json_file:
                data = json.load(json_file)
        else:
            data = json_data
    
        # collect node feature and centroids
        features = []
        centroids = np.empty((len(data), 2))
        for i,k in enumerate(data):
            if  data[k]['node_feature'] is None:
                raise ValueError('!! node feature is not defined !!')
            node_f = data[k]['node_feature']
            cent = data[k]['centroid']
            features.append(node_f)
            centroids[i,0] = int(round(cent[0]))
            centroids[i,1] = int(round(cent[1]))
        features = torch.tensor(features,dtype=torch.float32)
        return features, centroids

    def _process(  # type: ignore[override]
        self,
        instance_map: np.ndarray = None,
        features: torch.Tensor = None,
        centroids: np.ndarray = None,
        annotation: Optional[np.ndarray] = None,
        mat_path: Optional[str] = '/workspace/github/TopoPathology/HoVerNet_res/BRACS/BRACS_291_IC_20.mat',
        json_path: Optional[str] = '/workspace/github/TopoPathology/preprocessing/results/BRACS_291_IC_20_res.json',
        json_data: Optional[dict] = None,
    ) -> dgl.DGLGraph:
        """Generates a graph from a given instance_map and features
        Args:
            instance_map (np.array): Instance map depicting tissue components
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Union[None, np.array], optional): Optional node level to include.
                                                          Defaults to None.
        Returns:
            dgl.DGLGraph: The constructed graph
        """
        # collect node features and centroids
        try:
            json_features, json_centroids = self._fromjson(json_data=json_data,json_path=json_path)
            print('obtained data from json file')
        except:
            pass

        if instance_map is None:
            instance_map = sio.loadmat(mat_path)['inst_map']
        if features is None:
            features = json_features
        if centroids is None:
            centroids = json_centroids

        # add nodes
        num_nodes = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)

        # add image size as graph data
        image_size = (instance_map.shape[1], instance_map.shape[0])  # (x, y)

        # add node content
        self._set_node_centroids(centroids, graph)
        self._set_node_features(features, image_size, graph)
        if annotation is not None:
            self._set_node_labels(instance_map, annotation, graph)

        # build edges
        self._build_topology(instance_map, centroids, graph)
        return graph

    def _process_and_save(  # type: ignore[override]
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
        output_name: str = None,
    ) -> dgl.DGLGraph:
        """Process and save in provided directory
        Args:
            output_name (str): Name of output file
            instance_map (np.ndarray): Instance map depicting tissue components
                                       (eg nuclei, tissue superpixels)
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Optional[np.ndarray], optional): Optional node level to include.
                                                         Defaults to None.
        Returns:
            dgl.DGLGraph: [description]
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.bin"
        if output_path.exists():
            logging.info(
                f"Output of {output_name} already exists, using it instead of recomputing"
            )
            graphs, _ = load_graphs(str(output_path))
            assert len(graphs) == 1
            graph = graphs[0]
        else:
            graph = self._process(
                instance_map=instance_map,
                features=features,
                annotation=annotation)
            save_graphs(str(output_path), [graph])
        return graph

    def _set_node_centroids(
            self,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the centroids of the graphs
        Args:
            centroids (np.ndarray): Node centroids
            graph (dgl.DGLGraph): Graph to add the centroids to
        """
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)

    def _set_node_features(
            self,
            features: torch.Tensor,
            image_size: Tuple[int, int],
            graph: dgl.DGLGraph
    ) -> None:
        """Set the provided node features

        Args:
            features (torch.Tensor): Node features
            image_size (Tuple[int,int]): Image dimension (x, y)
            graph (dgl.DGLGraph): Graph to add the features to
        """
        if not torch.is_tensor(features):
            features = torch.FloatTensor(features)
        if not self.add_loc_feats:
            graph.ndata[FEATURES] = features
        elif (
                self.add_loc_feats
                and image_size is not None
        ):
            # compute normalized centroid features
            centroids = graph.ndata[CENTROID]

            normalized_centroids = torch.empty_like(centroids)  # (x, y)
            normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
            normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]

            if features.ndim == 3:
                normalized_centroids = normalized_centroids \
                    .unsqueeze(dim=1) \
                    .repeat(1, features.shape[1], 1)
                concat_dim = 2
            elif features.ndim == 2:
                concat_dim = 1

            concat_features = torch.cat(
                (
                    features,
                    normalized_centroids
                ),
                dim=concat_dim,
            )
            graph.ndata[FEATURES] = concat_features
        else:
            raise ValueError(
                "Please provide image size to add the normalized centroid to the node features."
            )

    @abstractmethod
    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the node labels of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
            annotation (np.ndarray): Annotations, eg node labels
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Generate the graph topology from the provided instance_map
        Args:
            instance_map (np.array): Instance map depicting tissue components
            centroids (np.array): Node centroids
            graph (dgl.DGLGraph): Graph to add the edges
        """

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information
        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "graphs")


class CentroidsKNNGraphBuilder(BaseGraphBuilder):
    """
    k-Nearest Neighbors Graph class for graph building.
    - Definition of neighborhood based on centroids.
    """

    def __init__(self, k: int = 5, thresh: int = None, **kwargs) -> None:
        """Create a graph builder that uses the (thresholded) kNN algorithm to define the graph topology.

        Args:
            k (int, optional): Number of neighbors. Defaults to 5.
            thresh (int, optional): Maximum allowed distance between 2 nodes. Defaults to None (no thresholding).
        """
        logging.debug("*** kNN Graph Builder ***")
        self.k = k
        self.thresh = thresh
        super().__init__(**kwargs)

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation"""
        regions = regionprops(instance_map)
        assert annotation.shape[0] == len(regions), \
            "Number of annotations do not match number of nodes"
        graph.ndata[LABEL] = torch.FloatTensor(annotation.astype(float))

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Build topology using (thresholded) kNN"""

        # build kNN adjacency
        adj = kneighbors_graph(
            centroids,
            self.k,
            mode="distance",
            include_self=False,
            metric="euclidean").toarray()

        # filter edges that are too far (ie larger than thresh)
        if self.thresh is not None:
            adj[adj > self.thresh] = 0

        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))

def two_hop_neighborhood(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Increases the connectivity of a given graph by an additional hop

    Args:
        graph (dgl.DGLGraph): Input graph
    Returns:
        dgl.DGLGraph: Output graph
    """
    A = graph.adjacency_matrix().to_dense()
    A_tilde = (1.0 * ((A + A.matmul(A)) >= 1)) - torch.eye(A.shape[0])
    ngraph = nx.convert_matrix.from_numpy_matrix(A_tilde.numpy())
    new_graph = dgl.DGLGraph()
    new_graph.from_networkx(ngraph)
    for k, v in graph.ndata.items():
        new_graph.ndata[k] = v
    for k, v in graph.edata.items():
        new_graph.edata[k] = v
    return new_graph