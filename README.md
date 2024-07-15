# MSGA-Net
Official implementation of MSGA-Net: Progressive Feature Matching via Multi-layer Sparse Graph Attention.

The paper has been accepted by TCSVT2024.

## Introduction
Feature matching is an essential computer vision task that requires the establishment of high-quality correspondences between two images. Constructing sparse dynamic graphs and extracting contextual information by searching for neighbors in feature space is a prevalent strategy in numerous previous works. Nonetheless, these works often neglect the potential connections between dynamic graphs from different layers, leading to underutilization of available information. To tackle this issue, we introduce a Sparse Dynamic Graph Interaction block for feature matching. This innovation facilitates the implicit establishment of dependencies by enabling interaction and aggregation among dynamic graphs across various layers. In addition, we design a novel Multiple Sparse Transformer to enhance the capture of the global context from the sparse graph. This block selectively mines significant global contextual information along spatial and channel dimensions, respectively. Ultimately, we present the Multi-layer Sparse Graph Attention Network (MSGA-Net), a framework designed to predict probabilities of correspondences as inliers and to recover camera poses.
