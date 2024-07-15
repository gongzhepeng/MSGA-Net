# MSGA-Net
Official implementation of MSGA-Net: Progressive Feature Matching via Multi-layer Sparse Graph Attention.

The paper has been accepted by TCSVT 2024.

## Introduction
Feature matching is an essential computer vision task that requires the establishment of high-quality correspondences between two images. Constructing sparse dynamic graphs and extracting contextual information by searching for neighbors in feature space is a prevalent strategy in numerous previous works. Nonetheless, these works often neglect the potential connections between dynamic graphs from different layers, leading to underutilization of available information. To tackle this issue, we introduce a Sparse Dynamic Graph Interaction block for feature matching. This innovation facilitates the implicit establishment of dependencies by enabling interaction and aggregation among dynamic graphs across various layers. In addition, we design a novel Multiple Sparse Transformer to enhance the capture of the global context from the sparse graph. This block selectively mines significant global contextual information along spatial and channel dimensions, respectively. Ultimately, we present the Multi-layer Sparse Graph Attention Network (MSGA-Net), a framework designed to predict probabilities of correspondences as inliers and to recover camera poses.


![image](https://github.com/gongzhepeng/MSGA-Net/blob/main/Frame.png)

## Requirements

### Installation
We recommend using Anaconda. To setup the environment, follow the instructions below.
```
conda create -n msga python=3.7 --yes
conda activate msga
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit==11.3 -c pytorch --yes
python install -r requirements.txt
```

### Dataset
Follow the instructions provided [here](https://github.com/zjhthu/OANet) for downloading and preprocessing datasets. 

### Test pretrained model
We provide the model trained on YFCC100M described in our paper. Run the test script to get results in our paper.
```
cd ./core 
python main.py --run_mode=test --model_path=../weight --res_path=../weight/yfcc/sift-2000/ --use_ransac=False
```
Set --use_ransac=True to get results after RANSAC post-processing.

### Train model on YFCC100M
After generating dataset for YFCC100M, run the tranining script.
```
cd ./core 
python main.py
```

## Acknowlegment
This repo benefits from [OANet](https://github.com/zjhthu/OANet) and [CLNet](https://github.com/sailor-z/CLNet). Thanks for their wonderful works.

## Citation
Thanks for citing our paper:
```
@inproceedings{liao2024vsformer,
  title={MSGA-Net: Progressive Feature Matching via Multi-layer Sparse Graph Attention},
  author={Gong, Zhepeng and Xiao, Guobao and Shi, Ziwei and Chen, Riqing and Yu, Jun},
  volume={34},
  pages={5765--5775},
  year={2024}
}
```
