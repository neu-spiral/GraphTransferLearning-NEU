# Contents

* [Acknowledgement](#acknowledgement)
* [Citing This Paper](#citing-this-paper)
* [Environment Setup](#environment-setup)
* [Running Framework](#running-framework) 
* [Datasets](#datasets)


## Acknowledgement 
This repository contains the source code for the Graph Transfer Learning project developed by the Northeastern University's SPIRAL research group.


## Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> A. Gritsenko, Y. Guo, K. Shayestehfard, A. Moharrer, J. Dy, S. Ioannidis, "Graph Transfer Learning", ICDM, 2021.


## Environment Setup
Please install the python dependencies found in `requirements.txt` with:
```bash
pip install -r requirements.txt
```

## Running Framework
To fully address the generic nature of the algorithm introduced in the original paper, we provide a fully-customizable framework with a wide variety of parameters for node embedding, model creation and training.

The following arguments can be specified to train node embeddings:
```bash
  --nembedding          Size of the output embedding vector
  --topology_similarity Similarity measure between nodes of the same graph in
                        graph topological space
  --embedding_type      Type of embedding function: skipgram, unified
  -embedding_similarity Similarity measures between nodes of the same graph in
                        embedding space
  --nwalks              Number of node2vec random walks
  --walk_length         Length of random walk
  --window_size         Width of sliding window in random walks
  --p                   Parameter p for node2vec random walks
  --q                   Parameter q for node2vec random walks
  --nnegative           Number of negative samples used in skip-gram
  --scale_negative      Specifies whether to scale outputs for negative
                        samples
  --graph_distance      Pairwise distance measure between nodes in the
                        embedding space (matrix D)
```
The following arguments can be specified to create and train model:
```bash
  --similarity_loss     Loss function between similarities in topological and
                        embedding spaces for nodes of the same graph
  --depth               Number of hidden layers in Prediction Branch
  --activation_function Activation function for Prediction Branch neurons
  --prediction_loss     Loss function for Prediction Branch
  --transfer_mode       Specifies transfer learning mode
  --alpha               Weight of graph matching loss
  --beta                Specifies whether to scale parts of P-optimization
                        loss
  --learning_rate       Learning rate
  --batch_size          Number of instances in each batch
  --epochs              Number of epochs
  --early_stopping      Number of epochs with no improvement after which
                        training will be stopped. If <=0, no early stopping is
                        used
```
For a full list of arguments to run a framework, you may use `--help`.


## Datasets
All datasets referenced in the original paper are presented in the folder `data`. A user can run the framework on either provided datasets, or any arbitrary ones by specifying the dataset folder via `--load_path` and `--dataset` parameters. 
The framework expects the following files to be present in the specified dataset directory:
 * a `GraphA.txt` file containing graph's A adjacency matrix, 
 * a `GraphALabels_cluster.txt` file containing class labels for each graph A node, 
 * a `GraphALabels_infection.txt` file containing infection labels for each graph A node. 
Optionally, a dataset directory can contain `GraphATrain.txt` and `GraphATest.txt` files containing node indices for train and test splits, respectively. If these files are not provided, graph nodes are split randomly into train and test subsets with a ratio 8:2.
We provide `GraphATrain.txt` and `GraphATest.txt` files for all real-world datasets for the reproducibility purposes. Additionally, we provide original dataset files for each real-world graph.

The following real-world datasets are presented in the folder `data`:
 * Zachary Karate Club
   > W. W. Zachary, “An information flow model for conflict andfission in small groups”, Journal of Anthropological Research, 1977
 * Email 
   > J. Leskovecet et al., “Graph  evolution:  Densification  andshrinking diameters”, ACM TKDD, 2007
 * Infectious Disease Transmission Dataset
   > M. Salathé et al., “A high-resolution human contact networkfor infectious disease transmission”, PNAS, 2010

For the details on synthetic dataset construction, please refer to Section V.A of the original paper.
