# Graph Anomaly Detection in Adversarial Environment

### Intro

We're attacking the following methods:

- **Autopart (Parameter-Free Graph Partitioning)**: This method was
proposed by Deepayan Chakrabarti in the paper AutoPart: Parameter-Free
Graph Partitioning and Outlier Detection.
The main idea is to reorder the adjacency matrix based on a parameter-free
clustering. This clustering is determined by minimizing a graph-compression
encoding cost. The optimal parameter _k_ of clusters is found by iterating
through different _k_, until the returns (efficiency of encoding) diminish.
This method is implemented in `model_autopart.py`.

- **Embedding-Based Graph Anomaly Detection**:
This method was proposed in an Embedding Approach to Anomaly Detection by
_Renjun Hu_, _Charu C. Aggarwal_, _Shuai Ma_, _Jinpeng Huai.
The goal is to find an optimal embedding for representing node clusters
through gradient descent. Anomalous nodes are then detected as nodes with
a high degree of connectivity across multiple different communities.
This method is implemented in `model_embedding.py`.

### Quickstart (Docker)
The fastest way to get started is through Docker. If you have Docker
installed, simply run
```
./docker.sh
```
to create the image (if not exists) and start a new container.
If the image already exists, the container is started directly.
Be sure that the `setup` files have the execution permission to run.

### Dependencies
If you don't want to use Docker, everything should still run on UNIX-based OS.
To download the dataset and install required packages and plugins, run
```
cd setup && ./setup.sh
```
This will also download and install the METIS library for graph partitioning
which is used by the ModelEmbedding method.
The METIS library is installed at `~/bin/metis-5.1.0`.

Setup the environment variables by running
```
export PATH_REPO_AGAD=$(pwd)
export METIS_DLL="$(echo ~/bin/metis-5.1.0/build/*/libmetis/libmetis.so)"
```

### Dataset
This repo supports usage of the `dblp` and `movielens` dataset.
Additionally, there is a `mock` dataset supplied which can be used to ensure
everything works correctly.

### Usage
This repo is written in `python3`.

To run the supplied models use
```
./detect.py [-m <model>] [-d <dataset>]
```
with the supported models being `autopart` (default) or `embedding`
that is using dataset `mock` (default),`dblp` or `movielens`

To run one of the available attacks on a model use
```
./attack.py [-a <attack>] [-m <model>] [-d <dataset>]
```
with the supported attacks being `random`, `heuristic` and `gradient`
