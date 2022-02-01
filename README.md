## **Reference**
This repository contains the code associated with our paper:

Paul Maragakis, Hunter Nisonoff, Brian Cole, and David E. Shaw, "A Deep-Learning View of Chemical Space Designed to Facilitate Drug Discovery," *Journal of Chemical Information and Modeling*, vol. 60, no. 10, 2020, pp. 4487–4496. [Text](https://doi.org/10.1021/acs.jcim.0c00321)

The corresponding dataset can be found on our [web site](https://www.deshawresearch.com/downloads/download_desmiles.cgi/).

## **Installing and setup**

To train your own data you'll need a GPU compatible with CUDA 10.1.  You can run without a GPU using our pre-trained data set, but performance will be lower than with a GPU.

You will need a compatible cuDNN and

```
python >=3.7, <=3.9
pytorch >= 1.0
fastai == 1.0.55
scipy
numpy
rdkit
```
The tests require `pytest` and `hypothesis`.

For the provided sample notebooks you will also need:
```
jupyter
matplotlib
seaborn
sentencepiece
```

**note:** The script [download_drd2_dataset.sh](https://github.com/DEShawResearch/DESMILES/blob/master/tests/download_drd2_dataset.sh) requires`rdkit==2018.09.01`  `scikit-learn==0.19.2` are needed.


**Conda**

For easy installation, we've also provided a conda [environment.yml](environment.yml).  Refer to the [miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) for instructions for installing conda.  The conda environment is all that is required for CPU applications.  For GPU applications, the environment is limited to CUDA 10.  Running DESMILES on GPUs not compatible with CUDA 10 requires building pytorch 1.0.0 from source.

**Containers**

We're including a [Dockerfile](Dockerfile) to build a containerized, GPU-enabled version of DESMILES.  You can build a docker image by running:

`docker build -t desmiles:1.0 https://github.com/DEShawResearch/DESMILES.git#1.0`

## **Using**

DESMILES identifies the data directory with the environment variable DESMILES_DATA_DIR. Set it with
```
export DESMILES_DATA_DIR=<Path/To/DESMILES/data>
```
Where `DESMILES/data` is the unpacked form of the [data set](https://www.deshawresearch.com/downloads/download_desmiles.cgi/).

If you are using a container, you'll need to make the data directory visible within the container and correspond to the environment variable `DESMILES_DATA_DIR` within the container. The provided Dockerfile defaults `DESMILES_DATA_DIR` such that the following bind mount will work:
 `-v <Path/to/DESMILES/data>:/desmiles/data`


### **Jupyter notebooks**

We provide two demo Jupyter notebooks:

* [intro_demo_of_DESMILES.ipynb](Notebooks/intro_demo_of_DESMILES.ipynb) shows simple ways to use a pretrained model to generate molecules.
* [overview_of_DESMILES.ipynb](Notebooks/overview_of_DESMILES.ipynb) shows examples of training a simple model and fine-tuning an existing model.

The `overview_of_DESMILES.ipynb` notebook demonstrates how to sample potential potent binders to DRD2 using the benchmark data set published by [Wengong Jin, Kevin Yang, Regina Barzilay, Tommi Jaakkola](https://arxiv.org/abs/1812.01070).
We've provided a script to download and preprocess the data
 [download_drd2_dataset.sh](tests/download_drd2_dataset.sh).

If you are using a container, don't forget to include port forwarding in the container and Jupyter:

* ```docker run -p 8888:8888 -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v <Path/to/DESMILES/data>:/desmiles/data desmiles:1.0```

* `jupyter notebook --ip "0.0.0.0" --no-browser --allow-root`

* edit the URL accordingly and access from a local browser
