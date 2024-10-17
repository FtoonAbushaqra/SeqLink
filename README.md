# SeqLink: A Robust Neural-ODE Architecture for Modelling Partially Observed Time Series



This repository contains the code for our enhanced Neural-ODE architecture. The code is written in PyTorch and follows the methodology described in our paper, available on [OpenReview\SeqLink](https://openreview.net/forum?id=WCUT6leXKf).


## Model Overview

![Model Architecture](Framework.png)

SeqLink comprises three key components: (1) ODE Auto-Encoder: This component uses neural ODEs to learn optimal hidden representations for each sample. It takes datasets as input, employing neural ODEs to capture continuous hidden trajectories that best represent each sample. Subsequently, it returns the most suitable representation for each sample. (2) Pyramidal Attention Mechanism: Designed to delineate correlations between samples, this method maps data with each other. By leveraging the learned representations as input, it discerns, for each sample, the most relevant representations of other samples. It then sorts these representations based on their relationships to each sample. (3) Link-ODE: A generalised ODE-based model tailored to modelling partially observed irregular time series. By utilising the best-hidden trajectories to fill in gaps in the data, this model incorporates learned latent states from another related sample alongside sample-specific latent states to represent each sample effectively.


Our code builds upon the publicly available ODE-RNN code from [Yulia Rubanova's GitHub](https://github.com/YuliaRubanova/latent_ode) For the baselines (RNN-VAE, Latent ODE, and ODE-RNN), we follow the implementation available at [Yulia Rubanova's GitHub](https://github.com/YuliaRubanova/latent_ode). For the CDE model, we follow the implementation available at [Patrick Kidger's GitHub](https://github.com/patrick-kidger/NeuralCDE). Finnaly, for TSMixer, we follow the implementation available at [ditschuk's GitHub](https://github.com/ditschuk/pytorch-tsmixer).

## Datasets
All datasets are available in the `Dataset folder`. Including original data and the (.pt) format to be used for the SeqLink  model

The learned representations generated using ODE-RNN are saved in the `datasets/latent_trajectories/` folder. To regenerate these, we recommend following the instructions from the original code repository, which we have modified in the `ODE trajectories/`.


The attention and pyramid module code is located in `pyramidal_attention.py`


To generate the final prediction, use the (.pt) dataset that already includes the attention weights, or generate it as described previously, and run the modified`Link-ODE` code.



## Repository Structure

- `Link_ODE/`:  Contains the source code for the SeqLink model, attention mechanism, and pyramid sorting.
- `data/`: Example data files used.
- `ODE trajectories/`: Contains the source code for the ODE auto-encoder model.


## Prerequisites
Install torchdiffeq from https://github.com/rtqichen/torchdiffeq.

To install the required dependencies, run:

```bash
pip install -r requirements.txt


