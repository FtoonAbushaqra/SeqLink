# SeqLink: A Robust Neural-ODE Architecture for Modelling Partially Observed Time Series


This repository contains the code for our enhanced Neural-ODE Architecture. The code is written in PyTorch and follows the methodology described in our paper available on [OpenReview]([link_to_your_paper](https://openreview.net/forum?id=WCUT6leXKf)).

This work is built on top of publicly available implementations of ODE-RNN  and latent ODE avaliable at [https://github.com/YuliaRubanova/latent_ode], Additionally, For the baselines (RNN-VAE, Latent ODE and ODE-RNN) the code is also available on [https://github.com/YuliaRubanova/latent_ode], While for the CDE model, we follow the implementation available at  (\url{https://github.com/patrick-kidger/NeuralCDE}). For TSMixer we follow the implementation available at  (\url{https://github.com/ditschuk/pytorch-tsmixer})
 


## Model Overview

![Model Architecture](path/to/your/model_image.png)

We build our code on the publicly available code for ODE-RNN at [Yulia Rubanova's GitHub](https://github.com/YuliaRubanova/latent_ode), using PyTorch. For the baselines (RNN-VAE, Latent ODE, and ODE-RNN) we follow the implementation available at [Yulia Rubanova's GitHub](https://github.com/YuliaRubanova/latent_ode). For the CDE model, we follow the implementation available at [Patrick Kidger's GitHub](https://github.com/patrick-kidger/NeuralCDE). For TSMixer we follow the implementation available at [ditschuk's GitHub](https://github.com/ditschuk/pytorch-tsmixer).

## Enhancements and Modifications

In our code, we use the ODE-RNN model to generate learned representations. Following that, we use the previously learned hidden states \(U\) to define a set of latent representations for each sequence based on the correlation between samples. We find the attention score between the samples and the learned representations from the auto-encoder. As shown in the figure above, we first map the original data \(x\) to the learned representations \(u\) by embedding both vectors as shown in the following equations, where \(\varphi\) refers to the embedding layer and \(\theta\) represents the learning weights:

\[ u = \varphi(x; \theta) \]

## Repository Structure

- `src/`: Contains the source code for the model, attention mechanism, and pyramid sorting.
- `data/`: Example data files used for training and testing.
- `notebooks/`: Jupyter notebooks demonstrating the usage of the model and visualizing results.
- `docs/`: Documentation and additional resources.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt



### Notes:
- Replace `path/to/your/model_image.png` with the actual path to your model image file.
- Replace `link_to_your_paper` with the actual link to your paper on OpenReview.
- Customize the usage instructions and other placeholder text as necessary.

This README provides a comprehensive overview of your project, properly attributes the original work, and explains your contributions.

