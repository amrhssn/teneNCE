# teneNCE: Temporal Network Noise Contrastive Estimation for Dynamic Link Prediction

## Introduction
In a variety of scientific and engineering contexts, processes can be modeled as dynamical systems over 
evolving networks where the structure and features change over time.
Examples include disease transmission through contact networks, information propagation in communication networks, 
and complex systems like brain networks, biological networks, ecological networks, and financial transaction networks.
Learning expressive representations for such networks that encode both structural connectivity and their temporal
evolution is essential for downstream data analytics and machine learning applications.
Traditional methods often struggle to integrate both the structural and temporal information inherent in these networks.

In this work, we present _**teneNCE (Temporal Network Noise Contrastive Estimation)**_, a novel approach for 
learning expressive representations of temporal networks. 
Our method addresses the challenge of encoding both the structural and temporal aspects of dynamic graphs 
into meaningful vector representations. 
This is achieved through a local and global self-supervised training of a recurrent message-passing neural network architecture 
designed to model information flow along time-respecting paths within the network.

Key features of our method include:
- **Recurrent Message-Passing Architecture:** Facilitates the flow of information over time-respecting paths in 
the temporal network, capturing the dynamic nature of interactions.
- **Contrastive Training Objective:** Combines link prediction, graph reconstruction, and 
contrastive predictive coding losses. This objective integrates self-supervised infoNCE losses at 
both local and global scales, balancing the extraction of short-term features with long-term dynamics.
- **Performance:** We evaluate teneNCE on datasets including _Enron_, _COLAB_, and _Facebook_, demonstrating 
superior performance in dynamic link prediction tasks compared to existing models.

Our approach effectively combines structural and temporal information, improving the accuracy of dynamic link predictions and offering a robust method for future research in temporal network analysis.

For further details on implementation and usage, please refer to the sections below.
This is a PyTorch implementation of the teneNCE model as described in the paper.



## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Running the Code](#running-the-code)
  - [Using the Enron Dataset](#using-the-enron-dataset)
- [Citation](#citation)
- [Contact](#contact)


<!--
Temporal Network Noise Contrastive Estimation (teneNCE) is a novel framework for learning dynamic representations of temporal networks. The model leverages noise contrastive estimation to efficiently learn from the temporal network structure and dynamics, allowing for scalable and interpretable representations that can be used for a variety of downstream tasks.
**Abstract:** 
-->
<!-- Consider adding an abstract-like summary here. -->

## Introduction 
We present a self-supervised method for learning representations of temporal networks, focusing on discrete-time versions to balance computational efficiency and accuracy. Our approach uses a recurrent message-passing neural network and a contrastive training objective that combines link prediction, graph reconstruction, and contrastive predictive coding losses. Empirical results on Enron, COLAB, and Facebook datasets demonstrate that our method outperforms existing models in dynamic link prediction tasks.
<!--
Evolving networks are complex data structures that emerge in a wide range of systems in science and engineering. Learning expressive representations for such networks that encode both structural connectivities and their temporal evolution is essential for downstream data analytics and machine learning applications. In this study, we introduce a self-supervised method for learning representations of temporal networks and employ these representations in the dynamic link prediction task. While temporal networks are typically characterized as a sequence of interactions over the continuous time domain, our study focuses on their discrete-time versions. This enables us to balance the trade-off between computational complexity and precise modeling of the interactions. We propose a recurrent message-passing neural network architecture for modeling the information flow over time-respecting paths of temporal networks. The key feature of our method is the contrastive training objective of the model, which is a combination of three loss functions: link prediction, graph reconstruction, and contrastive predictive coding losses. The contrastive predictive coding objective is implemented using infoNCE losses at both local and global scales of the input graphs. We empirically show that the additional self-supervised losses enhance the training and improve the model’s performance in the dynamic link prediction task. The proposed method is tested on Enron, COLAB, and Facebook datasets and exhibits superior results compared to existing models.

(Typically, you could add an abstract-like section here, summarizing the main contributions, methodology, and findings of the paper. However, including the full abstract may make the README too long. Instead, consider providing a concise summary or key points that highlight the essence of the work.)
-->
## Requirements

To run this code, you'll need to install the following dependencies:

- Python==
- scikit-learn==
- torch==
- torch-cluster==
- torch-geometric==
- torch-scatter==
- torch-sparse==
- torch-spline-conv==
- torchvision==

## Running the Code

To train and evaluate the teneNCE model, follow the steps below.

### Using the Enron Dataset

1. **Prepare the Dataset:**
   - Download the Enron dataset and place it in the `dataset/` directory or use the existing data provided in the repository.

2. **Configure the Model:**
   - Adjust the configuration settings in the `config.yaml` file if necessary.

3. **Run the Model:**
   - Execute the main script to start training:

   ```bash
   python main.py --config config.yaml
   ```
   
## Citation

Please cite our paper if you use this code in your own work:

```bibtex
@article{teneNCE,
  title={Contrastive Representation Learning for Dynamic Link Prediction in Temporal Networks},
  author={Amirhossein Nouranizadeh, Fatemeh Tabatabaei and Mohammad Rahmati},
  year={2024},
  journal={Journal Name},
  note={\textbf{*}equal contribution}
}
```

## Contact

For any questions or inquiries, please feel free to contact us:

- **Fatemeh Tabatabaei:** [tabatabaeifatemeh@gmail.com](mailto:tabatabaeifateme@gmail.com)
- **Amirhossein Nouranizadeh:** [nouranizadeh@gmail.com.com](mailto:nouranizadeh@gmail.com)




