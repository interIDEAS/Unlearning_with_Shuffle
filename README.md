This repository contains the experimental setup, code, and results for our paper on "Feature Unlearning: Theoretical Foundations and Practical Applications with Shuffling". In response to reviewer comments regarding the simplicity of our initial models, we have incorporated more complex models, namely FtFormer and ResNet, to demonstrate the robustness and scalability of our unlearning approach.

# Experiment Setup

## Models

We have used three different models to demonstrate the efficacy and robustness of our feature unlearning approach. Each model offers a distinct architecture and is suited for various types of data:

- **MLP (Multi-Layer Perceptron)**: A classical neural network model consisting of multiple layers of perceptrons. It is widely used for tabular data and serves as our baseline model.
- **FtFormer**: A transformer-based model designed to handle sequential data.
- **ResNet**: A deep residual network known for its effectiveness in image processing tasks.

## Unlearning based on the Feature Importance

Feature unlearning was conducted on multiple datasets. For each dataset, we identified and unlearned:
- The **top two most influential features**, as determined by their Shapley values.
- The **least two influential features**, to assess the impact of unlearning less critical information.

Each of these features was unlearned across three different architectural frameworks:
1. Baseline Model (as per the original experiments)
2. FtFormer
3. ResNet

### Results and Graphs
Corresponding results for each experiment are provided in the form of graphs, which illustrate the impact of feature unlearning on model performance and behavior. These graphs are crucial for visualizing the trade-offs and outcomes of unlearning specific features.


### Unlearning based on the Feature Correaltion

We also conducted experiments on features with high correlation coefficients to understand the effects of unlearning interdependent features. These experiments were carried out using the same three architectural frameworks mentioned above.

### Results and Graphs
Corresponding results for each experiment are provided in the form of graphs, which illustrate the impact of feature unlearning on model performance and behavior. These graphs are crucial for visualizing the trade-offs and outcomes of unlearning specific features.



## How to Run Experiments

To replicate our experiments or to run new experiments using the setup provided, follow the instructions below:

```bash
# Clone the repository
git clone https://github.com/your-repository-url.git
