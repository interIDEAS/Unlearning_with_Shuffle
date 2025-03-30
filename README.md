This repository contains the experimental setup, code, and results for our paper on "Feature Unlearning: Theoretical Foundations and Practical Applications with Shuffling". In response to reviewer comments regarding the simplicity of our initial models, we have incorporated more complex models, namely FtFormer and ResNet, to demonstrate the robustness and scalability of our unlearning approach.

## Experiment Setup

### Models

We expanded our experiments to include two additional, more complex models:
- **FtFormer**: A transformer-based model designed to handle sequential data.
- **ResNet**: A deep residual network known for its effectiveness in image processing tasks.

### Feature Unlearning

Feature unlearning was conducted on multiple datasets. For each dataset, we identified and unlearned:
- The **top two most influential features**, as determined by their Shapley values.
- The **least two influential features**, to assess the impact of unlearning less critical information.

Each of these features was unlearned across three different architectural frameworks:
1. Baseline Model (as per the original experiments)
2. FtFormer
3. ResNet

### Extended Experiments

We also conducted experiments on features with high correlation coefficients to understand the effects of unlearning interdependent features. These experiments were carried out using the same three architectural frameworks mentioned above.

## Results and Graphs

Corresponding results for each experiment are provided in the form of graphs, which illustrate the impact of feature unlearning on model performance and behavior. These graphs are crucial for visualizing the trade-offs and outcomes of unlearning specific features.

## How to Run Experiments

To replicate our experiments or to run new experiments using the setup provided, follow the instructions below:

```bash
# Clone the repository
git clone https://github.com/your-repository-url.git
