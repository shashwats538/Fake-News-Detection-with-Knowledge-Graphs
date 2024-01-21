# Heterogeneous Graphs for Fake News Detection

Exploration of the use of heterogeneous graphs centered around news articles for fake news detection. The construction of these graphs incorporates contextual information within a network structure. Specifically, the following node types are employed in the graphs, reformulating the problem as a graph classification task:

News articles
User postings (tweets)
User repostings (retweets)
User accounts
User timeline-posts

Project Structure:
data_preprocessing
Python files dedicated to loading and preprocessing data. It is essential to have a folder named data in the project's root directory, with two subfolders mirroring the structure of FakeNewsNet's dataset and fakenewsnet_dataset folders.

feature_extraction.py: Extracts node-related features such as retweet count and generates transformer-based text embeddings.
graph_structure.py: Contains functions for generating graphs from data. An example is provided in scripts/generate_graphs.py.
load_data.py: Helper functions for loading data from the data folder during graph construction.
text_summarization.py: Generates extractive and abstractive summaries from text (not yet in use).
visualization.py: Includes a function to visualize homogeneous graphs.
machine_learning


Python files related to graph machine learning.

gnn_models.py: Graph Neural Networks (GNNs) used for experiments, including SAGE, GAT, HGT. The architecture is currently adapted to graphs featuring all types of information, crucial for mean pooling node types.
gnn_training.py: Manages the training and evaluation of models.
scripts
generate_graphs.py: An example script demonstrating how to generate graphs. Parameters can be set to specify which node types should be considered.
run_experiment.py: An example script illustrating how the generated graphs can be used to run graph classification experiments.