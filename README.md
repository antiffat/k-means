# KMeansClustering Project

## Overview
This project implements the K-Means Clustering algorithm in Python from scratch using `numpy` and `pandas`. It clusters data into a specified number of groups (k) and calculates the purity of each cluster by comparing cluster assignments with actual labels.

## Features
- Initialize centroids randomly within the range of the dataset.
- Assign data points to the nearest centroid using the Euclidean distance metric.
- Recalculate centroids as the mean of points assigned to each cluster.
- Check for convergence based on labels and centroid stability.
- Compute and display cluster purities to evaluate clustering quality.
- Load and process data from a CSV file.

## Requirements
- Python 3.x
- `numpy` library
- `pandas` library

## Files
- `main.py`: Contains the implementation of the K-Means algorithm.
- `iris_kmeans.txt`: A sample dataset file for testing the algorithm.
- `README.md`: This documentation file.

## How to Use
### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd kmeans_clustering
   ```
2. Install the required dependencies:
   ```bash
   pip install numpy pandas
   ```

### Running the Script
1. Place your dataset file in the project directory (e.g., `iris_kmeans.txt`). The file should have feature values in the first columns and actual labels in the last column.
2. Run the script:
   ```bash
   python main.py
   ```
3. Enter the number of clusters (`k`) when prompted.
4. The script will output iteration details, total sum of distances, and purities of each cluster until convergence.

### Sample Input Data Format
The dataset file should be in CSV format (e.g., `iris_kmeans.txt`):
```
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
6.7,3.1,4.7,1.5,Iris-versicolor
...
```

### Output
- Iteration number and total sum of distances.
- Cluster purities (e.g., percentage of each actual label in a cluster).
- Centroids of the clusters at convergence.
- Notification of convergence.

## Project Structure
```
|-- main.py
|-- iris_kmeans.txt
|-- README.md
```

## Example Output
```
Enter the number of clusters (k): 3

--------------------------------------------------------------------
Iteration 0: Total sum of distances: 150.25
Cluster 1 Purity: 90.00% Iris-setosa, 10.00% Iris-versicolor
Cluster 2 Purity: 85.00% Iris-versicolor, 15.00% Iris-virginica
Cluster 3 Purity: 100.00% Iris-virginica

--------------------------------------------------------------------
Iteration 1: Total sum of distances: 100.75
Cluster 1 Purity: 100.00% Iris-setosa
Cluster 2 Purity: 88.00% Iris-versicolor, 12.00% Iris-virginica
Cluster 3 Purity: 95.00% Iris-virginica

Convergence reached: No changes in labels or centroids.
```
---
