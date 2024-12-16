import numpy as np
import pandas as pd

class KMeansClustering:
    # By default, k is set to 3, but we can also change it when we crate the instance of this class.
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(((data_point - centroids) ** 2).sum(axis=1))

    def fit(self, data, actual_labels):
        # Initialize centroids using np.random.uniform
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        self.centroids = np.random.uniform(min_vals, max_vals, (self.k, data.shape[1]))

        labels = np.zeros(data.shape[0], dtype=int)
        previous_labels = None
        previous_centroids = np.zeros_like(self.centroids)

        iteration = 0
        while True:
            total_distance = 0
            # E-step: assign points to the nearest centroid
            for i, point in enumerate(data):
                distances = self.euclidean_distance(point, self.centroids)
                total_distance += np.min(distances)
                labels[i] = np.argmin(distances)

            purity_dict = self.compute_purity(labels, actual_labels)
            purity_output = "\n".join([f"Cluster {i+1} Purity: {p}" for i, p in purity_dict.items()])

            print("\n")
            print("--------------------------------------------------------------------")
            print("\n")

            # Print iteration details including purities
            print(f"Iteration {iteration}: Total sum of distances: {total_distance:.2f}")
            print(purity_output)

            # Check for convergence
            if np.array_equal(labels, previous_labels) and np.allclose(self.centroids, previous_centroids):
                print("Convergence reached: No changes in labels or centroids.")
                break
            previous_labels = labels.copy()
            previous_centroids = self.centroids.copy()

            # M-step: recompute centroids
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.k):
                cluster_points = data[labels == i]
                if cluster_points.size > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = data[np.random.choice(data.shape[0])]

            self.centroids = new_centroids
            iteration += 1

        return labels, self.centroids

    def compute_purity(self, labels, actual_labels):
        purity_dict = {}
        for i in range(self.k):
            cluster_labels = actual_labels[labels == i]
            label_counts = pd.Series(cluster_labels).value_counts(normalize=True)
            purity_dict[i] = ", ".join([f"{perc * 100:.2f}% {label}" for label, perc in label_counts.items()])
        return purity_dict

    def load_data(self, filepath):
        df = pd.read_csv(filepath, header=None)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        return data, labels

def main():
    k = int(input("Enter the number of clusters (k): "))
    clustering = KMeansClustering(k)
    data, actual_labels = clustering.load_data('iris_kmeans.txt')

    labels, centroids = clustering.fit(data, actual_labels)

if __name__ == "__main__":
    main()
