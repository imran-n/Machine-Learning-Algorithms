import numpy as np
import math
import collections

#Importing and instantiating Iris datastes
from sklearn.datasets import load_iris
iris_dataset = load_iris()


class KNearestNeighbor:



    def __init__(self , k):
        self.k = k

    def algo(self, features, labels, testDataset):

        '''
        In that algorithm, first I've checked the shape of the data and find out how many times I have to check the distance, mean's how many training results are there.
        And, clculated the distance between all the points of the training data vs the test data and stored them into a list one by one.

        Then at the end, I've written a while loop which loops k times. for every time:
            1. Get the index of the minimum distance and find the level of this index. Then store the level inside another list called labelsOfMinimun.
            2. Remove the minimum elements. Because, now after removing this value, the next minimum value will be the second minimum value before.

        Than I've used counter to count and return a dictionary of which label got selected for how many times.

        Finally, this function returns the maximum used label [in numeric data]


        '''

        self.distances = []
        for i in range(features.shape[0]):
            distance_square = 0
            flistUse = features[i]

            for j in range(features.shape[1]):
                distance_square += math.pow((flistUse[j] - testDataset[j]), 2)
            self.distances.append(math.sqrt(distance_square))

        labelsOfMinimum = []
        while(self.k > 0):
            labelsOfMinimum.append(labels[self.distances.index(min(self.distances))])
            self.distances.remove(min(self.distances))
            self.k -= 1

        px = collections.Counter(labelsOfMinimum)

        return px.most_common()[0][0]





knn = KNearestNeighbor(k = 100)   #Creating an object of KNearestNeighbor class which contains the algorithm.
X_new = [4,0.2,3,5]   #A random data to test the model.
predicted_index = knn.algo(iris_dataset.data, iris_dataset.target, X_new)   #Run the algorithm
print(iris_dataset.target_names[predicted_index])  #Logging the output from algorithm.