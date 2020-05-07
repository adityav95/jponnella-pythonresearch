#This script demoes the creation of functions for the kNN algorithm

#Calculating euclidean distance between two points in n-dimensional vectors

def distance(p1,p2):
    """Finds the Euclidean distance between 2 points of the same dimensions. Note that p1 and p2 must be input as an array."""
    import numpy as np
    x = np.sqrt(sum(np.power(p2-p1,2)))
    return(x)

p1 = np.array([1, 1])
p2 = np.array([4, 4])

distance(p1,p2)
	# 4.242640687119285

p1 = np.array([1,2,3,4,5])
p2 = np.array([11,12,13,14,15])

distance(p1,p2)
	# 22.360679774997898

#Defining a function to determine the highest voter in a list of voters where each value is the voter's unique type/class

def majority_vote(votes):
    """Return the most common element in a list of votes. If there are multiple voters with the same element, one of them is returned at random."""
    vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1

    winners = []
    max_choice = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_choice:
            winners.append(vote)
            
    import random
    return(random.choice(winners))

 votes = [1,2,3,1,2,3,1,2,3,3,3,3]
 majority_vote(votes)
 	# 3

#Different way to define the majority_vote function is to use the mode function

def majority_vote(votes):
    """Return the most common element in a list of votes. If there are multiple voters with the same element, the lowest value of them is returned."""
    import scipy.stats as ss
    mode, count = ss.mstats.mode(votes)

#Find nearest neighbors out of all given points
def find_nearest_neighbors(p, points, k):
	"""Find the k nearest neighbors of point p in array points. If multiple points are the equally far, the ones with lower values in lower dimensions are preferred. For example, if (2,2) and (2,4) are equally distant, (2,2) would be chosen."""
	import numpy as np
	distances = np.zeros(points.shape[0])
	for i in range(len(distances)):
		distances[i] = distance(p,points[i])
	ind = np.argsort(distances)
	return ind[0:k]

#Predicting kNN:
def knn_predict(p, points, outcomes, k):
	"""This function predicts the nearest neighbor value given a set of points and their class."""
	ind = find_nearest_neighbors(p, points, k)
	return majority_vote(outcomes[ind])

import numpy as np
points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
outcomes = np.array([0,0,0,0,1,1,1,1,1])
knn_predict(np.array([2.5,2.7]),points, outcomes, 2)
	# 1

knn_predict(np.array([1.0,2.7]),points, outcomes, 2)
	# 0



#Generating synthetic data to test the algorithm
def generate_synth_data(n):
	"""This function generates n points of two classes - 0 and 1 and generates the 2 sets of points in 2D space and their respective classes."""
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1,n)))
    return (points, outcomes)

n = 50

(points, outcomes) = generate_synth_data(n)

#Plotting the generated data
import matplotlib.pyplot as plt
plt.figure()
plt.plot(points[:n,0], points[:n,1], "bo")
plt.plot(points[n:,0], points[n:,1], "ro")

#Making a prediction grid
def make_prediction_grid(predictors, outcomes, limits, h, k):
	"""Classify each point on a predictor grid."""
	(x_min, x_max, y_min, y_max) = limits
	xs = np.arange(x_min, x_max, h)
	ys = np.arange(y_min, y_max, h)
	xx, yy = np.meshgrid(xs, ys)

	prediction_grid = np.zeros(xx.shape, dtype = int)
	for i,x in enumerate(xs):
		for j, y in enumerate(ys):
			p = np.array([x, y])
			prediction_grid[j,i] = knn_predict(p, predictors, outcomes, k)

	return(xx, yy, prediction_grid)


#Plot prediction grid
def plot_prediction_grid(xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)



#PLotting the data for 2 sets of data points:
k = 5; limits = (-3,4,-3,4); h = 0.1; filename = "knn_synth_5.pdf"; predictors = points;
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)


k = 50; limits = (-3,4,-3,4); h = 0.1; filename = "knn_synth_50.pdf"; predictors = points;
(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)


#Testing with the iris dataset and comparing our algo with SciKitLearn algo:
from sklearn import datasets
iris = datasets.load_iris()

k = 5; limits = (4,8,1,5); h = 0.1; filename = "iris_5.pdf"; predictors = iris.data[:,0:2]; outcomes = iris.target;

#Method 1 for plotting the data:
plt.plot(predictors[outcomes==0][:,0], predictors[outcomes==0][:,1], "ro")
plt.plot(predictors[outcomes==1][:,0], predictors[outcomes==1][:,1], "bo")
plt.plot(predictors[outcomes==2][:,0], predictors[outcomes==2][:,1], "yo")

#Method 2 for plotting the data:

(xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, prediction_grid, filename)

#Testing using the sklearn function:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(predictors, outcomes)
sk_prediction = knn.predict(predictors) #These are the predictions using kNN on the Iris training set using sklearn


sk_prediction.shape
	# (150,)

my_predictions = np.array([knn_predict(p, predictors, outcomes, 5) for p in predictors]) #These are the predictions using kNN on the Iris training set using our algorithm

#Checking which of the algorithms is better in this case on the training set
my_pred_check = np.array(my_predictions==outcomes)
print(100*np.mean(my_pred_check))
	#84.666667 when k = 5; 83.333334 when k = 50

sklearn_check = np.array(sk_prediction==outcomes)
print(100*np.mean(sklearn_check))
	#83.333334