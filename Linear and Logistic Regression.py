# Week 5: Fitting models to data
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size = n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc = 0, scale = 1, size = n)


plt.figure()
plt.plot(x,y,"o")
xx = np.array([0,10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title("Regression Line")


# Computing RSS and estimating it for the data generated above
def compute_rss(y_estimate, y):
  return sum(np.power(y-y_estimate, 2))

def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x
  
rss = compute_rss(estimate_y(x, beta_0, beta_1), y)

# Trying to fit a simple least squares based regression to the data generated above
import statsmodels.api as sm     # Package has stuff similar to modelling in R
mod = sm.OLS(y,x)
model_1 = mod.fit()
print(model_1.summary())
	# The above model uses the original x which has only the actual x values and no column with constants i.e. '1' and hence gives a model passing through the origin i.e. only slope and no intercept


X = sm.add_constant(x) # This method adds a constant to the input dataset which can be used to estimate the intercept
mod = sm.OLS(y,X)
model_1 = mod.fit()
print(model_1.summary()) #Much closer to the actual values with slope = 1.9685 and intercept = 5.2370

# Trying out multi-variable regression

n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1

np.random.seed(1)

x1 = 10 * ss.uniform.rvs(size=n)
x2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x1 + beta_2 * x2 + ss.norm.rvs(loc = 0, scale = 1, size = n)

X = np.stack([x1, x2], axis = 1) # Converting the input vectors to a matrix

# Plotting a 3D plot to visualize the data (NOTE: Needs some crazy stuff to get mpl_toolkits working, so never actually ran this bit of code)
from mpl_toolkits.mplot3D import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0], X[:,1],y, c=y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")


# Linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept = True)
lm.fit(X,y)

X_0 = np.array([2,4])
lm.predict(X_0.reshape(1,-1)) # The reshape command is used to indicate that the array is one observation with 2 variables. If the reshape was (-1,1) that would be 2 observations with 1 variable
lm.score(X,y) # Gives R. sqr. values for (X, y)


# Splitting data into train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size = 0.5, random_state = 1) # Random state is the seed to be used if results are to be replicated


# Generating data for classification
h = 1
sd = 1
n = 50

def gen_data(n,h,sd1,sd2):
	x1 = ss.norm.rvs(-h,sd1,n)
	y1 = ss.norm.rvs(0,sd1,n)
	
	x2 = ss.norm.rvs(h,sd2,n)
	y2 = ss.norm.rvs(0,sd2,n)
	return (x1,y1,x2,y2)

(x1,y1,x2,y2) = gen_data(50,1,1,1.5)
(x1,y1,x2,y2) = gen_data(1000,1.5,1,1.5)

def plot_data(x1,x2,y1,y2):
	 plt.figure(figsize=(8,8))
	 plt.plot(x1,y1,"o", ms=2)
	 plt.plot(x2,y2,"o", ms=2)
	 plt.xlabel("$X_1$")
	 plt.ylabel ("$X_2$")

plot_data(x1,x2,y1,y2)

# Logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

X = np.vstack((np.vstack((x1,y1)).T, np.vstack((x2,y2)).T)) #This line is used to create the dataset used for the logistic regression model
X.shape

n = 1000
y = np.hstack((np.repeat(1,n), np.repeat(2,n)))
y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size = 0.5, random_state = 1)

clf.fit(X_train, y_train)

clf.score(X_train, y_train)
clf.score(X_test, y_test)


# Prediction using the built model
clf.predict_proba(np.array([-2,0]).reshape(1,-1))    # This predicts the probabilities of both classes being true i.e. gives two prob. values for each test observation
clf.predict(np.array([-2,0]).reshape(1,-1))    # This predicts the class itself i.e. the max of the two probabilities

