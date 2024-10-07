import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    """
    This function computes the parameters w of a linear plane which best fits
    the training dataset.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    
    Returns
    -------
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of the line computed by the pocket
        algorithm that best separates the two classes of training data points.
        The dimensions of this vector is (d+1) as the offset term is accounted
        in the computation.
    """
    # Add a column of 1's to X_train
    X = np.hstack((np.ones((np.size(X_train, 0), 1)), X_train))

    # Compute w using least squares method given by w = inv(X_tr * X) * X_tr * y_train
    # Step 1: Compute the pseudoinverse (Moore-Penrose) of X to account for when X is rank-deficient
    # If X has linearly independent columns, its pseudo-inverse is equivalent to X_pinv = inv(X_tr * X) * X_tr 
    X_pinv = np.linalg.pinv(X)
    # Step 2: Compute w = X_pinv * y_train
    w = np.matmul(X_pinv, y_train)

    return w

    pass


def mse(X_train, y_train, w):
    """
    This function finds the mean squared error introduced by the linear plane
    defined by w.

    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        the ith component represents the output observed in the training set
        for the ith row in X_train matrix which corresponds to the ith input
        feature. Each element in y_train takes a Real value.
    w: numpy.ndarray with shape (d+1,)
        represents the coefficients of a linear plane.

    Returns
    -------
    avgError: float
        he mean squared error introduced by the linear plane defined by w.
    """

    # TODO: add your implementation here
    # HINT: You may find the implemented pred function useful. The pred function 
    # finds the prediction by the linear regression model defined by w for the
    # input datapoint x_i.

    # Add a column of 1's to X_train
    X = np.hstack((np.ones((np.size(X_train, 0), 1)), X_train))

    # For each sample, calculate the squared error as (pred(X[i], w) - y_train[i])^2
    N = np.size(X, 0)
    total_squared_err = 0
    for i in range(N):
        total_squared_err += (pred(X[i], w) - y_train[i]) ** 2
    
    # Divide by the total squared error number of samples to obtain mean squared error
    return total_squared_err / N


    pass


def pred(x_i, w):
    """
    This function finds finds the prediction by the classifier defined by w.

    Parameters
    ----------
    x_i: numpy.ndarray with shape (d+1,)
        Represents the feature vector of (d+1) dimensions of the ith test
        datapoint.
    w: numpy.ndarray with shape (d+1,)
        Represents the coefficients of a linear plane.
    
    Returns
    -------
    pred_i: int
        The predicted class.
    """

    pred_i = np.dot(x_i, w)
    return pred_i


def test_SciKit(X_train, X_test, y_train, y_test):
    """
    This function will output the mean squared error on the test set, which is
    obtained from the mean_squared_error function imported from sklearn.metrics
    library to report the performance of the model fitted using the 
    LinearRegression model available in the sklearn.linear_model library.
    
    Parameters
    ----------
    X_train: numpy.ndarray with shape (N,d)
        Represents the matrix of input features where N is the total number of 
        training samples and d is the dimension of each input feature vector.
    X_test: numpy.ndarray with shape (M,d)
        Represents the matrix of input features where M is the total number of
        testing samples and d is the dimension of each input feature vector.
    y_train: numpy.ndarray with shape (N,)
        The ith component represents output observed in the training set for the
        ith row in X_train matrix which corresponds to the ith input feature.
    y_test: numpy.ndarray with shape (M,)
        The ith component represents output observed in the test set for the
        ith row in X_test matrix which corresponds to the ith input feature.
    
    Returns
    -------
    error: float
        The mean squared error on the test set.
    """

    # TODO: initiate an object of the LinearRegression type. 
    model = LinearRegression()
    
    # TODO: run the fit function to train the model. 
    model.fit(X_train, y_train)
    
    # TODO: use the predict function to perform predictions using the trained
    # model. 
    pred = model.predict(X_test)
    
    # TODO: use the mean_squared_error function to find the mean squared error
    # on the test set. Don't forget to return the mean squared error.
    mse = mean_squared_error(y_test, pred)

    return mse
    pass


def subtestFn():
    """
    This function tests if your solution is robust against singular matrix.
    X_train has two perfectly correlated features.
    """

    X_train = np.asarray([[1, 2],
                          [2, 4],
                          [3, 6],
                          [4, 8]])
    y_train = np.asarray([1, 2, 3, 4])
    
    try:
      w = fit_LinRegr(X_train, y_train)
      print("weights: ", w)
      print("NO ERROR")
    except:
      print("ERROR")


def testFn_Part2():
    """
    This function loads diabetes dataset and splits it into train and test set.
    Then it finds and prints the mean squared error from your linear regression
    model and the one from the scikit library.
    """

    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    
    w = fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e = mse(X_test, y_test, w)
    
    #Testing Part 2b
    scikit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print(f"Mean squared error from Part 2a is {e}")
    print(f"Mean squared error from Part 2b is {scikit}")


if __name__ == "__main__":
    print (f"{12*'-'}subtestFn{12*'-'}")
    subtestFn()

    print (f"{12*'-'}testFn_Part2{12*'-'}")
    testFn_Part2()

# The mse computed by our own implementation is almost identical to the mse computed by scikit-learn:
# Mean squared error from Part 2a is 2980.23720310032
# Mean squared error from Part 2b is 2980.2372031003188