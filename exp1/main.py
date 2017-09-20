from numpy import loadtxt, where, e, reshape, transpose, log, zeros, array
from pylab import scatter, xlabel, ylabel, legend, show

data = loadtxt('./data/data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature1/Exam 1 score')
ylabel('Feature2/Exam 2 score')
legend(['Fail', 'Pass'])
show()


def sigmoid(X):
    den = 1.0 + e ** (-1.0 * X)
    gz = 1.0 / den
    return gz


def compute_cost(theta, X, y):
    m = X.shape[0]
    theta = reshape(theta, (len(theta), 1))

    J = (1. / m) * (
        -transpose(y).dot(log(sigmoid(X.dot(theta)))) - transpose(1 - y).dot(log(1 - sigmoid(X.dot(theta)))))

    grad = transpose((1. / m) * transpose(sigmoid(X.dot(theta)) - y).dot(X))
    return J[0][0]


def compute_grad(theta, X, y):
    theta.shape = (1, 3)
    grad = zeros(3)
    h = sigmoid(X.dot(theta.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * -1
    theta.shape = (3,)
    return grad


def predict(theta, X):
    m, n = X.shape
    p = zeros(shape=(m, 1))
    h = sigmoid(X.dot(theta.T))
    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0
    return p


p = predict(array(theta), it)
print'Train Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0)
