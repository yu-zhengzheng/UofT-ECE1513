import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm


def load_data():
    X = np.load('data2D.npy')
    valid_batch = int(len(X) / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(len(X))
    np.random.shuffle(rnd_idx)
    val_data = X[rnd_idx[:valid_batch]]
    data = X[rnd_idx[valid_batch:]]

    return data, val_data


def truncated_normal(size, threshold=1):
    # This function generates a truncated normal distribution
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def distanceFunc(X, MU):
    X1 = torch.unsqueeze(X, -1)
    MU1 = torch.unsqueeze(MU.T, 0)

    pair_dist = torch.sum((X1 - MU1) ** 2, 1)

    return pair_dist


def log_GaussPDF(X, mu, sigma):
    dim = X.shape[-1]

    Pi = torch.tensor(float(np.pi))
    sigma_2 = (torch.square(sigma)).T  # 1 X K
    diff = distanceFunc(X, mu)  # N X K

    log_PDF = diff / sigma_2  # N X K
    log_PDF += dim * torch.log(2 * Pi)
    log_PDF += 2 * dim * torch.log(sigma_2)
    log_PDF *= -0.5  # N X K

    return log_PDF


def log_posterior(log_PDF, log_pi):
    log_joint = log_PDF + log_pi.T  # N X K
    log_marginal = torch.logsumexp(log_joint, dim=1)  # N x 1

    return log_joint, log_marginal


def train_gmm(train_data, test_data, k=5, epoch=1000):
    # Load the data
    X_train = torch.from_numpy(train_data)
    X_test = torch.from_numpy(test_data)

    # Initialize logits
    logits = truncated_normal(k, threshold=1)
    logits = torch.rand(k, requires_grad=True)

    # Initialize the mean
    mu = torch.randn((k, X_train.shape[1]), requires_grad=True)

    # Initialize sigma.
    sigma = np.abs(truncated_normal((k, 1), threshold=1))
    sigma = torch.tensor(sigma, requires_grad=True)
    optimizer = torch.optim.Adam([logits, mu, sigma], lr=0.1,
                                 betas=(0.9, 0.99), eps=1e-5)
    # Train the model
    for i in range(epoch):
        logpi = F.log_softmax(logits)

        # Fill in the blank (2lines below) of the functions log_GaussPDF and log_posterior
        log_PDF = log_GaussPDF(torch.from_numpy(train_data), mu, sigma)  # fill in here
        _, log_marginal = log_posterior(log_PDF, logpi)  # fill in here

        # Compute the marginal mean.
        loss = -log_marginal.mean()

        # Update the paramenters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Perform on Test data
    logpi = F.log_softmax(logits)

    # log_GaussPDF and log_posterior
    log_PDF = log_GaussPDF(torch.from_numpy(test_data), mu, sigma)  # fill in here
    log_joint_test, log_marginal = log_posterior(log_PDF, logpi)  # fill in here
    test_loss = -log_marginal.mean()

    # Conver to numpy:
    test_loss = test_loss.detach().numpy()
    log_joint_test = log_joint_test.detach().numpy()
    pi = torch.exp(logpi).detach().numpy()
    mu = mu.detach().numpy()
    sigma = sigma.detach().numpy()

    return test_loss, log_joint_test, pi, mu, sigma


def test_GMM(k=5):
    train_data, test_data = load_data()
    test_loss, log_joint_test, pi, mu, sigma = train_gmm(train_data, test_data, k)

    index = log_joint_test.argmax(axis=1)
    index = index.reshape(len(index), 1)
    new_X = np.concatenate((test_data, index), axis=1)

    color_list = ['g', 'b', 'm', 'y', 'c']
    for i in range(len(mu)):
        tmp = new_X[new_X[..., -1] == i]
        plt.scatter(tmp[:, 0], tmp[:, 1], c=color_list[i])
    plt.scatter(mu[:, 0], mu[:, 1], s=300, c='r', marker='+')
    # visualize the result of GMM test for report
    plt.show()


test_GMM(k=5)