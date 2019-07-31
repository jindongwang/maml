import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] [%(name)s] %(message)s')

logger = logging.getLogger('main')


class MAML(object):
    def __init__(self):
        self.n_task = 10
        self.dim = 20
        self.n_sample = 100
        self.epoch = 10000
        self.lr_inner = 0.0001
        self.lr_meta = 0.0001
        self.theta = np.random.normal(size=self.dim).reshape(self.dim, 1)

    def sample_data(self, n):
        x = np.random.rand(n, self.dim)
        y = np.random.choice([0, 1], size=n, p=[.5, .5]).reshape(n, 1)
        return x, y

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def model(self, x, W):
        return self.sigmoid(np.matmul(x, W))

    def gradient(self, x, y, y_pred):
        loss = -((np.matmul(y.T, np.log(y_pred)) + np.matmul(1 -
                                                             y.T, np.log(1 - y_pred))) / self.n_sample)[0][0]
        grad = np.matmul(x.T, (y_pred - y)) / self.n_sample
        return grad, loss

    def train(self):
        x_test, y_test = self.sample_data(self.n_sample)
        losses, accs = [], []
        for e in range(self.epoch):
            self.thetas = []
            # Inner loop, learn each \theta_i'
            for i in range(self.n_task):
                x, y = self.sample_data(self.n_sample)
                y_pred = self.model(x, self.theta)
                grad_i, loss = self.gradient(x, y, y_pred)
                theta_i = self.theta - self.lr_inner * grad_i
                self.thetas.append(theta_i)

            # Meta learning to learn theta
            for i in range(self.n_task):
                x, y = self.sample_data(self.n_sample)
                y_pred = self.model(x, self.thetas[i])
                grad_i, _ = self.gradient(x, y, y_pred)
                self.theta -= self.lr_meta * grad_i

            if (e + 1) % 1000 == 0:
                y_pred = self.model(x_test, self.theta)
                y_label = [0 if x < 0.5 else 1 for x in y_pred]
                acc = np.mean(np.array(y_label)==y_test)
                _, loss = self.gradient(x_test, y_test, y_pred)
                logger.info(
                    '[Epoch {:5d}] [loss: {:.6f}] [acc: {:.4f}]'.format(e + 1, loss, acc))
                losses.append(loss)
                accs.append(acc)
                
        plt.figure(figsize=(8,3))
        plt.subplot(121)
        plt.plot(np.arange(len(accs)),np.array(losses))
        plt.subplot(122)
        plt.plot(np.arange(len(accs)),np.array(accs))
        plt.show()
        return self.theta


if __name__ == '__main__':
    logger.info('Start MAML')

    maml = MAML()
    maml.train()
