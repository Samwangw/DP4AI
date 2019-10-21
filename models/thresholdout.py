import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle
import datetime

TEST = False
DEBUG = False


class Thresholdout:

    def __init__(self):
        self.num_data = 50  # generated data size
        self.num_feature = 1  # feature size
        self.learning_rate = 0.001
        self.num_iter = 500
        self.min_err = 0.0001
        self.tolerance = 1.0 / np.sqrt(self.num_data)
        self.threshold = 4.0 / np.sqrt(self.num_data)
        self.train = []  # train set
        self.train_labels = []  # train set labels
        self.train_labels_noise = []
        self.thresholdout = []  # thresholdout set
        self.thresholdout_labels = []  # thresholdout set labels
        self.fresh = []
        self.fresh_labels = []

    def generate_test_data(self):
        self.train = np.zeros((self.num_data, self.num_feature))
        self.train_labels = np.zeros((self.num_data, 1))
        self.thresholdout = np.zeros((self.num_data, self.num_feature))
        self.thresholdout_labels = np.zeros((self.num_data, 1))
        for i in range(0, self.num_data):
            for j in range(0, self.num_feature):
                if j == 0:
                    self.train[i][j] = 0.1
                    self.thresholdout[i][j] = 0.1
                else:
                    self.train[i][j] = i / self.num_data * 2
            self.train_labels[i][0] = np.random.randint(-1, 1)
        file = open('train.pkl', 'wb')
        pickle.dump(self.train, file)
        file.close()
        file = open('trainlabels.pkl', 'wb')
        pickle.dump(self.train_labels, file)
        file.close()

    def generate_data(self):
        self.train = []
        self.train_labels = []
        # generate data for train and thresholdout
        if TEST:
            f = open("train.pkl", "rb")
            self.train = pickle.load(f)
            f.close()
            f = open("trainlabels.pkl", "rb")
            self.train_labels = pickle.load(f)
            f.close()
            self.thresholdout = np.random.random((self.num_data, self.num_feature))
            self.thresholdout_labels = np.random.randint(-1, 1, (self.num_data, 1))
        else:
            self.train = np.random.normal(0, 1, (self.num_data, self.num_feature))
            self.train_labels = np.random.normal(0, 1, (self.num_data, 1))
            self.train_labels_noise = self.train_labels + np.random.normal(0, self.tolerance, (self.num_data, 1))
            self.thresholdout = np.random.normal(0, 1, (self.num_data, self.num_feature))
            self.thresholdout_labels = np.sign(np.random.normal(0, 1, (self.num_data, 1)))
            self.fresh = np.random.normal(0, 1, (self.num_data, self.num_feature))
            self.fresh_labels = np.sign(np.random.normal(0, 1, (self.num_data, 1)))

    def linear_regression(self):
        initial_b = 0
        initial_m = np.zeros(self.num_feature)

        [b1, m1] = self.model1(self.train, self.train_labels, initial_b, initial_m, self.learning_rate,
                               self.num_iter)
        # print(b1,m1)
        [b2, m2] = self.model3(self.train, self.train_labels_noise, initial_b, initial_m, self.learning_rate,
                               self.num_iter)
        [b3, m3] = self.optimizer4(self.train, self.train_labels, initial_b, initial_m, self.learning_rate,
                                   self.num_iter)
        return [self.compute_precision(self.train, self.train_labels, b1, m1),
                self.compute_precision(self.train, self.train_labels,
                                       b2, m2), self.compute_precision(self.train, self.train_labels, b3, m3)]

    def compute_final_loss(self, b, m):
        for i in range(0, self.num_data):
            err = self.train_labels[i] - np.dot(m, self.train[i]) - b
            print(i, ': ', err)

    def generate_noise(self):
        noise = np.random.normal(0, self.tolerance)
        return noise

    def model1(self, sample, sample_labels, initial_b, initial_m, learning_rate, num_iter):
        b = initial_b
        m = initial_m
        last_err = 0
        i = 0
        while i < self.num_iter:
            b, m = self.compute_gradient(sample, sample_labels, b, m, learning_rate)
            err = self.computer_loss(sample, sample_labels, b, m)
            if abs(last_err - err) < self.min_err:
                break
            i = i + 1
            last_err = err
        return [b, m]

    def model2(self, initial_b, initial_m, learning_rate, num_iter):
        b = initial_b
        m = initial_m
        last_err = 0
        i = 0
        while i < self.num_iter:
            b, m = self.compute_gradient(b, m, learning_rate)
            err1 = self.computer_loss(self.train, self.train_labels, b, m)
            err2 = self.computer_loss(self.thresholdout, self.thresholdout_labels, b, m)
            if np.abs(err1 - err2) >= np.abs(self.generate_noise()):
                err = err2 + self.generate_noise()
            else:
                err = err1
            if err < self.min_err:
                break
            # after = computer_error(b, m, data)
            if abs(last_err - err) < self.min_err:
                break
            i = i + 1
            last_err = err
        return [b, m]

    def model3(self, sample, sample_labels, initial_b, initial_m, learning_rate, num_iter):
        b = initial_b
        m = initial_m
        last_err = 0
        i = 0
        while i < self.num_iter:
            b, m = self.compute_gradient(sample, sample_labels, b, m, learning_rate)
            err = self.computer_loss(sample, sample_labels, b, m)
            if abs(last_err - err) < self.min_err:
                break
            i = i + 1
            last_err = err
        return [b, m]

    def optimizer4(self, sample, sample_labels, initial_b, initial_m, learning_rate, num_iter):
        b = initial_b
        m = initial_m
        last_err = 0
        i = 0
        while i < self.num_iter:
            b, m = self.compute_gradient_noise(sample, sample_labels, b, m, learning_rate)
            err = self.computer_loss(sample, sample_labels, b, m)
            if abs(last_err - err) < self.min_err:
                break
            i = i + 1
            last_err = err
        return [b, m]

    def compute_gradient(self, sample, sample_labels, b_cur, m_cur, learning_rate):
        b_gradient = 0
        m_gradient = np.zeros(self.num_feature)

        #
        # 偏导数， 梯度
        for i in range(0, self.num_data):
            x = sample[i]
            y = sample_labels[i]
            pre = (np.dot(m_cur, x)) + b_cur
            b_gradient += -(2.0 / self.num_data) * (y - pre)
            m_gradient += -(2.0 / self.num_data) * x * (y - pre)
        new_b = b_cur - (learning_rate * b_gradient)
        new_m = m_cur - (learning_rate * m_gradient)
        return [new_b, new_m]

    def compute_gradient_noise(self, sample, sample_labels, b_cur, m_cur, learning_rate):
        b_gradient = 0
        m_gradient = np.zeros(self.num_feature)

        #
        # 偏导数， 梯度
        for i in range(0, self.num_data):
            x = sample[i]
            y = sample_labels[i]
            pre = (np.dot(m_cur, x)) + b_cur + self.generate_noise()
            b_gradient += -(2.0 / self.num_data) * (y - pre)
            m_gradient += -(2.0 / self.num_data) * x * (y - pre)
        new_b = b_cur - (learning_rate * b_gradient)
        new_m = m_cur - (learning_rate * m_gradient)
        return [new_b, new_m]

    def computer_loss(self, sample, sample_labels, b, m):
        totalError = 0
        for i in range(0, self.num_data):
            x = sample[i]
            y = sample_labels[i]
            pre = np.dot(m, x) + b
            totalError += (y - pre) ** 2
        return totalError / self.num_data

    def compute_precision(self, data, labels, b, m):
        hit = 0.0
        for i in range(0, self.num_data):
            x = data[i]
            y = labels[i]
            pre = np.dot(m, x) + b
            hit += np.abs(pre - y)
        return hit / self.num_data

    def run(self):
        num_repeat = 30
        step = 20
        x = np.zeros(step)
        err_std = np.zeros(step)
        err_thresh = np.zeros(step)
        err_fresh = np.zeros(step)
        for repeat in range(num_repeat):
            for i in range(step):
                self.num_feature = (i + 1) * 25
                print("Round", repeat, ':', self.num_feature, 'features', datetime.datetime.now())
                # self.generate_test_data()
                self.generate_data()
                [err1, err2, err3] = self.linear_regression()
                x[i] = self.num_feature
                err_std[i] += err1 / num_repeat
                err_thresh[i] += err2 / num_repeat
                err_fresh[i] += err3 / num_repeat
        # write out results
        try:
            f = open('Thresholdout _' + "N" + str(self.num_data) + "_iter" + str(num_repeat) + ".txt", 'w')
            f.write(str(x))
            f.write("\n")
            f.write(str(err_std))
            f.write("\n")
            f.write(str(err_thresh))
            f.write("\n")
            f.write(str(err_fresh))
            f.write("\n")
            f.close()
        except:
            print("write fail")

        # draw the figure
        try:
            plt.plot(x, err_std, color='red', label='Std', linestyle='-.')
            plt.plot(x, err_thresh, color='blue', label='fix noise on labels', linestyle=':')
            plt.plot(x, err_fresh, color='green', label='dynamic noise on labels', linestyle='--')
            plt.title(u'Compare')
            plt.xlabel(u'Feature')
            plt.ylabel(u'Error')
            plt.legend()
            filename = 'Thresholdout _' + "N" + str(self.num_data) + "_iter" + str(num_repeat) + '.png'
            plt.savefig(filename)
            plt.close()
        except:
            print("draw fail")
