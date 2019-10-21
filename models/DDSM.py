import random
import math
import numpy as np
import matplotlib.pyplot as plt


class DDSM:
    def __init__(self, m=20, n=100, g=10, e1=0.5, e2=0.5, max_buy_price=100, max_sell_price=50):
        self.M = m  # sellers
        self.N = n  # buyers
        self.G = g  # the number of groups
        self.e1 = e1
        self.e2 = e2
        self.max_buy_price = max_buy_price
        self.max_sell_price = max_sell_price

        # variables for computation
        self.q = None
        self.b = None
        self.e = 0
        self.l = None
        self.bmax = 0
        self.nmax = 0

    def init_group(self):
        # self.l = [[] for i in range(self.G)]
        # generate quotation profile for buyers and sellers
        self.q = np.random.randint(1, self.max_sell_price, size=self.M)
        self.b = np.random.randint(1, self.max_buy_price, size=self.N)
        self.e = self.e1 + self.e2
        g_dict = dict()
        for i in range(self.N):
            k = np.random.randint(self.G)
            if g_dict.__contains__(k):
                g_dict[k].append(self.b[i])
            else:
                g_dict[k] = []
                g_dict[k].append(self.b[i])
        self.l = list(g_dict.values())
        # self.l[np.random.randint(self.G)].append(self.b[i])
        self.l = np.array(self.l)

        self.bmax = np.max(self.b)
        self.nmax = np.max([len(self.l[i]) for i in range(self.l.shape[0])])

    def basicDDSM(self):
        # compute group bids
        g = self.compute_group_bids()
        [self.q, g] = self.sort_quotations_bids(self.q, g)
        w = self.compute_social_welfare(g)
        [ps, pg, welfare] = self.compute_dist(w)
        return welfare

    def improvedDDSM(self):
        # compute group bids
        g = self.compute_group_bids()
        [self.q, g] = self.sort_quotations_bids(self.q, g)
        w = self.compute_social_welfare(g)
        [key, welfare] = self.compute_improved_dist(w)
        return welfare

    def compute_group_bids(self):
        L = self.l.shape[0]
        self.g_dict = dict()
        g = []
        for i in range(L):
            minb = float('inf')
            for b in self.l[i]:
                if minb > b:
                    minb = b
            bl = minb * len(self.l[i])
            if self.g_dict.__contains__(minb * len(self.l[i])):
                bl += np.random.rand()
                # raise ("Same group bid found. Exit.")
            g.append(bl)
            self.g_dict[bl] = self.l[i]
        return g

    def sort_quotations_bids(self, q, g):
        q = sorted(q)
        g = sorted(g, reverse=True)
        return [q, g]

    def compute_social_welfare(self, g):
        w = dict()
        for ps in range(1, np.max(self.q) + 1):
            for pg in range(ps, self.nmax * self.bmax + 1):
                # find max m to make qm <= ps
                ks = 0
                for m in range(len(self.q)):
                    if self.q[m] <= ps:
                        ks = m
                    else:
                        break
                # find max l to make gl >= pg
                kg = 0
                for l in range(len(g)):
                    if g[l] >= pg:
                        kg = l
                    else:
                        break
                k = min(ks, kg) + 1
                # compute ws
                wso = [self.q[i] for i in range(ks + 1)]
                ws = random.sample(wso, k)
                # compute wg
                wgo = [g[i] for i in range(kg + 1)]
                wg = random.sample(wgo, k)
                w[(ps, pg)] = 0
                for i in range(k):
                    wl = 0
                    for j in range(len(self.g_dict[wg[i]])):
                        wl += self.g_dict[wg[i]][j] - self.q[i]
                    w[(ps, pg)] += wl
        return w

    def compute_dist(self, w):
        pr_m1 = dict()
        w1 = dict()
        for ps in range(1, np.max(self.q) + 1):
            ppg = 0
            for potential_pg in range(self.bmax * self.nmax + 1):
                if w.__contains__((ps, potential_pg)) and ppg < w[(ps, potential_pg)]:
                    ppg = w[(ps, potential_pg)]
            w1[ps] = ppg  # find max pg in w(ps,pg)
            delta_w1 = self.nmax * self.bmax - 1
            pr_m1[ps] = math.exp(self.e1 * w1[ps] / (2 * delta_w1))
        sum = np.sum(list(pr_m1.values()))
        for ps in pr_m1.keys():
            pr_m1[ps] /= sum
        ps = np.random.choice(list(pr_m1.keys()), p=list(pr_m1.values()))
        pr_m2 = dict()
        w2 = dict()
        for pg in range(ps, self.nmax * self.bmax + 1):
            w2[(ps, pg)] = w[(ps, pg)]
            delta_w2 = self.nmax * self.bmax - 1
            pr_m2[pg] = math.exp(self.e2 * w[(ps, pg)] / (2 * delta_w2))
        sum = np.sum(list(pr_m2.values()))
        for pg in pr_m2.keys():
            pr_m2[pg] /= sum
        pg = np.random.choice(list(pr_m2.keys()), p=list(pr_m2.values()))
        return [ps, pg, w[(ps, pg)]]

    def compute_improved_dist(self, w):
        delta_w = self.nmax * self.bmax - 1
        pr_m = dict()
        sum = 0
        for ps in range(1, np.max(self.q) + 1):
            for pg in range(ps, self.nmax * self.bmax + 1):
                value = math.exp(((self.e1 + self.e2) * w[(ps, pg)]) / (2 * delta_w))
                pr_m[(ps, pg)] = value
                sum += value
        x = 0
        for (ps, pg) in pr_m.keys():
            pr_m[(ps, pg)] /= sum
            # if x % 500 == 0:
            #     print(x, (ps, pg), w[(ps, pg)], ":", pr_m[(ps, pg)])
            x += 1
        key_list = list(pr_m.keys())
        key_index = np.random.choice([i for i in range(0, len(key_list))], p=list(pr_m.values()))
        return [key_list[key_index], w[key_list[key_index]]]

    def __name__(self):
        return "M" + str(self.M) + "_N" + str(self.N) + "_G" + str(self.G)


def ex1():
    max_repeat = repeat = 1000
    ddsm = None
    basic = dict()
    while repeat > 0:
        print("Round", max_repeat - repeat + 1)
        ddsm = DDSM(m=40, n=5, g=30, max_buy_price=50, max_sell_price=20)
        ddsm.init_group()
        i = 0.1
        while i < 1:
            ddsm.e1 = i
            ddsm.e2 = 1 - ddsm.e1
            try:
                w = ddsm.basicDDSM()
                if basic.keys().__contains__(ddsm.e1):
                    basic[ddsm.e1] += w
                else:
                    basic[ddsm.e1] = w
                i += 0.1
            except:
                ddsm.init_group()
                continue

        repeat -= 1

    for key in basic.keys():
        basic[key] /= max_repeat
    x = list(basic.keys())
    basic_list = list(basic.values())
    # write out results
    f = open('DDSM EX2_' + ddsm.__name__() + "_iter" + str(max_repeat) + ".txt", 'w')
    f.write(str(x))
    f.write("\n")
    f.write(str(basic_list))
    f.close()
    # draw the figure
    plt.plot(x, basic_list, color='red', label='Basic', linestyle='-.')
    plt.title(u'compare different e')
    plt.xlabel(u'e1')
    plt.ylabel(u'Welfare')
    plt.legend()
    filename = 'DDSM EX1_' + ddsm.__name__() + "_iter" + str(max_repeat) + '.png'
    plt.savefig(filename)
    plt.close()


def ex2():
    max_repeat = repeat = 1000
    basic = dict()
    impro = dict()
    ddsm = None
    while repeat > 0:
        print("\rRound", max_repeat - repeat + 1)
        ddsm = DDSM(m=40, n=5, g=30, max_buy_price=50, max_sell_price=20)
        ddsm.init_group()
        init_N = 100
        step = 50
        ddsm.N = init_N
        while ddsm.N <= 1000:
            print("\r-----%d-----" % ddsm.N, end="")
            try:
                w1 = ddsm.basicDDSM()
                if basic.keys().__contains__(ddsm.N):
                    basic[ddsm.N] += w1
                else:
                    basic[ddsm.N] = w1
                w2 = ddsm.improvedDDSM()
                if impro.keys().__contains__(ddsm.N):
                    impro[ddsm.N] += w2
                else:
                    impro[ddsm.N] = w2
                ddsm.N += step
            except:
                ddsm.init_group()
                continue
        repeat -= 1
    for key in basic.keys():
        basic[key] /= max_repeat
        impro[key] /= max_repeat
    x = list(basic.keys())
    basic_list = list(basic.values())
    impro_list = list(impro.values())
    f = open('DDSM EX2_' + ddsm.__name__() + "_iter" + str(max_repeat) + ".txt", 'w')
    f.write(str(x))
    f.write("\n")
    f.write(str(basic_list))
    f.write("\n")
    f.write(str(impro_list))
    f.close()

    plt.plot(x, basic_list, color='red', label='Basic', linestyle='-.')
    plt.plot(x, impro_list, color='blue', label='Improved', linestyle=':')

    plt.title(u'Compare')
    plt.xlabel(u'N')
    plt.ylabel(u'Welfare')

    plt.legend()

    filename = 'DDSM EX2_' + ddsm.__name__() + "_iter" + str(max_repeat) + '.png'
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    ex1()
