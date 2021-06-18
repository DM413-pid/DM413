import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import pandas as pd
from utilis1 import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.distributions.distribution as dis

use_gpu = torch.cuda.is_available()
id = 3
torch.cuda.set_device(id)
def distribution_calibration1(query, base_means, base_cov, n_b_match, alpha=1e-6):
    mean = base_means * n_b_match.unsqueeze(1)
    mean = torch.sum(mean, 0)
    mean = torch.cat([torch.unsqueeze(mean, 0), torch.unsqueeze(query, 0)], 0)
    cov = base_cov.permute(2, 1, 0) * n_b_match
    cov = cov.permute(2, 1, 0)
    calibrated_mean = torch.mean(mean, axis=0)
    calibrated_cov = torch.mean(cov, axis=0) *5
    # calibrated_mean = calibrated_mean.cpu().detach().numpy()
    # calibrated_cov = calibrated_cov.cpu().detach().numpy()

    return calibrated_mean, calibrated_cov

def distribution_calibration(query, base_means, base_cov, k,alpha=0):
    base_means = torch.tensor(base_means).cuda()
    base_cov = torch.tensor(base_cov).cuda()
    dist = []
    for i in range(base_means.shape[0]):
        dist.append(torch.norm(query-base_means[i]))
    _, index = torch.topk(torch.tensor(dist), k, dim=0, largest=False, sorted=True, out=None)
    mean = torch.cat([base_means[index, ], torch.unsqueeze(query, 0)], 0)
    calibrated_mean = torch.mean(mean, axis=0)
    calibrated_cov = torch.mean(base_cov[index, ], axis=0)+alpha

    return calibrated_mean, calibrated_cov

def compute_optimal_transport(M, r, c, lam, epsilon=1e-6):
    r = r.cuda()
    c = c.cuda()
    n_runs, n, m = M.shape
    P = torch.exp(- lam * M)
    P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)

    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    # normalize this matrix
    while torch.max(torch.abs(u - P.sum(2))) > epsilon:
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        if iters == maxiters:
            break
        iters = iters + 1
    return P, torch.sum(P * M)

if __name__ == '__main__':
    # ---- data loading
    dataset = 'miniImagenet'
    #dataset = 'CUB'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 1
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    n_base_class =64


    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)

    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg).cuda()  # torch.Size([10000, 5, 16, 640])
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)  # [10000, 80, 640]
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples).cuda() #[10000, 80]

    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_bias = []
    # base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    base_features_path = "./checkpoints/%s/base_features.plk" % dataset


    # 计算每个base类特征的均值和方差
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])  # (600, 640)
            beta = 0.5
            feature = np.power(feature[:, ], beta)
            mean = np.mean(feature, axis=0)
            cov = np.cov(feature.T)
            diag = np.diag(cov)
            base_means.append(mean)
            base_cov.append(cov)
    base_means = torch.tensor(base_means).cuda()
    base_cov = torch.tensor(base_cov).cuda()


    # # ---- classification for each task
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))
    tqdm_gen = tqdm(range(n_runs))
    for i in tqdm_gen:
        support_data = ndatas[i][:n_lsamples]
        support_label = labels[i][:n_lsamples]
        query_data = ndatas[i][n_lsamples:]
        query_label = labels[i][n_lsamples:]
        # ---- Tukey's transform
        beta = 0.5
        support_data = torch.pow(support_data[:, ], beta)
        query_data = torch.pow(query_data[:, ], beta)


        # ---- distribution calibration and feature sampling
        sampled_data = []
        sampled_label = []
        num_sampled = int(500/n_shot)
        # base 和 novel 类的匹配
        dist = (support_data.unsqueeze(1) - base_means.unsqueeze(0)).norm(dim=2).unsqueeze(0)
        #dist = torch.cosine_similarity(support_data.unsqueeze(1), base_means.unsqueeze(0), -1).unsqueeze(0)
        r = torch.ones(1, n_ways*n_shot)  # 行和约束
        mean_support = torch.mean(support_data, 0)
        c = (mean_support.unsqueeze(0) - base_means.unsqueeze(1)).norm(dim=2)
        #c = torch.cosine_similarity(mean_support.unsqueeze(0), base_means.unsqueeze(1), -1)
        c = n_ways * c/torch.sum(c, 0)
        c = c.view(1,-1)
        n_b_match, _ = compute_optimal_transport(dist, r, c, lam=0, epsilon=1e-6)
        n_b_match = n_b_match.view(n_ways, n_base_class).to(torch.float32)

        for j in range(n_lsamples):
            mean, cov = distribution_calibration1(support_data[j], base_means, base_cov, n_b_match[j,])
            distribute = MultivariateNormal(mean, cov.to(torch.float32))
            samp_data = distribute.rsample(sample_shape=torch.Size([num_sampled]))
            sampled_data.append(samp_data)
            sampled_label.extend([support_label[j]] * num_sampled)
        sampled_data = torch.tensor([item.cpu().detach().numpy() for item in sampled_data]).cuda()
        sampled_data = sampled_data.reshape(n_ways * n_shot * num_sampled, -1)
        sampled_label = torch.tensor(sampled_label).cuda()
        X_aug = torch.cat([support_data, sampled_data], 0)
        Y_aug = torch.cat([support_label, sampled_label], 0)
        #X_aug = scaleEachUnitaryDatas(X_aug)
        #query_data = scaleEachUnitaryDatas(query_data)

        a_data = torch.cat([X_aug, query_data], 0)
        a_label = torch.cat([Y_aug, query_label], 0)

        m1 = ['*'] * 5
        m2 = ['o'] * 2500
        m3 = ['s'] * 75
        mark = np.concatenate([m1, m2, m3]).tolist()
        c1 = np.concatenate([['#BC3C29FF', '#20854EFF', '#E18727FF', '#0072B5FF', '#7876B1FF'][:]])
        c2 = ['#BC3C29FF', '#20854EFF', '#E18727FF', '#0072B5FF', '#7876B1FF', ] * 15
        c2 = np.concatenate([c2[:]])
        c3 = [i * 500 for i in [['#BC3C29FF'], ['#20854EFF'], ['#E18727FF'], ['#0072B5FF'], ['#7876B1FF']]]
        c3 = np.concatenate(c3[:])
        col = np.concatenate([c1, c3, c2])


        model = TSNE(learning_rate=50)

        tsne_features = model.fit_transform(a_data.cpu().numpy())
        xs = tsne_features[:, 0]
        ys = tsne_features[:, 1]
        #plt.scatter(xs, ys, c=Y_aug.cpu().numpy(), marker=mark)
        for i in range(len(col)):
            plt.plot(xs[i], ys[i], marker=mark[i], color=col[i], markersize=3)

        plt.show()

        # ##标准化
    #
    #
    #    # ---- train classifier
    #     classifier = LR(max_iter=1000).fit(x=X_aug, y=Y_aug)
    #     # classifier = LR_1(max_iter=1000).fit(x=[X_aug,X_relation], y=Y_aug)
    #     predicts = classifier.predict(query_data)
    #     # predicts = classifier.predict([query_data,query_relation])
    #     acc = np.mean(predicts.cpu().numpy() == query_label.cpu().numpy())
    #     tqdm_gen.set_description('run {}, acc={:.4f}'.format(i, acc))
    #     acc_list.append(acc)
    #     # test = pd.DataFrame(data=acc_list)  # 数据有三列，列名分别为one,two,three
    #     # test.to_csv('ACC_DC_1.csv', encoding='gbk')
    # print('%s %d way %d shot  ACC : %f' % (dataset, n_ways, n_shot, float(np.mean(acc_list))))
    #
    #
