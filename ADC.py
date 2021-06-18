import pickle
import numpy as np
import torch
from tqdm import tqdm
from utilis1 import *
import pandas as pd
from torch.distributions.multivariate_normal import MultivariateNormal
## miniImagenet数据集 1- shot 实验

use_gpu = torch.cuda.is_available()
id = 4
torch.cuda.set_device(id)

def distribution_calibration(query, base_means, base_cov, n_b_match, alpha=1e-6):
    mean = base_means * n_b_match.unsqueeze(1)
    mean = torch.sum(mean, 0)
    mean = torch.cat([torch.unsqueeze(mean, 0), torch.unsqueeze(query, 0)], 0)
    cov = base_cov.permute(2, 1, 0) * n_b_match
    cov = cov.permute(2, 1, 0)
    calibrated_mean = torch.mean(mean, axis=0)
    calibrated_cov = torch.mean(cov, axis=0)
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
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_base_class = 64
    n_samples = n_lsamples + n_usamples


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
        num_sampled = int(750/n_shot)
        ## base 和 novel 类的匹配
        # cost矩阵
        dist = (support_data.unsqueeze(1) - base_means.unsqueeze(0)).norm(dim=2).unsqueeze(0)

        # 行和约束
        r = torch.ones(1, n_ways*n_shot)
        # mean_base = torch.mean(base_means, 0)
        # r = (mean_base.unsqueeze(0) - support_data.unsqueeze(1)).norm(dim=2).view(1, -1)
        # r = n_ways * r / torch.sum(r, 1)

        # 列和约束
        mean_support = torch.mean(support_data, 0)
        c = (mean_support.unsqueeze(0) - base_means.unsqueeze(1)).norm(dim=2)
        c = n_ways * c/torch.sum(c, 0)
        c = c.view(1,-1)

        #weight 矩阵
        n_b_match, _ = compute_optimal_transport(dist, r, c, lam=0, epsilon=1e-6)
        n_b_match = n_b_match.view(n_ways, n_base_class).to(torch.float32)

        for j in range(n_lsamples):
            mean, cov = distribution_calibration(support_data[j], base_means, base_cov, n_b_match[j,])
            distribute = MultivariateNormal(mean, cov.to(torch.float32))
            samp_data = distribute.rsample(sample_shape=torch.Size([num_sampled]))
            sampled_data.append(samp_data)
            sampled_label.extend([support_label[j]] * num_sampled)
        sampled_data = torch.tensor([item.cpu().detach().numpy() for item in sampled_data]).cuda()
        sampled_data = sampled_data.reshape(n_ways * n_shot * num_sampled, -1)
        sampled_label = torch.tensor(sampled_label).cuda()
        X_aug = torch.cat([support_data, sampled_data], 0)
        Y_aug = torch.cat([support_label, sampled_label], 0)

        # ##标准化
        X_aug = normDatas(X_aug)
        query_data = normDatas(query_data)

        classifier = LR(max_iter=1000).fit(x=X_aug, y=Y_aug)
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts.cpu().numpy() == query_label.cpu().numpy())
        tqdm_gen.set_description('run {}, acc={:.4f}'.format(i, acc))
        acc_list.append(acc)
    print('%s %d way %d shot  ACC : %f' % (dataset, n_ways, n_shot, float(np.mean(acc_list))))

