

import sys
import torch.utils.data
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from wideresnet import MultiHeadWideResNet
from data_class import MultitaskDataset
from loss import *
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import KernelDensity


from kmeans_strategy import init_centers, Kmeans_dist
from scipy.spatial.distance import cdist

import logging

import math 

cudnn.benchmark = True

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def hybrid1(scores,embs, K,lambda_=0.5):
    num_knn = int(np.ceil(scores.shape[0]/K))
    dist_matrix = torch.cdist(embs, embs, p=2)
    nearest_dist, _ = torch.topk(dist_matrix, num_knn, largest=False,
                              sorted=False)
    nearest_dist = nearest_dist.cpu().numpy()
    anchor_dist = np.max(nearest_dist,1,keepdims=True)
    score_pos, _ = torch.topk(scores, int(scores.shape[0] * 0.1), largest=True,
                              sorted=False)
    anchor_score = score_pos.min()
    boundary_score = torch.abs(scores - anchor_score).cpu().numpy()
    boundary_score = (boundary_score-boundary_score.min())/(boundary_score.max()-boundary_score.min())
    idx_ = np.argmin(boundary_score)
    idx_active = [idx_]
    embs = embs.cpu().numpy()
    for _ in range(K-1):
        score = 0.5+(anchor_dist>=cdist(embs,embs[idx_active],'euclidean')).sum(1)/(2*num_knn)
        score = score+boundary_score
        score[idx_active] = 1e3
        idx_ = np.argmin(score)
        idx_active.append(idx_)
    return torch.tensor(idx_active)


class TransClassifier():
    def __init__(self, total_num_trans, num_trans_list, args):
        self.n_trans = total_num_trans
        self.n_trans_list = num_trans_list
        self.n_heads = len(self.n_trans_list)

        self.ndf = 256

        self.args = args
        self.n_channels = 1
        if args.dataset == 'cifar10':
            self.n_channels = 3
        if args.dataset == 'blood':
            self.n_channels = 3

        self.netWRN = MultiHeadWideResNet(
                                 self.args.depth, num_trans_list, 
                                 n_channels=self.n_channels, 
                                 widen_factor=self.args.widen_factor,
                                 dropRate=0.3).cuda()
        self.optimizer = torch.optim.Adam(self.netWRN.parameters(),
                                          lr=self.args.lr)

        self.__oe_loss__ = self.args.oe_loss

        self.ori_batch_size = self.args.batch_size//self.n_trans
        self.m = self.args.m


    def set_additional_optimization_options(self):
        #////// Nesterov SGD optimizer //////
        if self.args.sgd_opt:
            self.optimizer = torch.optim.SGD(self.netWRN.parameters(),
                                             lr=self.args.lr,
                                             momentum=0.9,
                                             weight_decay=5e-4,
                                             nesterov=True)
        # ////////////////////////////////////

        if self.args.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: cosine_annealing(
                    step,
                    self.args.epochs * len(list(range(0, len(x_train), self.args.batch_size))),
                    1,  # since lr_lambda computes multiplicative factor
                    1e-6 / self.args.lr))

        if self.args.epoch_lr_decay:
            self.epoch_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10, gamma=0.1)


    def get_dataloaders(self, x_train, mt_train_labels, y_train, x_test, mt_test_labels, y_test):
        train_dataset = MultitaskDataset(x_train, mt_train_labels, y_train, self.n_trans)
        test_dataset = MultitaskDataset(x_test, mt_test_labels, y_test, self.n_trans, test=True)

        train_loader = DataLoader(train_dataset, batch_size=self.ori_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.ori_batch_size, shuffle=True)
        return train_loader, test_loader


    def set_loss_function(self):
        self.multihead_ce = MultiheadCrossEntropy(self.n_trans, ad_temp=self.args.ad_temp)
        self.multihead_ce_uniform = MultiheadCrossEntropyAgainstUniform(self.n_trans, ad_temp=self.args.ad_temp)
        self.multihead_ce_zerodriven = MultiheadCrossEntropyZeroDriven(self.n_trans, ad_temp=self.args.ad_temp)

    def set_anomaly_score_function(self):
        self.multihead_ce_score = MultiheadCrossEntropyAnomalyScore(self.n_trans, ad_temp=self.args.ad_temp)
        self.latent_gauss_score = LatentGaussianRepresentationAnomalyScore(self.n_trans, ad_temp=self.args.ad_temp)
        self.multihead_ce_uniform_score = MultiheadCrossEntropyAgainstUniformAnomalyScore(self.n_trans, ad_temp=self.args.ad_temp)

    def process_minibatch(self, mini_batch):
        mini_batch['id'] = mini_batch['id'].cuda()
        mini_batch['data'] = mini_batch['data'].view(-1, *mini_batch['data'].shape[2:]).float().cuda() # (10, 36, 3, 32, 32)
        mini_batch['gt_labels'] = mini_batch['gt_labels'].view(-1, *mini_batch['gt_labels'].shape[2:]).long().cuda() # 
        mini_batch['mt_labels'] = mini_batch['mt_labels'].view(-1, *mini_batch['mt_labels'].shape[2:]).long().cuda() # (10, 36, 3)
        mini_batch['p_labels'] = mini_batch['p_labels'].view(-1, *mini_batch['p_labels'].shape[2:]).long().cuda() # (10, 36, 3)
        mini_batch['weights'] = mini_batch['weights'].view(-1, *mini_batch['weights'].shape[2:]).long().cuda() # (10, 36, 3)

        mini_batch['ori_minibatch_size'] = len(mini_batch['id'])
        mini_batch['minibatch_size'] = mini_batch['data'].shape[0]
        return mini_batch


    def train_init(self, train_loader):
        self.netWRN.train()
        total_loss = 0.0
        update_num = 1
        
        all_zs = []
        
        for mini_batch in train_loader:
            mini_batch = self.process_minibatch(mini_batch)
            xs = mini_batch['data'] # 
            gt_labels = mini_batch['gt_labels'] # 
            mt_labels = mini_batch['mt_labels'] # 
            p_labels = mini_batch['p_labels']
            weights = mini_batch['weights']

            zs_tc, mt_zs_ce = self.netWRN(xs)

            all_zs.append(zs_tc)
            # all_zs[idx] = zs_tc
            zs = torch.reshape(zs_tc, (mini_batch['ori_minibatch_size'], self.n_trans, self.ndf))

            pos_loss = self.multihead_ce(mt_zs_ce, mt_labels)

            tc = tc_loss(zs, self.m)
            ce = pos_loss.mean()
            if self.args.reg:
                loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
            else:
                loss = ce + self.args.lmbda * tc

            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return torch.cat(all_zs, 0)

    def train_one_epoch_sup(self, labeled_loader):
        self.netWRN.train()
        total_loss = 0.0
        update_num = 1

        all_zs = []

        for mini_batch in labeled_loader:
            mini_batch = self.process_minibatch(mini_batch)
            xs = mini_batch['data'] # 
            gt_labels = mini_batch['gt_labels'] # 
            mt_labels = mini_batch['mt_labels'] # 
            p_labels = mini_batch['p_labels']
            weights = mini_batch['weights']

            zs_tc, mt_zs_ce = self.netWRN(xs)

            all_zs.append(zs_tc)
            # all_zs[idx] = zs_tc
            zs = torch.reshape(zs_tc, (mini_batch['ori_minibatch_size'], self.n_trans, self.ndf))

            pos_loss = self.multihead_ce(mt_zs_ce, mt_labels)
            
            if self.args.oe_method == 'zero_driven':
                neg_loss = self.multihead_ce_zerodriven(mt_zs_ce, mt_labels)
            elif self.args.oe_method == 'max_ent':
                neg_loss = self.multihead_ce_uniform(mt_zs_ce)
            else:
                raise NotImplementedError

            _loss = torch.cat([pos_loss[gt_labels==0], neg_loss[gt_labels==1]],0)

            tc = tc_loss(zs, self.m)
            ce = _loss.mean()

            loss = ce + self.args.lmbda * tc

            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            update_num += 1

        total_loss /= update_num
        print("Average training loss: ", total_loss)
        logging.info(f'Average training loss: {total_loss}')

        return torch.cat(all_zs, 0)


    def train_one_epoch(self, epoch, train_loader, labeled_loader, ratio=None):
        self.netWRN.train()
        total_loss = 0.0
        update_num = 1
        
        all_zs = []
        
        # if epoch == 0 or self.args.oe_loss != 'supervise':
        for mini_batch in train_loader:
            mini_batch = self.process_minibatch(mini_batch)
            xs = mini_batch['data'] # 
            gt_labels = mini_batch['gt_labels'] # 
            mt_labels = mini_batch['mt_labels'] # 
            p_labels = mini_batch['p_labels']
            weights = mini_batch['weights']

            zs_tc, mt_zs_ce = self.netWRN(xs)

            all_zs.append(zs_tc)
            zs = torch.reshape(zs_tc, (mini_batch['ori_minibatch_size'], self.n_trans, self.ndf))

            pos_loss = self.multihead_ce(mt_zs_ce, mt_labels)

            if self.args.oe and ratio is not None:

                # rank against anomaly scores

                # ranking: training loss anomaly score 
                if self.args.oe_rank == 'training_obj':
                    logp_sz = self.multihead_ce_score(mt_zs_ce, mt_labels)

                # ranking: latent gaussian anomaly score 
                elif self.args.oe_rank == 'latent_gauss':
                    logp_sz = self.latent_gauss_score(zs)

                else:
                    raise NotImplementedError

                neg_loss = self.multihead_ce_uniform_score(mt_zs_ce).detach()


                logp_sz -= neg_loss

                num_rej = int(math.ceil(logp_sz.shape[0] * ratio))

                loss_accept, idx_accept = torch.topk(logp_sz, logp_sz.shape[0] - num_rej, largest=False, sorted=False)
                loss_reject, idx_reject = torch.topk(logp_sz, num_rej, largest=True, sorted=False)

                idx_accept = (torch.arange(self.n_trans).to(xs).repeat(idx_accept.size()[0]) + idx_accept.repeat_interleave(self.n_trans)*self.n_trans).long()
                idx_reject = (torch.arange(self.n_trans).to(xs).repeat(idx_reject.size()[0]) + idx_reject.repeat_interleave(self.n_trans)*self.n_trans).long()
                

                if self.args.oe_method == 'zero_driven':
                    neg_loss = self.multihead_ce_zerodriven(mt_zs_ce, mt_labels)


                elif self.args.oe_method == 'max_ent':
                    neg_loss = self.multihead_ce_uniform(mt_zs_ce)
    

                else:
                    raise NotImplementedError

                if self.args.oe_loss == 'weighted':
                    _loss = torch.cat([pos_loss[idx_accept],(1-self.args.oe_weight)*pos_loss[idx_reject]+(self.args.oe_weight)*neg_loss[idx_reject]],0)
                elif self.args.oe_loss == 'radical':
                    _loss = torch.cat([pos_loss[idx_accept], neg_loss[idx_reject]], 0)
                elif self.args.oe_loss == 'refine':
                    _loss = pos_loss[idx_accept]
                elif self.args.oe_loss == 'known_gt':
                    _loss = torch.cat([pos_loss[gt_labels==0], neg_loss[gt_labels==1]],0)
                else:
                    raise NotImplementedError
            else:
                _loss = pos_loss

            tc = tc_loss(zs, self.m)
            ce = _loss.mean()
            if self.args.reg:
                loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
            else:
                loss = ce + self.args.lmbda * tc

            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            if labeled_loader is not None:
                mini_batch = next(iter(labeled_loader))
                mini_batch = self.process_minibatch(mini_batch)
                xs = mini_batch['data'] # 
                gt_labels = mini_batch['gt_labels'] # 
                mt_labels = mini_batch['mt_labels'] # 
                p_labels = mini_batch['p_labels']
                weights = mini_batch['weights']

                zs_tc, mt_zs_ce = self.netWRN(xs)

                # all_zs.append(zs_tc)
                # all_zs[idx] = zs_tc
                zs = torch.reshape(zs_tc, (mini_batch['ori_minibatch_size'], self.n_trans, self.ndf))

                pos_loss = self.multihead_ce(mt_zs_ce, mt_labels)
                
                if self.args.oe_method == 'zero_driven':
                    neg_loss = self.multihead_ce_zerodriven(mt_zs_ce, mt_labels)
                elif self.args.oe_method == 'max_ent':
                    neg_loss = self.multihead_ce_uniform(mt_zs_ce)
                else:
                    raise NotImplementedError

                _loss = torch.cat([pos_loss[gt_labels==0], neg_loss[gt_labels==1]],0)

                tc = tc_loss(zs, self.m)
                ce = _loss.mean()

                loss = ce + self.args.lmbda * tc

                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            if self.args.lr_decay:
                self.scheduler.step()

            update_num += 1


        if labeled_loader is not None:
            for mini_batch in labeled_loader:
                mini_batch = self.process_minibatch(mini_batch)
                xs = mini_batch['data'] # 
                gt_labels = mini_batch['gt_labels'] # 
                mt_labels = mini_batch['mt_labels'] # 
                p_labels = mini_batch['p_labels']
                weights = mini_batch['weights']

                zs_tc, mt_zs_ce = self.netWRN(xs)

                all_zs.append(zs_tc)

        total_loss /= update_num
        print("Average training loss: ", total_loss)
        logging.info(f'Average training loss: {total_loss}')

        return torch.cat(all_zs, 0)



    def evaluation(self, epoch, test_loader, means):
        self.netWRN.eval()

        # evaluation
        print("## Training Objective Anomaly Score")
        logging.info("## Training Objective Anomaly Score")
        # make the training objective equivalent to anomaly score
        val_loss = 0.0
        val_update_num = 0
        true_label = []
        pred = []
        with torch.no_grad():

            for mini_batch in test_loader:
                mini_batch = self.process_minibatch(mini_batch)
                xs = mini_batch['data'] # 
                gt_labels = mini_batch['gt_labels'] # 
                mt_labels = mini_batch['mt_labels'] # 


                # anomaly score
                zs_tc, mt_zs_ce = self.netWRN(xs)

                val_probs = self.multihead_ce_score(mt_zs_ce, mt_labels)


                true_label += list(gt_labels.cpu().data.numpy())
                pred += list(val_probs.cpu().data.numpy())


                # validation loss
                ce = self.multihead_ce(mt_zs_ce, mt_labels)


                zs = torch.reshape(zs_tc, (mini_batch['ori_minibatch_size'], self.n_trans, self.ndf))
                # zs = torch.reshape(zs_tc, (batch_range // n_rots, n_rots, ndf))
                tc = tc_loss(zs, self.m)

                if self.args.reg:
                    _val_loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
                else:
                    _val_loss = ce + self.args.lmbda * tc

                val_loss += _val_loss.mean().item()
                val_update_num += 1

            auc = roc_auc_score(true_label, pred)
            ap = average_precision_score(true_label, pred)
            print("Epoch:", epoch, ", AUC: ", auc, ", AP: ", ap)
            logging.info("Epoch:" + f'{epoch}' + 
                         ", AUC: " + f'{auc}' + 
                         ", AP: " + f'{ap}')

            val_loss /= val_update_num
            print("Average validation loss: ", val_loss)
            logging.info(f'Average validation loss: {val_loss}')


        print("## Latent Gaussian Anomaly Score")
        logging.info("## Latent Gaussian Anomaly Score")
        true_label = []
        pred = []
        with torch.no_grad():
            for mini_batch in test_loader:
                mini_batch = self.process_minibatch(mini_batch)
                xs = mini_batch['data'] # 
                gt_labels = mini_batch['gt_labels'] # 
                mt_labels = mini_batch['mt_labels'] # 


                zs, _ = self.netWRN(xs)
                zs = torch.reshape(zs, (mini_batch['ori_minibatch_size'], self.n_trans, self.ndf))

                logp_sz = self.latent_gauss_score(zs, means)

                true_label += list(gt_labels.cpu().data.numpy())
                pred += list(logp_sz.cpu().data.numpy())

            # val_probs_rots_latentGauss = val_probs_rots_latentGauss.sum(1)
            auc_latent_gauss = roc_auc_score(true_label, pred)
            ap = average_precision_score(true_label, pred)
            print("Epoch:", epoch, ", AUC: ", auc_latent_gauss, ", AP: ", ap)
            logging.info("Epoch:" + f'{epoch}' + 
                         ", AUC: " + f'{auc_latent_gauss}' + 
                         ", AP: " + f'{ap}')

        return auc, auc_latent_gauss


    def compute_ad_score(self, epoch, train_data, means):
        train_loader = DataLoader(train_data, batch_size=self.ori_batch_size, shuffle=False, drop_last=False)
        
        ad_scores = []
        # for idx in range(len(train_data)):
        with torch.no_grad():
            for mini_batch in train_loader:
                mini_batch = self.process_minibatch(mini_batch)
                xs = mini_batch['data'] # (360, 3, 32, 32)
                gt_labels = mini_batch['gt_labels'] # 
                mt_labels = mini_batch['mt_labels'] # (10, 36, 3)

                zs_tc, mt_zs_ce = self.netWRN(xs)


                # all_zs[idx] = zs_tc
                zs = torch.reshape(zs_tc, (mini_batch['ori_minibatch_size'], self.n_trans, self.ndf))

                # ranking: training loss anomaly score 
                if self.args.oe_rank == 'training_obj' or epoch == 0:
                    logp_sz = self.multihead_ce_score(mt_zs_ce, mt_labels)

                # ranking: latent gaussian anomaly score 
                elif self.args.oe_rank == 'latent_gauss':
                    logp_sz = self.latent_gauss_score(zs, means)

                else:
                    raise NotImplementedError

                ad_scores.append(logp_sz)

        ad_scores = torch.cat(ad_scores, 0)

        return ad_scores


    def get_all_zs(self, train_data):
        train_loader = DataLoader(train_data, batch_size=self.ori_batch_size, shuffle=False, drop_last=False)

        all_zs = []
        with torch.no_grad():
            for mini_batch in train_loader:
                mini_batch = self.process_minibatch(mini_batch)
                xs = mini_batch['data'] # (360, 3, 32, 32)
                gt_labels = mini_batch['gt_labels'] # 
                mt_labels = mini_batch['mt_labels'] # (10, 36, 3)

                zs_tc, mt_zs_ce = self.netWRN(xs)

                all_zs.append(zs_tc)

        all_zs = torch.cat(all_zs, 0)
        all_zs = torch.reshape(all_zs, (len(train_data), self.n_trans * self.ndf))

        return all_zs


    def estimate_ratio(self, y_score, y_true, active_idx):
        score_range = np.max(y_score) - np.min(y_score)
        bw_p = score_range / len(active_idx)
        bw_q = score_range / len(active_idx)
        kde_p = KernelDensity(kernel="gaussian", bandwidth=bw_p).fit(y_score[:, np.newaxis])
        kde_q = KernelDensity(kernel="gaussian", bandwidth=bw_q).fit(y_score[active_idx, np.newaxis])
        log_p = kde_p.score_samples(y_score[active_idx, np.newaxis])
        log_q = kde_q.score_samples(y_score[active_idx, np.newaxis])
        ratio = np.mean(np.exp(log_p - log_q) * y_true[active_idx])
        return ratio


    def get_active_learning_idx_uniform_score(self, ad_scores, K=20):
        ad_scores = ad_scores.cpu()
        min_s = torch.min(ad_scores)
        max_s = torch.max(ad_scores)
        active_sample_neighbor = min_s + torch.rand(K) * (max_s - min_s)
        active_idx = []
        for v in active_sample_neighbor:
            _, idx = torch.topk(torch.abs(ad_scores - v), 1, largest=False)
            active_idx.append(idx)
        active_idx = torch.cat(active_idx, 0)

        return active_idx


    def assign_pseudo_labels(self, ad_scores, train_data, alpha=0.1):
        #### hard constraint assignment ####
        # loss_accept, idx_accept = torch.topk(ad_scores, int(ad_scores.shape[0] * 0.9), largest=False, sorted=False)
        loss_reject, idx_reject = torch.topk(ad_scores, int(ad_scores.shape[0] * alpha), largest=True, sorted=False)

        pseudo_labels = torch.zeros(ad_scores.size()[0])
        pseudo_labels[idx_reject] = 1.0
        pseudo_labels = torch.repeat_interleave(pseudo_labels, self.n_trans)

        train_data.p_labels = pseudo_labels

        print("Finish assigning pseudo labels.")

        return train_data


    def assign_pseudo_labels_active_true_label(self, ad_scores, train_data, active_rand_idx, ratio=0.1, q_weight=100):
        # print('active_rand_idx:', active_rand_idx)
        num_pos = int(ad_scores.shape[0] * ratio)

        # unfilled set with values -1
        pseudo_labels = -1*torch.ones(ad_scores.size()[0])

        # put known values in
        gt_labels = train_data.y[::self.n_trans]
        pseudo_labels[active_rand_idx] = torch.from_numpy(gt_labels[active_rand_idx]).float()

        # assign other pseudo labels
        num_known_pos = torch.sum(pseudo_labels == 1)
        num_unknown_pos = num_pos - num_known_pos
        
        updated_ratio = num_unknown_pos / (ad_scores.shape[0] - active_rand_idx.shape[0])

        # assemble labeled dataset
        labeled_aug_idx = []
        for idx in active_rand_idx.numpy():
            labeled_aug_idx.append(np.arange(self.n_trans) + idx*self.n_trans)
        labeled_aug_idx = np.concatenate(labeled_aug_idx).astype(np.long)
        x = train_data.x_aug[labeled_aug_idx]
        y = train_data.y[labeled_aug_idx]
        mt_y = train_data.mt_labels[labeled_aug_idx]
        labeled_data = MultitaskDataset(x, mt_y, y, self.n_trans)

        # assemble unlabeled dataset
        unlabled_aug_idx = []
        for idx in range(ad_scores.shape[0]):
            if idx not in active_rand_idx:
                unlabled_aug_idx.append(np.arange(self.n_trans) + idx*self.n_trans)
        unlabled_aug_idx = np.concatenate(unlabled_aug_idx).astype(np.long)
        x = train_data.x_aug[unlabled_aug_idx]
        y = train_data.y[unlabled_aug_idx]
        mt_y = train_data.mt_labels[unlabled_aug_idx]
        unlabeled_data = MultitaskDataset(x, mt_y, y, self.n_trans)

        return unlabeled_data, labeled_data, updated_ratio




    def min_max_normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())


    def get_active_learning_idx_hybrid(self, ad_scores, embs, K=40, alpha=1.0):
        idx_active = []

        # ad_scores = self.compute_ad_scores(train_data)
        ad_scores = self.min_max_normalize(ad_scores)
        most_pos_idx = torch.argmax(ad_scores).item()
        idx_active.append(most_pos_idx)
        ad_scores[most_pos_idx] = 0.

        # dist_matrix = self.get_pairwise_dist(train_data)
        dist_matrix = torch.cdist(embs, embs, p=2)
        dist_matrix = self.min_max_normalize(dist_matrix)
        K -= 1
        for _ in range(K):
            dist_to_active = dist_matrix[idx_active].min(0)[0]
            # print(dist_to_active)
            score = ad_scores + alpha*dist_to_active
            idx = torch.argmax(score).item()
            idx_active.append(idx)
            ad_scores[idx] = 0.
        
        return torch.tensor(idx_active)



    def fit_trans_classifier(self, x_train, mt_train_labels, y_train, x_test, mt_test_labels, y_test):

        train_loader, test_loader = self.get_dataloaders(x_train, mt_train_labels, y_train, x_test, mt_test_labels, y_test)

        self.set_additional_optimization_options()

        self.set_loss_function()
        self.set_anomaly_score_function()

        print("Training")
        self.netWRN.train()
        bs = self.args.batch_size
        N, sh, sw, nc = x_train.shape
        n_rots = self.n_trans
        self.m = self.args.m
        ndf = 256

        aucs = np.zeros(self.args.epochs)
        aucs_latent_gauss = np.zeros(self.args.epochs)

        active_rand_idx = None # for active learning
        K = self.args.K # for active learning
        q_weight = N//n_rots / K - 1 # 6666 / K + 1 # 100 # 6600 / K # 100 # queried sample weight
        ratio = None
        labeled_loader = None

        train_data = train_loader.dataset
        unlabeled_loader = train_loader

        all_zs = self.train_init(train_loader)
        
        all_zs = torch.reshape(all_zs, (all_zs.shape[0]//n_rots, n_rots, ndf))
        means = all_zs.mean(0, keepdim=True)

        # active learning
        ad_scores = self.compute_ad_score(0, train_data, means)

        #### assign true labels with K-means++ ####
        if self.args.query_strategy == 'kdist':
            all_zs = self.get_all_zs(train_data).cpu().numpy()
            active_rand_idx = torch.Tensor(init_centers(all_zs, K=K)).long()
            # all_zs = self.get_all_zs(train_data).cpu()
            # active_rand_idx = torch.Tensor(Kmeans_dist(all_zs, K=K, tau=0.01)).long()
            print("Assign K=%d true labels with K-means++ initialization." % K)
        #### end ####

        #### assign true labels with K-means++ ####
        if self.args.query_strategy == 'hybr2':
            all_zs = self.get_all_zs(train_data).cpu()
            _ad_scores = ad_scores.cpu()
            active_rand_idx = self.get_active_learning_idx_hybrid(_ad_scores, all_zs, K=K).long()
            print("Assign K=%d true labels with Hybrid2." % K)
        #### end ####

        if self.args.query_strategy == 'hybr1':
            all_zs = self.get_all_zs(train_data).cpu()
            _ad_scores = ad_scores.cpu()
            active_rand_idx = hybrid1(_ad_scores, all_zs, K).long()
            print("Assign K=%d true labels with Hybrid1." % K)
    
        print("Query idx:", active_rand_idx)
        gt_ori_labels = train_data.y[::self.n_trans]

        p_ori_labels = train_data.p_labels[::self.n_trans]
        print("Pseudo label:", p_ori_labels[active_rand_idx])
        print("True label:", gt_ori_labels[active_rand_idx])
        print("Incorrect ratio:", torch.sum(torch.Tensor(p_ori_labels[active_rand_idx]) != torch.Tensor(gt_ori_labels[active_rand_idx])) / len(active_rand_idx))
        print("MLE Contamination ratio:", np.sum(gt_ori_labels[active_rand_idx]) / len(gt_ori_labels[active_rand_idx]))

        if ratio is None:
            if self.args.use_true_ratio:
                ratio = 0.1
                print("Use true ratio:", ratio)
            else:
                ratio = self.estimate_ratio(ad_scores.cpu().numpy(), gt_ori_labels, active_rand_idx.numpy())
                print("Use estimated ratio:", ratio)
        unlabeled_data, labeled_data, ratio = self.assign_pseudo_labels_active_true_label(ad_scores, train_data, active_rand_idx, ratio=ratio, q_weight=q_weight)
            
        batch_size = train_loader.batch_size
        shuffle = True
        unlabeled_loader = DataLoader(unlabeled_data, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        labeled_loader = DataLoader(labeled_data, batch_size=K, shuffle=shuffle, drop_last=False)

        for epoch in range(self.args.epochs):

            if self.args.oe_loss == 'supervise':
                all_zs = self.train_one_epoch_sup(labeled_loader)
            else:
                all_zs = self.train_one_epoch(epoch, unlabeled_loader, labeled_loader, ratio=ratio)


            all_zs = torch.reshape(all_zs, (all_zs.shape[0]//n_rots, n_rots, ndf))
            means = all_zs.mean(0, keepdim=True)

            auc, auc_latent_gauss = self.evaluation(epoch, test_loader, means)

            aucs[epoch] = auc
            aucs_latent_gauss[epoch] = auc_latent_gauss

            if self.args.epoch_lr_decay:
                self.epoch_scheduler.step()

            sys.stdout.flush()

        # save training logs
        np.save(self.args._foldername + './aucs.npy', aucs)
        np.save(self.args._foldername + './aucs_latent_gauss.npy', aucs_latent_gauss)

        print(aucs[-5:])
        print(aucs_latent_gauss[-5:])

        return aucs[-1], aucs_latent_gauss[-1]

