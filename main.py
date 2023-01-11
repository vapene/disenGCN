#!/usr/bin/env python3

import argparse
import os
import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class DataReader:
    def __init__(self, data_name, data_dir):
        # Reading the data...
        tmp = []
        prefix = os.path.join(data_dir, 'ind.%s.' % data_name)
        for suffix in ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']:
            with open(prefix + suffix, 'rb') as fin:
                tmp.append(pickle.load(fin, encoding='latin1'))
        x, y, tx, ty, allx, ally, graph = tmp
        with open(prefix + 'test.index') as fin:
            tst_idx = [int(i) for i in fin.read().split()]
        assert np.sum(x != allx[:x.shape[0], :]) == 0
        assert np.sum(y != ally[:y.shape[0], :]) == 0

        # Spliting the data...
        trn_idx = np.array(range(x.shape[0]), dtype=np.int64)
        val_idx = np.array(range(x.shape[0], allx.shape[0]), dtype=np.int64)
        tst_idx = np.array(tst_idx, dtype=np.int64)
        assert len(trn_idx) == x.shape[0]
        assert len(trn_idx) + len(val_idx) == allx.shape[0]
        assert len(tst_idx) > 0
        assert len(set(trn_idx).intersection(val_idx)) == 0
        assert len(set(trn_idx).intersection(tst_idx)) == 0
        assert len(set(val_idx).intersection(tst_idx)) == 0

        # Building the graph...
        graph = nx.from_dict_of_lists(graph)
        assert min(graph.nodes()) == 0
        n = graph.number_of_nodes()
        assert max(graph.nodes()) + 1 == n
        n = max(n, np.max(tst_idx) + 1)
        for u in range(n):
            graph.add_node(u)
        assert graph.number_of_nodes() == n
        assert not graph.is_directed()

        # Building the feature matrix and the label matrix...
        d, c = x.shape[1], y.shape[1]
        feat_ridx, feat_cidx, feat_data = [], [], []
        allx_coo = allx.tocoo()
        for i, j, v in zip(allx_coo.row, allx_coo.col, allx_coo.data):
            feat_ridx.append(i)
            feat_cidx.append(j)
            feat_data.append(v)
        tx_coo = tx.tocoo()
        for i, j, v in zip(tx_coo.row, tx_coo.col, tx_coo.data):
            feat_ridx.append(tst_idx[i])
            feat_cidx.append(j)
            feat_data.append(v)
        if data_name.startswith('nell.0'):
            isolated = np.sort(np.setdiff1d(range(allx.shape[0], n), tst_idx))
            for i, r in enumerate(isolated):
                feat_ridx.append(r)
                feat_cidx.append(d + i)
                feat_data.append(1)
            d += len(isolated)
        feat = spsprs.csr_matrix((feat_data, (feat_ridx, feat_cidx)), (n, d))
        targ = np.zeros((n, c), dtype=np.int64)
        targ[trn_idx, :] = y
        targ[val_idx, :] = ally[val_idx, :]
        targ[tst_idx, :] = ty
        targ = dict((i, j) for i, j in zip(*np.where(targ)))
        targ = np.array([targ.get(i, -1) for i in range(n)], dtype=np.int64)
        print('#instance x #feature ~ #class = %d x %d ~ %d' % (n, d, c))

        # Storing the data...
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.graph, self.feat, self.targ = graph, feat, targ

    def get_split(self):
        # *val_idx* contains unlabeled samples for semi-supervised training.
        return self.trn_idx, self.val_idx, self.tst_idx

    def get_graph_feat_targ(self):
        return self.graph, self.feat, self.targ


# noinspection PyUnresolvedReferences
def thsprs_from_spsprs(x):
    x = x.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack((x.row, x.col)).astype(np.int32)).long()
    val = torch.from_numpy(x.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(x.shape))


# noinspection PyUnresolvedReferences
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias


# noinspection PyUnresolvedReferences
class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False): # nb_size: 20
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        # if include_self=false
        nb_all = torch.zeros(n, nb_size, dtype=torch.int64) # 2708. 20
        nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size: # 20개가 안될 경우
                nb_v.extend([-1] * (nb_size - len(nb_v))) # 이후에는 -1로 채운다
                nb[v] = torch.LongTensor(nb_v) # neighbor
            else:
                popkids.append(v) # 20개보다 많으면 제외
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids
        # Graph with 2708 nodes and 5278 edges
        # nb tensor([[ 633, 1862, 2582,  ...,   -1,   -1,   -1],
        #         [   2,  652,  654,  ...,   -1,   -1,   -1],
        #         [   1,  332, 1454,  ...,   -1,   -1,   -1],
        #         ...,
        #         [ 287,   -1,   -1,  ...,   -1,   -1,   -1],
        #         [ 165,  169, 1473,  ...,   -1,   -1,   -1],
        #         [ 165,  598, 1473,  ...,   -1,   -1,   -1]]) torch.Size([2708, 20])
        #  pop 24 [88, 95, 109, 306, 415, 598, 733, 1013, 1042, 1072, 1131, 1169, 1224, 1358, 1441, 1483, 1542, 1623, 1701, 1810, 1914, 1986, 2034, 2045]

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all
        nb_size = nb.size(1) # 20
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        print('192',pk_nb.shape)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all


# noinspection PyUnresolvedReferences
class RoutingLayer(nn.Module):
    def __init__(self, dim, num_caps): #  # 7*16, 7
        super(RoutingLayer, self).__init__()
        assert dim % num_caps == 0
        self.d, self.k = dim, num_caps # 112, 7
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k) # shape[1,7]

    def forward(self, x, neighbors, max_iter): # feat:x, nbsize: 20, self.routit: 7
        dev = x.device # 2708,112
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        n, m = x.size(0), neighbors.size(0) // x.size(0)  # 2708, 20
        d, k, delta_d = self.d, self.k, self.d // self.k # d 112 k 7 delta_D 16
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d) # torch.Size([2708, 112])
        # x.view(n,k,delta_d) torch.Size([2708, 7, 16]) # normalize by 16
        z = torch.cat([x, self._cache_zero_d], dim=0) # 한 줄 붙이기 torch.Size([2709, 112])
        z = z[neighbors].view(n, m, k, delta_d) # neigh tensor([ 633, 1862, 2582,  ...,   -1,   -1,   -1], device='cuda:0') torch.Size([54160])
        # z[neighbors].shape torch.Size([54160, 112]), z[neighbors].view(n, m, k, delta_d).shape) torch.Size([2708, 20, 7, 16])
        u = None
        for clus_iter in range(max_iter): # routit 7
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
                # expand torch.Size([54160, 7]) view torch.Size([2708, 20, 7])
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2) # 가장 높은 채널
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)  # torch.Size([2708, 7, 16])
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d) # 2708, 112


class CapsuleNet(nn.Module):  # CapsuleNet = DisenGCN
    def __init__(self, nfeat, nclass, hyperpm):
        super(CapsuleNet, self).__init__()
        ncaps, rep_dim = hyperpm.ncaps, hyperpm.nhidden * hyperpm.ncaps  # 7, 'number of channels per layer.'
                                                                        # 16, 'Number of hidden units per capsule(channel).'
        self.pca = SparseInputLinear(nfeat, rep_dim) # 7, 7*16
        conv_ls = []
        for i in range(hyperpm.nlayer): # 5
            conv = RoutingLayer(rep_dim, ncaps) # 7*16, 7
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls
        self.mlp = nn.Linear(rep_dim, nclass) # 112, nclass
        self.dropout = hyperpm.dropout # 0
        self.routit = hyperpm.routit # 6 Number of iterations when routing

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb): # feat, NeibSampler(graph, hyperpm.nbsz: 20)
        nb = nb.view(-1) #  torch.Size([54160]) tensor([ 633, 1862, 2582,  ...,   -1,   -1,   -1], device='cuda:0')
        # 2708 * 20 = 54160
        x = fn.relu(self.pca(x))  # self.pca = Wx+b  Zik
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        x = self.mlp(x)
        return fn.log_softmax(x, dim=1)

class EvalHelper:
    # noinspection PyUnresolvedReferences
    def __init__(self, dataset, hyperpm):
        use_cuda = torch.cuda.is_available() and not hyperpm.cpu
        dev = torch.device('cuda' if use_cuda else 'cpu')
        graph, feat, targ = dataset.get_graph_feat_targ()
        targ = torch.from_numpy(targ).to(dev)
        feat = thsprs_from_spsprs(feat).to(dev)
        trn_idx, val_idx, tst_idx = dataset.get_split()
        trn_idx = torch.from_numpy(trn_idx).to(dev)
        val_idx = torch.from_numpy(val_idx).to(dev)
        tst_idx = torch.from_numpy(tst_idx).to(dev)
        nfeat, nclass = feat.size(1), int(targ.max() + 1)
        model = CapsuleNet(nfeat, nclass, hyperpm).to(dev)
        optmz = optim.Adam(model.parameters(),
                           lr=hyperpm.lr, weight_decay=hyperpm.reg)
        self.graph, self.feat, self.targ = graph, feat, targ
        self.trn_idx, self.val_idx, self.tst_idx = trn_idx, val_idx, tst_idx
        self.model, self.optmz = model, optmz
        self.neib_sampler = NeibSampler(graph, hyperpm.nbsz).to(dev) # nbzs: 20 size of sampled neigh

    def run_epoch(self, end='\n'):
        self.model.train()
        self.optmz.zero_grad()
        prob = self.model(self.feat, self.neib_sampler.sample())
        loss = fn.nll_loss(prob[self.trn_idx], self.targ[self.trn_idx])
        loss.backward()
        self.optmz.step()
        print('trn-loss: %.4f' % loss.item(), end=end)
        return loss.item()

    def print_trn_acc(self):
        print('trn-', end='')
        trn_acc = self._print_acc(self.trn_idx, end=' val-')
        val_acc = self._print_acc(self.val_idx)
        return trn_acc, val_acc

    def print_tst_acc(self):
        print('tst-', end='')
        tst_acc = self._print_acc(self.tst_idx)
        return tst_acc

    def _print_acc(self, eval_idx, end='\n'):
        self.model.eval()
        prob = self.model(self.feat, self.neib_sampler.nb_all)[eval_idx]
        targ = self.targ[eval_idx]
        pred = prob.max(1)[1].type_as(targ)
        acc = pred.eq(targ).double().sum() / len(targ)
        acc = acc.item()
        print('acc: %.4f' % acc, end=end)
        return acc



# noinspection PyUnresolvedReferences
def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# noinspection PyUnresolvedReferences
def train_and_eval(datadir, datname, hyperpm):
    set_rng_seed(23)
    agent = EvalHelper(DataReader(datname, datadir), hyperpm) # disGCN & neighbor sampler 정의
    tm = time.time()
    best_val_acc, wait_cnt = 0.0, 0
    model_sav = tempfile.TemporaryFile()
    neib_sav = torch.zeros_like(agent.neib_sampler.nb_all, device='cpu') # 2708, 20
    for t in range(hyperpm.nepoch): # 200
        print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
        agent.run_epoch(end=' ') # run disGCN
        _, cur_val_acc = agent.print_trn_acc() # print rain, val acc
        if cur_val_acc > best_val_acc:
            wait_cnt = 0
            best_val_acc = cur_val_acc
            model_sav.close()
            model_sav = tempfile.TemporaryFile()
            torch.save(agent.model.state_dict(), model_sav)
            neib_sav.copy_(agent.neib_sampler.nb_all)
        else:
            wait_cnt += 1
            if wait_cnt > hyperpm.early:
                break
    print("time: %.4f sec." % (time.time() - tm))
    model_sav.seek(0)
    agent.model.load_state_dict(torch.load(model_sav))
    agent.neib_sampler.nb_all.copy_(neib_sav)
    return best_val_acc, agent.print_tst_acc()


def main(args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='cora')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=200,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=10,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=5,
                        help='Number of conv layers.')
    parser.add_argument('--ncaps', type=int, default=7,
                        help='Maximum number of capsules per layer.')
    parser.add_argument('--nhidden', type=int, default=16,
                        help='Number of hidden units per capsule.')
    parser.add_argument('--routit', type=int, default=6,
                        help='Number of iterations when routing.')
    parser.add_argument('--nbsz', type=int, default=20,
                        help='Size of the sampled neighborhood.')
    args = parser.parse_args()

    with RedirectStdStreams(stdout=sys.stderr):
        val_acc, tst_acc = train_and_eval(args.datadir, args.datname, args)
        print('val=%.2f%% tst=%.2f%%' % (val_acc * 100, tst_acc * 100))
    return val_acc, tst_acc


if __name__ == '__main__':
    print('(%.4f, %.4f)' % main())

