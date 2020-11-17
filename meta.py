import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from copy import deepcopy
from models import EDSR
from loss_functions import perceptionLoss

## Adapted from: https://github.com/dragen1860/MAML-Pytorch/
## All credits for the code structure goes to dragen1860.

class Learner(nn.Module):
    def __init__(self, config, load_weights=''):
        """
        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()
        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        if load_weights != '':
            edsr = EDSR()
            edsr.load_state_dict(torch.load(load_weights))
            self.vars = nn.ParameterList([param for name, param in edsr.named_parameters() if not(name.startswith('sub') or name.startswith('add'))])
            del edsr
        else:
            for i, (name, param) in enumerate(self.config):
                if name is 'conv2d':
                    # [ch_out, ch_in, kernelsz, kernelsz]
                    w = nn.Parameter(torch.ones(*param[:4]))
                    # gain=1 according to cbfin's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    # [ch_out]
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))

                elif name is 'convt2d':
                    # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                    w = nn.Parameter(torch.ones(*param[:4]))
                    # gain=1 according to cbfin's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    # [ch_in, ch_out]
                    self.vars.append(nn.Parameter(torch.zeros(param[1])))

                elif name is 'linear':
                    # [ch_out, ch_in]
                    w = nn.Parameter(torch.ones(*param))
                    # gain=1 according to cbfinn's implementation
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    # [ch_out]
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))

                elif name is 'bn':
                    # [ch_out]
                    w = nn.Parameter(torch.ones(param[0]))
                    self.vars.append(w)
                    # [ch_out]
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))

                    # must set requires_grad=False
                    running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])

                elif (name is 'resblock_leakyrelu') or (name is 'resblock_relu') :
                    w1, w2 = nn.Parameter(torch.ones(*param[:4])), nn.Parameter(torch.ones(*param[:4]))
                    b1, b2 = nn.Parameter(torch.zeros(param[0])), nn.Parameter(torch.zeros(param[0]))
                    torch.nn.init.kaiming_normal_(w1)
                    torch.nn.init.kaiming_normal_(w2)
                    self.vars.append(w1)
                    self.vars.append(b1)
                    self.vars.append(w2)
                    self.vars.append(b2)

                elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                              'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'pixelshuffle', 'sub_mean', 'add_mean']:
                    continue
                else:
                    raise NotImplementedError


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            elif name is 'pixelshuffle':
                tmp = name + ':' + str((tuple(param)))
                info += tmp + '\n'
            elif (name is 'resblock_relu') or (name is 'resblock_leakyrelu'):
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'

            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchronized of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchronized of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            # ToDo: Prelu (tricky cause it has parameters).

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name is 'pixelshuffle':
                x = F.pixel_shuffle(x, param[0])
            elif name is 'resblock_leakyrelu':
                w1, b1, w2, b2 = vars[idx], vars[idx + 1], vars[idx + 2], vars[idx + 3]
                y = F.conv2d(x, w1, b1, stride=param[4], padding=param[5])
                y = F.leaky_relu(y, negative_slope=param[7], inplace=param[8])
                y = F.conv2d(y, w2, b2, stride=param[4], padding=param[5])
                y = y.mul(param[6])
                x = x.add(y)
                idx += 4
            elif name is 'resblock_relu':
                w1, b1, w2, b2 = vars[idx], vars[idx + 1], vars[idx + 2], vars[idx + 3]
                y = F.conv2d(x, w1, b1, stride=param[4], padding=param[5])
                y = F.relu(y, inplace=param[7])
                y = F.conv2d(y, w2, b2, stride=param[4], padding=param[5])
                y = y.mul(param[6])
                x = x.add(y)
                idx += 4
            elif name is 'sub_mean':
                x = F.conv2d(x, torch.eye(3).view(3,3,1,1).cuda() / torch.Tensor(param[1]).view(3, 1, 1, 1).cuda(), -1 * torch.Tensor(param[0]).cuda() / torch.Tensor(param[1]).cuda())
            elif name is 'add_mean':
                x = F.conv2d(x, torch.eye(3).view(3,3,1,1).cuda() / torch.Tensor(param[1]).view(3, 1, 1, 1).cuda(), torch.Tensor(param[0]).cuda() / torch.Tensor(param[1]).cuda())
            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x

    def zero_grad(self, vars=None):
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, config, update_lr, meta_lr, update_step, update_step_test, k_support=10, load_weights='', finetune_param_nb=0, loss_func='MSE'):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.k_spt = k_support
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.finetune_depth = finetune_param_nb
        self.net = Learner(config, load_weights)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        if loss_func == 'MSE':
            self.loss_func = F.mse_loss
        elif loss_func == 'perception':
            self.loss_func = perceptionLoss()
        else:
            raise NotImplementedError


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz, c_, h, w]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz, c_, h, w]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        # The number of tasks handled is basically the batch size. Default will be = 1.

        losses_q = [0 for _ in range(self.update_step + 1)]# losses_q[i] is the loss on step i
        print("num of images in task="+str(k_spt))
        for i in range(self.k_spt):

            # 1. run the i-th task and compute loss for k=0
            #print(grid)
            reconstructed = self.net(x_spt[i], vars=None, bn_training=True)
            loss = self.loss_func(reconstructed, y_spt[i]) # ToDo: Make the loss function customizable.
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                reconstructed_q = self.net(x_qry, self.net.parameters(), bn_training=True) # not updated weights
                loss_q = self.loss_func(reconstructed_q, y_qry)
                losses_q[0] += loss_q

            # this is the loss after the first update
            with torch.no_grad():
                # [setsz, nway]
                reconstructed_q = self.net(x_qry, fast_weights, bn_training=True) # updated weights
                loss_q = self.loss_func(reconstructed_q, y_qry)
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                reconstructed = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = self.loss_func(reconstructed, y_spt[i])
                #print(loss)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                # Keep track of the loss on the query set
                reconstructed_q = self.net(x_qry, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = self.loss_func(reconstructed_q, y_qry)
                losses_q[k + 1] += loss_q
                del reconstructed_q
                del reconstructed

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num
        # THIS IS THE SUM OF THE LOSSES OVER ALL k BECAUSE WE DID NOT RESET GRADS WITH zero_grad() at each new step SO BY DEFAULT PYTORCH SUM THEM UP (because it's useful for LSTM and other stuff).

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        del losses_q

        return loss_q.item()

    def finetuning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz, c_, h, w]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz, c_, h, w]
        :return:
        """
        assert len(x_spt.shape) == 4

        task_num = x_spt.size()[0]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we fine tuning on the copied model instead of self.net
        net = deepcopy(self.net)

        losses_q = [0 for _ in range(self.update_step + 1)]

        # 1. run the i-th task and compute loss for k=0
        reconstructed = net(x_spt)
        loss = self.loss_func(reconstructed, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            reconstructed_q = net(x_qry, net.parameters(), bn_training=True)
            loss_q = self.loss_func(reconstructed_q, y_qry)
            losses_q[0] += loss_q

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            reconstructed_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = self.loss_func(reconstructed_q, y_qry)
            losses_q[1] += loss_q

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            reconstructed = net(x_spt, fast_weights, bn_training=True)
            loss = self.loss_func(reconstructed, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            reconstructed_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = self.loss_func(reconstructed_q, y_qry)
            losses_q[k + 1] += loss_q

        del net

        loss_q = losses_q[-1] / task_num

        return loss_q.item()

    def test(self, x_spt, y_spt, x_qry):
        assert len(x_spt.shape) == 4

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we fine tuning on the copied model instead of self.net

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = deepcopy(self.net).to(device)

        # 1. run the i-th task and compute loss for k=0
        reconstructed = net(x_spt)
        reconstructed_without = net(x_qry).data
        loss = self.loss_func(reconstructed, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            reconstructed = net(x_spt, fast_weights, bn_training=True)
            loss = self.loss_func(reconstructed, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            if k == self.update_step_test - 1: # Only compute the query after the last updating step and return it.
                reconstructed_q = net(x_qry, fast_weights, bn_training=True)
        del net

        return reconstructed_q, reconstructed_without


