from collections import defaultdict
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)
                           

    # init delta (adv. perturbation) - FILL ME
    delta = None

    # total number of updates - FILL ME
    real_epochs = epochs / m
    # total_updates = real_epochs * (len(data_tr.dataset) / batch_size)
    total_updates = int(np.ceil(epochs * len(data_tr)/batch_size))



    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    iter_count = 0
    while iter_count < total_updates:
        for X, Y in loader_tr:
            if delta is None and iter_count == 0:
                delta = torch.zeros_like(X).to(device)
            elif delta is None:
                raise ValueError("delta is None but iter_count > 0")
            X = X.to(device)
            Y = Y.to(device)
            for j in range(m):
                # if torch.is_tensor(delta) and X.shape[0] != delta.shape[0]:
                #     delta = delta[:X.shape[0]]
                noised_x = X + delta[:X.shape[0]]
                noised_x.requires_grad = True
                outputs = model(noised_x)
                # outputs to labels?
                loss = criterion(outputs, Y)
                loss.backward()

                # might need to take this from X + delta!
                g_adv = noised_x.grad

                optimizer.step()

                delta[ : X.shape[0]] = delta[ : X.shape[0]] + eps * torch.sign(g_adv)
                delta = torch.clamp(delta, -eps, eps)

                optimizer.zero_grad()

                iter_count += 1
                if iter_count % scheduler_step_iters == 0:
                    lr_scheduler.step()
                if iter_count == total_updates:
                    break
                noised_x.grad.zero_()
            if iter_count == total_updates:
                break

    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma


    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        # with torch.no_grad():
        #     counts = np.zeros(4, dtype=int)
        #     for _ in range(ceil(n / batch_size)):
        #         this_batch_size = min(batch_size, n)
        #         n -= this_batch_size
        #
        #         batch = x.cuda().repeat((this_batch_size, 1, 1, 1))
        #         noise = torch.randn_like(batch, device='cuda') * self.sigma
        #         predictions = self.model(batch + noise).argmax(1)
        #         counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
        #     return counts
        # FILL ME
        # USE SIGMA
        x = x.cuda()
        class_dict = defaultdict(lambda: 0)
        num_iters = round(n // batch_size)
        for i in range(num_iters):
            x_batch = x.expand(batch_size, *x.shape[1:])
            # cov_matrix = (self.sigma * self.sigma) * torch.eye()
            noise = torch.normal(0, (self.sigma), x_batch.shape).cuda()
            # noise = torch.randn_like(x_batch, device='cuda') * self.sigma

            # noise = torch.randn(x_batch.shape)
            pred = self.model(x_batch + noise)
            class_ids = pred.argmax(1)
            for class_id in class_ids:
                class_dict[class_id.item()] += 1

        #last iter
        last_batch_size = n % batch_size
        if last_batch_size > 0:
            x_batch = x.expand(last_batch_size, *x.shape[1:])
            noise = torch.normal(0, self.sigma * self.sigma, x_batch.shape).cuda()
            # noise = torch.randn(x_batch.shape)
            pred = self.model(x_batch + noise)
            class_ids = pred.argmax(1)
            for class_id in class_ids:
                class_dict[class_id.item()] += 1


        num_class = 4 # hardcoded
        class_list = [0 for _ in range(num_class)]
        for key in class_dict:
            class_list[key] = class_dict[key]

        return class_list
        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c) - FILL ME
        class_counts = self._sample_under_noise(x, n0, batch_size)
        c_a = torch.tensor(class_counts).argmax()
        counts = self._sample_under_noise(x, n, batch_size)
        p_a = proportion_confint(counts[c_a], n , 2 * alpha, method="beta")[0]
        if p_a <= 0.5:
            return "ABSTAIN", 0

        c = c_a
        radius = self.sigma * norm.ppf(p_a)


        # compute lower bound on p_c - FILL ME
        

        # done
        return c, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        mask = torch.rand(self.dim).to(device).requires_grad_()
        trigger = torch.rand(self.dim).to(device).requires_grad_()

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        optimizer = optim.SGD([mask,trigger], lr=self.step_size)
        count_niters = 0
        while count_niters < self.niters:
            for x,y in data_loader:
                tensor_ct = torch.tensor(c_t).expand(x.shape[0]).to(device)
                optimizer.zero_grad()
                x,y = x.to(device),y.to(device)
                masked_x = (1-mask) * x + mask * trigger
                output = self.model(masked_x)
                loss = self.loss_func(output, tensor_ct) + self.lambda_c * mask.abs().sum()
                loss.backward()
                optimizer.step()
                count_niters += 1
                if(count_niters == self.niters):
                    break

        # done
        return mask, trigger
