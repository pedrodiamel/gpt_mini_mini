# STD MODULES
import math
import os
import shutil
import time

import numpy as np

# TORCH MODULE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.text import Perplexity
from tqdm import tqdm

# LOCAL MODULE
from . import logger, losses, models
from .neuralnet import NeuralNetAbstractNLP


class NeuralNetTransformer(NeuralNetAbstractNLP):
    r"""Transformer Neural Network"""

    def __init__(
        self, pathproject, nameproject, no_cuda=True, parallel=False, seed=1, print_freq=10, gpu=0, compiler=False
    ):
        super(NeuralNetTransformer, self).__init__(
            pathproject, nameproject, no_cuda, parallel, seed, print_freq, gpu, compiler
        )

    def create(
        self,
        arch,
        vocab_size,
        block_size,
        n_layers,
        n_embd,
        n_heads,
        loss,
        lr,
        optimizer,
        lrsch,
        grad_clip=1.0,
        pretrained=False,
    ):
        cfg_opt = {"momentum": 0.9, "weight_decay": 5e-4}
        cfg_scheduler = {"step_size": 45, "gamma": 0.1}

        super(NeuralNetTransformer, self).create(
            arch,
            vocab_size,
            block_size,
            n_layers,
            n_embd,
            n_heads,
            loss,
            lr,
            optimizer,
            lrsch,
            grad_clip,
            pretrained,
            cfg_opt=cfg_opt,
            cfg_scheduler=cfg_scheduler,
        )

        if os.path.exists(self.visdom_log_pathname):
            self.plotter.viz.replay_log(self.visdom_log_pathname)

        # self.metrics = Perplexity(device=self.device)

        # Set the graphic visualization
        self.logger_train = logger.Logger("Trn", ["loss"], ["perp"], self.plotter)
        self.logger_val = logger.Logger("Val", ["loss"], ["perp"], self.plotter)

    def training(self, data_loader, epoch=0):
        self.logger_train.reset()
        data_time = logger.AverageMeter()
        batch_time = logger.AverageMeter()

        # switch to evaluate mode
        self.net.train()

        end = time.time()
        for i, (x, y) in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = x.size(0)

            if self.cuda:
                x, y = x.cuda(), y.cuda()

            # fit (forward)
            yh = self.net(x)

            # measure accuracy and record loss
            loss = self.criterion(yh.view(-1, self.vocab_size), y.view(-1))
            # self.metrics.update(yh, y)  # [yh]_(B,T,V), [y]_(B,T)
            # perp = self.metrics.compute().cpu()  # [perp]_(1)
            perp = 0.0

            # clip the gradient
            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

            # optimizer
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # update
            self.logger_train.update(
                {"loss": loss.item()},
                {"perp": perp},
                batch_size,
            )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                self.logger_train.logger(
                    epoch,
                    epoch + float(i + 1) / len(data_loader),
                    i,
                    len(data_loader),
                    batch_time,
                )
        # self.metrics.reset()

    def evaluate(self, data_loader, epoch=0):
        self.logger_val.reset()
        batch_time = logger.AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        with torch.no_grad():
            end = time.time()
            for i, (x, y) in enumerate(data_loader):
                batch_size = x.size(0)

                if self.cuda:
                    x, y = x.cuda(), y.cuda()

                # fit (forward)
                yh = self.net(x)

                # measure accuracy and record loss
                # loss = self.criterion(yh.view(-1, self.vocab_size), y.view(-1))
                loss = self.criterion(yh.view(-1, yh.size(-1)), y.view(-1))

                # self.metrics.update(yh, y)  # [yh]_(B,T,V), [y]_(B,T)
                # perp = self.metrics.compute().cpu()  # [perp]_(1)
                perp = 0.0

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # update
                self.logger_val.update(
                    {"loss": loss.item()},
                    {"perp": perp},
                    batch_size,
                )

                if i % self.print_freq == 0:
                    self.logger_val.logger(
                        epoch,
                        epoch,
                        i,
                        len(data_loader),
                        batch_time,
                        bplotter=False,
                        bavg=True,
                        bsummary=False,
                    )

        # save validation loss
        self.vallosses = self.logger_val.info["loss"]["loss"].avg
        perp = self.logger_val.info["metrics"]["perp"].avg
        # self.metrics.reset()

        self.logger_val.logger(
            epoch,
            epoch,
            i,
            len(data_loader),
            batch_time,
            bplotter=True,
            bavg=True,
            bsummary=True,
        )

        return perp

    def top_k_filtering(self, logits, top_k=0):
        """Filter a distribution of logits using top-k filtering"""

        out = logits.clone()
        if top_k > 0:
            # Remove all tokens with a probability less than the
            # last token of the top-k
            # indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            # logits[indices_to_remove] = float("-inf")
            v, _ = torch.topk(logits, top_k)
            out[out < v[:, [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def generate(self, idx, steps, temperature=1.0, sample=False, top_k=None):
        self.net.eval()
        # idx is (B, T) array of indices in the current context
        for _ in range(steps):
            # crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]

            # get the predictions
            logits = self.net(idx_cond)

            # pluck the logits at the final step and scale by temperature
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature  # becomes (B, C)

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                # apply top-k filtering
                logits = self.top_k_filtering(logits, top_k=top_k)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution or take the most likely
            if sample:
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def __call__(self, x, steps, temperature=1.0, sample=False, top_k=None):
        x = x.cuda() if self.cuda else x
        y = self.generate(x, steps, temperature=temperature, sample=sample, top_k=top_k)
        return y.cpu().numpy()

    def _create_model(self, arch, vocab_size, block_size, n_layers, n_embd, n_heads, pretrained):
        self.net = None

        kw = {
            "vocab_size": vocab_size,
            "block_size": block_size,
            "n_layers": n_layers,
            "n_embd": n_embd,
            "n_heads": n_heads,
            "pretrained": pretrained,
        }
        self.net = models.__dict__[arch](**kw)

        self.s_arch = arch
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.pretrained = pretrained

        if self.compiler:
            self.unoptimazed_net = self.net
            self.net = torch.compile(self.net)

        if self.cuda:
            self.net.cuda()
        if self.parallel and self.cuda:
            self.net = nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def _create_loss(self, loss):
        # create loss
        if loss == "cross":
            self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise NotImplementedError("Loss {} not implemented".format(loss))

        if self.cuda:
            self.criterion.cuda()

        self.s_loss = loss
