# STD MODULES
import os

# TORCH MODULES
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import graphic as gph, netlearningrate, utils

# ----------------------------------------------------------------------------------------------
# Neural Net Abstract Class


class NeuralNetAbstractNLP(object):
    """
    Abstract NLP Neural Net
    """

    def __init__(
        self,
        pathproject,
        nameproject,
        no_cuda=True,
        parallel=False,
        seed=1,
        print_freq=10,
        gpu=0,
        compiler=False,
    ):
        """NeuralNetAbstractNLP Constructor

        Args:
            pathproject (str): path project directory
            nameproject (str): name of the project
            no_cuda (bool, optional): cuda used. Defaults to True.
            parallel (bool, optional): parallel option. Defaults to False.
            seed (int, optional): seed set. Defaults to 1.
            print_freq (int, optional): loss and metrics print frequency. Defaults to 10.
            gpu (int, optional): gpu index. Defaults to 0.
        """

        # cuda
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.parallel = not no_cuda and parallel
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.set_device(gpu)
            torch.cuda.manual_seed(seed)
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.compiler = compiler

        # set project directory
        self.nameproject = nameproject
        self.pathproject = os.path.join(pathproject, nameproject)

        # Set the graphic visualization
        self.plotter = None
        self.visdomname = "visdom.json"
        self.visdom_log_pathname = os.path.join(self.pathproject, self.visdomname)

        # initialization var
        self.print_freq = print_freq
        self.num_input_channels = 0
        self.num_output_channels = 0
        self.size_input = 0
        self.lr = 0.0001
        self.start_epoch = 0

        self.s_arch = ""
        self.s_optimizer = ""
        self.s_lerning_rate_sch = ""
        self.s_loss = ""

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.lrscheduler = None
        self.vallosses = 0

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
        cfg_model=None,
        cfg_loss=None,
        cfg_opt=None,
        cfg_scheduler=None,
    ):
        """Create neural network"""

        self.s_arch = arch
        self.s_optimizer = optimizer
        self.s_lerning_rate_sch = lrsch
        self.s_loss = loss

        if cfg_model is None:  # defaul configurate
            cfg_model = {}
        if cfg_loss is None:  # defaul configurate
            cfg_loss = {}
        if cfg_opt is None:  # defaul configurate
            cfg_opt = {}
        if cfg_scheduler is None:  # defaul configurate
            cfg_scheduler = {}

        # create project directory
        self.pathmodels = os.path.join(self.pathproject, "models")
        if not os.path.exists(self.pathproject):
            os.makedirs(self.pathproject)
        if not os.path.exists(self.pathmodels):
            os.makedirs(self.pathmodels)

        # create visual visdom plot
        self.plotter = gph.VisdomLinePlotter(env_name=self.nameproject, log_to_filename=self.visdom_log_pathname)

        self._create_model(arch, vocab_size, block_size, n_layers, n_embd, n_heads, pretrained, **cfg_model)
        self._create_loss(loss, **cfg_loss)
        self._create_optimizer(optimizer, lr, grad_clip, **cfg_opt)
        self._create_scheduler_lr(lrsch, **cfg_scheduler)

    def training(self, data_loader, epoch=0):
        pass

    def evaluate(self, data_loader, epoch=0):
        pass

    def test(self, data_loader):
        pass

    def inference(self, x):
        pass

    def representation(self, data_loader):
        pass

    def fit(self, train_loader, val_loader, epochs=100, snapshot=10):
        best_prec = 0
        print("\nEpoch: {}/{}(0%)".format(self.start_epoch, epochs))
        print("-" * 25)

        self.evaluate(val_loader, epoch=self.start_epoch)
        for epoch in range(self.start_epoch, epochs):
            try:
                self._to_beging_epoch(epoch, epochs, train_loader, val_loader)

                self.adjust_learning_rate(epoch)
                self.training(train_loader, epoch)

                print("\nEpoch: {}/{} ({}%)".format(epoch, epochs, int((float(epoch) / epochs) * 100)))
                print("-" * 25)

                prec = self.evaluate(val_loader, epoch + 1)

                # remember best eval metric and save checkpoint
                is_best = prec > best_prec
                best_prec = max(prec, best_prec)
                if epoch % snapshot == 0 or is_best or epoch == (epochs - 1):
                    self.save(epoch, best_prec, is_best, "chk{:06d}.pth.tar".format(epoch))

                self._to_end_epoch(epoch, epochs, train_loader, val_loader)

            except KeyboardInterrupt:
                print("Ctrl+C, saving snapshot")
                is_best = False
                best_prec = 0
                self.save(epoch, best_prec, is_best, "chk{:06d}.pth.tar".format(epoch))
                return

    def _to_beging_epoch(self, epoch, epochs, train_loader, val_loader, **kwargs):
        pass

    def _to_end_epoch(self, epoch, epochs, train_loader, val_loader, **kwargs):
        pass

    def _create_model(self, arch, vocab_size, block_size, n_layers, n_embd, n_heads, pretrained, **kwargs):
        pass

    def _create_loss(self, loss, **kwargs):
        pass

    def _create_optimizer(self, optimizer="adamw", lr=0.0001, grad_clip=1.0, **kwargs):
        self.optimizer = None
        self.grad_clip = grad_clip

        # create optimizer
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, amsgrad=True)
        elif optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer {} not found".format(optimizer))

        self.lr = lr
        self.s_optimizer = optimizer

    def _create_scheduler_lr(self, lrsch, **kwargs):
        self.lrscheduler = None

        if lrsch == "fixed":
            pass
        elif lrsch == "step":
            self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **kwargs)
        elif lrsch == "cyclic":
            self.lrscheduler = netlearningrate.CyclicLR(self.optimizer, **kwargs)
        elif lrsch == "exp":
            self.lrscheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **kwargs)
        elif lrsch == "plateau":
            self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **kwargs
            )  # 'min', patience=10
        else:
            assert False

        self.s_lerning_rate_sch = lrsch

    def adjust_learning_rate(self, epoch):
        """
        Update learning rate
        """

        if epoch == 0:
            self.plotter.plot("lr", "learning rate", epoch, self.lr)
            return self.lr

        # update
        if self.s_lerning_rate_sch == "fixed":
            lr = self.lr
        elif self.s_lerning_rate_sch == "plateau":
            self.lrscheduler.step(self.vallosses)
            for param_group in self.optimizer.param_groups:
                lr = float(param_group["lr"])
                break
        else:
            self.lrscheduler.step()
            lr = self.lrscheduler.get_last_lr()  # .get_lr()[0]

        # draw
        self.plotter.plot("lr", "learning rate", epoch, lr)

    def resume(self, pathnammodel):
        """
        Resume: optionally resume from a checkpoint
        """
        net = self.net.module if self.parallel else self.net
        start_epoch, prec = utils.resumecheckpoint(pathnammodel, net, self.optimizer)

        self.start_epoch = start_epoch
        return start_epoch, prec

    def save(self, epoch, prec, is_best=False, filename="checkpoint.pth.tar"):
        """
        Save model
        """
        print(">> save model epoch {} ({}) in {}".format(epoch, prec, filename))
        net = self.net.module if self.parallel else self.net
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": self.s_arch,
                # TODO August 27, 2023: Model config
                "vocab_size": self.vocab_size,
                "block_size": self.block_size,
                "n_layers": self.n_layers,
                "n_embd": self.n_embd,
                "n_heads": self.n_heads,
                "state_dict": net.state_dict(),
                "prec": prec,
                "optimizer": self.optimizer.state_dict(),
            },
            is_best,
            self.pathmodels,
            filename,
        )

    def load(self, pathnamemodel):
        bload = False
        if pathnamemodel:
            if os.path.isfile(pathnamemodel):
                print("=> loading checkpoint '{}'".format(pathnamemodel))
                checkpoint = (
                    torch.load(pathnamemodel)
                    if self.cuda
                    else torch.load(pathnamemodel, map_location=lambda storage, loc: storage)
                )

                self._create_model(
                    checkpoint["arch"],
                    checkpoint["vocab_size"],
                    checkpoint["block_size"],
                    checkpoint["n_layers"],
                    checkpoint["n_embd"],
                    checkpoint["n_heads"],
                    False,
                )
                self.net.load_state_dict(checkpoint["state_dict"])

                print("=> loaded checkpoint for {} arch!".format(checkpoint["arch"]))
                bload = True
            else:
                print("=> no checkpoint found at '{}'".format(pathnamemodel))
        return bload

    def __str__(self):
        return str(
            "Name: {} \n"
            "arq: {} \n"
            "loss: {} \n"
            "optimizer: {} \n"
            "lr: {} \n"
            "Model: \n{} \n".format(
                self.nameproject,
                self.s_arch,
                self.s_loss,
                self.s_optimizer,
                self.lr,
                self.net,
            )
        )
