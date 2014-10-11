# -*- coding: utf-8 -*-
"""
Train algorithm based on spipy.optimize

"""

from neurolab.core import Train
import neurolab.tool as tool

class TrainSO(Train):
    """
    Train class Based on scipy.optimize

    """

    def __init__(self, net, input, target, **kwargs):
        self.net = net
        self.input = input
        self.target = target
        self.kwargs = kwargs
        self.x = tool.np_get_ref(net)
        self.lerr = 1e10

    def grad(self, x):
        self.x[:] = x
        gr = tool.ff_grad(self.net, self.input, self.target)[1]
        return gr

    def fcn(self, x):
        self.x[:] = x
        err = self.error(self.net, self.input, self.target)
        self.lerr = err
        return err

    def step(self, x):
        self.epochf(self.lerr, self.net, self.input, self.target)

    def __call__(self, net, input, target):
        raise NotImplementedError("Call abstract metod __call__")


class TrainBFGS(TrainSO):
    """
    BroydenFletcherGoldfarbShanno (BFGS) method
    Using scipy.optimize.fmin_bfgs

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin_bfgs
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        self.kwargs['maxiter'] = self.epochs

        x = fmin_bfgs(self.fcn, self.x.copy(), fprime=self.grad, callback=self.step,
                      **self.kwargs)
        self.x[:] = x


class TrainCG(TrainSO):
    """
    Newton-CG method
    Using scipy.optimize.fmin_ncg

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin_cg
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        x = fmin_cg(self.fcn, self.x.copy(), fprime=self.grad, callback=self.step, **self.kwargs)
        self.x[:] = x
        return None


class TrainNCG(TrainSO):
    """
    Conjugate gradient algorithm
    Using scipy.optimize.fmin_ncg

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin_ncg
        #if 'disp' not in self.kwargs:
        #    self.kwargs['disp'] = 0
        x = fmin_ncg(self.fcn, self.x.copy(), fprime=self.grad, callback=self.step, **self.kwargs)
        self.x[:] = x
        return None

class TrainRHC(TrainSO):
    """
    Randomized Hill Climbing (RHC) method
    Using scipy.optimize.fmin

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import fmin
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        self.kwargs['maxiter'] = self.epochs

        x = fmin(self.fcn, self.x.copy(), callback=self.step, **self.kwargs)
        self.x[:] = x
		

class TrainBH(TrainSO):
    """
    Basin-Hopping (BH) method
    Using scipy.optimize.basinhopping

    :Support networks:
        newff (multi-layers perceptron)
    :Parameters:
        input: array like (l x net.ci)
            train input patterns
        target: array like (l x net.co)
            train target patterns
        epochs: int (default 500)
            Number of train epochs
        show: int (default 100)
            Print period
        goal: float (default 0.01)
            The goal of train

    """

    def __call__(self, net, input, target):
        from scipy.optimize import basinhopping
        if 'disp' not in self.kwargs:
            self.kwargs['disp'] = 0
        self.kwargs['niter'] = self.epochs
		
        def customcallback(xtrial, energy_trial, accept):
            print(xtrial)
            self.step(xtrial)
		
        x = basinhopping(self.fcn, self.x.copy(), callback=customcallback, **self.kwargs).x
        self.x[:] = x