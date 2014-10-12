"""
Train algorithm based on deap

"""

from neurolab.core import Train
import neurolab.tool as tool

class TrainGA(Train):
    """
    Train class Based on deap

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
        import random
        from deap import algorithms
        from deap import base
        from deap import creator
        from deap import tools
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Attribute generator
        toolbox.register("attr_bool", random.randint, 0, 1)

        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(self.x))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def evalOneMax(individual):
            err = self.fcn(individual)
            self.step(individual)
            return (-err,)

        toolbox.register("evaluate", evalOneMax)
        toolbox.register("mate", tools.cxBlend)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.05, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=self.epochs, halloffame=hof, verbose=False)

        self.x[:] = hof[0]