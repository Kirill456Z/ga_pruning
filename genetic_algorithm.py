import numpy as np
import scipy.stats as sps
import copy
import numpy as np
import torch
from utils import count_parameters, plot_history
from IPython.display import clear_output
from tqdm.notebook import tqdm

### Representations

class BaseRepresenation:
    """Base class to transform between binary string and object"""

    def __init__(self, length : int):
        self.length = length
    
    def get_object(self, gene):
        """Get object from binary string"""
        return np.ones(self.length)
    
class BinaryEncodedNumber(BaseRepresenation):
    """Encode real number as binary string (testing simple GA)"""
    def __init__(self, min_value = -1, max_value = 1, precision = 10):
        """
        Attributes
        ----------
        min_value : float
          smallest encoded value
        max_value : float
          largest ecoded value
        precision : int
          encoding precision (number of bits)
        """
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(precision)
        
    def get_object(self, gene):
        scale = (1 / 2) ** np.arange(1, self.length + 1)
        return (gene * scale).sum(axis = 1) * (self.max_value - self.min_value) + self.min_value
    
    
class ModelPruningEncoder(BaseRepresenation):
    """Encode model parameters as binary string where 0 corresponds to pruned parameter"""
    def __init__(self, model):
        self.model = model
        self.shapes = [param.shape for param in model.parameters()]
        length = np.array([param.numel() for param in model.parameters()]).sum()
        super().__init__(length)
    
    def get_object(self, gene):
        gene_reshaped = self.__reshape_gene(gene)
        pruned_model = self.__prune_model(gene_reshaped)
        return pruned_model
        
    def __reshape_gene(self, gene):
        res = []
        cur_idx = 0
        for shape in self.shapes:
            numel = np.array(shape).prod()
            cur_part = gene[cur_idx : cur_idx + numel]
            cur_idx += numel
            res.append(torch.Tensor(cur_part.reshape(shape)))
        return res
    
    def __prune_model(self, mask):
        res = copy.deepcopy(self.model)
        with torch.no_grad():
            for param_ref, mask_part in zip(res.parameters(), mask):
                param_ref.data = param_ref * mask_part
        return res

class CNNMaskPrune(BaseRepresenation):
    """
    Encode model parameters as binary string
    in fully connected layers each parameters is reprented as a bit
    in convolutional layers each filter is represented as a bit
    (0 corresponds to either pruned fc paramter or one convolution filter)
    """
    def __init__(self, model):
      self.model = model
      self.fc_length = 0
      self.conv_length = 0
      self.conv_params = 0
      for _, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
          self.conv_length += next(iter(layer.parameters())).shape[0]
          self.conv_params += count_parameters(layer)
      for _, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
          self.fc_length += count_parameters(layer)
      length = self.fc_length + self.conv_length
      super().__init__(length)

    def __prune_conv(self, model, gene):
      cur_i = 0
      for _, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
          param = next(iter(layer.parameters()))
          len = param.shape[0]
          cur_gen = gene[cur_i : cur_i + len].reshape(-1, 1, 1, 1)
          with torch.no_grad():
            param *= cur_gen
          cur_i += len

    def __prune_fc(self, model, gene):
      cur_i = 0
      for _, layer in model.named_modules():    
        if isinstance(layer, torch.nn.Linear):
          for param in layer.parameters():
            len = torch.numel(param)
            cur_gen = gene[cur_i: cur_i + len].reshape(param.shape)
            with torch.no_grad():
              param *= cur_gen
            cur_i += len
      return self

    def get_object(self, gene):
      res = copy.deepcopy(self.model)
      conv_gen = gene[0 : self.conv_length]
      fc_gen = gene[self.conv_length :]
      self.__prune_conv(res, conv_gen)
      self.__prune_fc(res, fc_gen)
      return res


### Mutators 

class BaseMutator:
    def mutate(self, gene):
        return gene

class RandomMutator(BaseMutator):
    def __init__(self, mutation_prob):
        self.mut_prob = mutation_prob
        
    def mutate(self, gene):
        index = sps.bernoulli(self.mut_prob).rvs(gene.shape)
        return np.where(index == 1, 1 - gene, gene)


class OneWayMutator(BaseMutator):
  def __init__(self, mutation_prob = 0.15):
    self.mut_prob = mutation_prob
  
  def mutate(self, gene):
    mutation_type = sps.bernoulli(0.5).rvs(1)[0]
    index = sps.bernoulli(self.mut_prob).rvs(gene.shape)
    return np.where(np.logical_and(index == 1, gene == mutation_type), 1 - gene, gene)

class ConvGeneMutator(BaseMutator):
    """Class for mutating binary strings which represent CNN"""
    def __init__(self, fc_len, conv_len, fc_proba = 0.15, conv_proba = 0.01):
      """
      Attributes
      -----------
      fc_len : int
        number of bits which represent fully connected layer paramters
      conv_len : int
        number of bits which represent convolution filters
      fc_proba : float
        probability of fc paramter bit changing
      conv_proba : float
        probability of convolution filter bit changing
      """
      self.fc_len = fc_len
      self.conv_len = conv_len
      self.fc_mutator = RandomMutator(fc_proba)
      self.conv_mutator = RandomMutator(conv_proba)

    def mutate(self, gen):
      conv_mutated = self.conv_mutator.mutate(gen[:, :self.conv_len])
      fc_mutated = self.fc_mutator.mutate(gen[:, self.conv_len : ])
      return np.hstack((conv_mutated, fc_mutated))

class FixedSparsityMutator:
    def __init__(self, sparsity = 0.4, num_changes_distr = sps.randint(1, int(5e4))):
      self.sparsity = sparsity
      self.num_changes_distr = num_changes_distr

    def __change_bits(self, gene, is_changing_zeros = True, num_changes = 0):
      num_changes = int(num_changes)
      if is_changing_zeros:
        bits_idx = np.arange(0, len(gene))[gene == 0]
      else:
        bits_idx = np.arange(0, len(gene))[gene == 1]
      idxs_to_change = np.random.choice(bits_idx, num_changes, replace = False)
      res = gene
      res[idxs_to_change] = 1 - res[idxs_to_change]
      return res

    def __mutate_single(self, gene):
      gene = np.ravel(gene)
      ones_count = gene.sum()
      ones_desired = np.floor(len(gene) * self.sparsity)
      if ones_count > ones_desired:
        gene = self.__change_bits(gene, False, ones_count - ones_desired)
      if ones_count < ones_desired:
        gene = self.__change_bits(gene, True, ones_desired - ones_count)
      num_changes = self.num_changes_distr.rvs(1)[0]
      res = gene
      res = self.__change_bits(res, True, num_changes)
      res = self.__change_bits(res, False, num_changes)
      return gene.reshape(1, -1)
    
    def mutate(self, genes):
      return np.vstack([self.__mutate_single(gene) for gene in genes])

### Selectors

class BaseSelector:
    def select(self, population, desired_size):
        if desired_size >= len(population):
            return population
        return population[np.random.choice(np.arange(0, len(population)), desired_size, False), :]
    
class FitnessSelector:
    """Select genes with lowest fittness function value"""
    def __init__(self, fitness_func = None):
        """
        Attributes
        -----------
        fitness_func : func(model) -> float
          should return fitness of a given model
        """
        self.fitness_func = fitness_func
        
    def select(self, population, desired_size, fitness = None):
        if fitness is None:
            if self.fitness_func is None:
              raise Exception('neither loss function nor loss values were provided')
            fitness = self.fitness_func(population)
        index = np.argsort(fitness)
        return population[index[:desired_size]]

### Crossovers

class BaseCrossover:
    def crossover(self, population, desired_size):
        index = np.arange(0, len(population))
        new_points = population[np.random.choice(index, desired_size - len(population)), :]
        return np.vstack((population, new_points))
    
class OnePointCrossover(BaseCrossover):
    def crossover(self, population, desired_size):
        gen_length = population.shape[1]
        new_points = []
        while len(population) + len(new_points) < desired_size:
            index = np.arange(0, len(population))
            parents = population[np.random.choice(index, 2, False), :]
            split = np.random.randint(0, gen_length + 1)
            new = np.hstack((parents[0][: split], parents[1][split : ]))
            new_points.append(new)
        return np.vstack((population, np.array(new_points)))

class LossCrossover(BaseCrossover):
    def __init__(self, winner_prob = 0.75):
      self.winner_prob = winner_prob

    def crossover(self, population, desired_size, loss):
        N = population.shape[0]
        index = np.arange(0, N)
        new_points = []
        while len(population) + len(new_points) < desired_size:
          p1, p2 = np.random.choice(index,2, replace = False)
          parent1 = population[p1]
          parent2 = population[p2]
          loss1, loss2 = loss[p1], loss[p2]
          winner = parent1 if loss1 < loss2 else parent2
          looser = parent2 if loss1 < loss2 else parent1
          take_from_winner = sps.bernoulli(self.winner_prob).rvs(len(winner))
          new = np.where(take_from_winner == 1, winner, looser)
          new_points.append(new)
        return np.vstack((population, np.array(new_points)))

### Algorithms

class MultiTargetLoss:
  def __init__(self, representation, mutator, selector, crossover, metrics, loss):
    self.representation = representation
    self.mutator = mutator
    self.selector = selector
    self.crossover = crossover
    self.metrics = metrics
    self.loss = loss

  def prune(self, max_iter = 30, N = 30, selected_num = 5):
    gene_length = self.representation.length
    initial = torch.ones(gene_length)
    population = np.vstack([self.mutator.mutate(initial.reshape(1, -1)) for _ in tqdm(range(N))])
    population = np.vstack([population, initial]) 
    print('start population generated')
    elite_fitness_history = []
    elite_metrics_history = []
    cur_elite = None

    for iter_no in tqdm(range(max_iter)):
      clear_output(wait = True)
      if iter_no != 0:
        plot_history(elite_metrics_history, elite_fitness_history)
      population_fitness = []
      population_metrics = []
      for i, gen in enumerate(tqdm(population)):
        pruned = self.representation.get_object(torch.Tensor(gen).cuda())
        metrics = self.metrics(pruned)
        fitness = self.loss(metrics)
        population_fitness.append(fitness)
        population_metrics.append(metrics)
    
      elite_index = np.argmin(population_fitness)
      best_fitness = population_fitness[elite_index]
      elite_fitness_history.append(best_fitness.item())
      elite_metrics_history.append(population_metrics[elite_index])
      elite = population[elite_index]
      cur_elite = elite 
    
      selected = self.selector.select(population, selected_num, population_fitness)
      del population
      crossed = self.crossover.crossover(selected, N - 1, population_fitness)
      del selected
      mutated = self.mutator.mutate(crossed)
      del crossed
      population = np.vstack((mutated, elite.reshape(1, -1)))
    return self.representation.get_object(torch.Tensor(cur_elite).cuda())






