import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tqdm.notebook import tqdm

def count_parameters(model):
  res = 0
  for param in model.parameters():
    res += torch.numel(param)
  return res

def estimate_accuracy(model, loader, max_iter = 100):
    eval_time = 0
    sample_size = 0
    top1_hits = 0
    top5_hits = 0
    
    for iter_no, (X_batch, y_batch) in enumerate(loader):
        eval_start_time = time.time()
        predictions = model(X_batch.cuda())
        eval_end_time = time.time()
        eval_time += (eval_end_time - eval_start_time)
        predicted_labels = torch.argsort(predictions, axis = 1, descending = True)[:, :5]
        
        is_correct = (predicted_labels.cpu() == y_batch.reshape(-1, 1))
        top1_hits += is_correct[:, 0].sum()
        top5_hits += is_correct.sum()
        sample_size += X_batch.shape[0]
        del X_batch
        del predictions
        if iter_no >= max_iter:
          break
    return top1_hits / sample_size, top5_hits / sample_size

def plot_history(history, fitness):
  hist = np.array(history)

  sns.set_theme(font_scale = 1.5, style = 'darkgrid')

  num_iter = np.arange(1, len(hist) + 1)

  plt.figure(figsize = (30, 12))
  plt.subplot(1, 2, 1)
  plt.title('Метрики')
  sns.lineplot(x = num_iter, y = hist[:, 0], label = 'Доля весов')
  sns.lineplot(x = num_iter, y = hist[:, 1], label = 'Точность 1')
  sns.lineplot(x = num_iter, y = hist[:, 2], label = 'Точность 5')
  plt.xlabel('Номер итерации')
  plt.subplot(1,2,2)
  plt.title('Функция потерь')
  sns.lineplot(x = num_iter, y = fitness)
  plt.xlabel('Номер итерации')
  plt.show()

def stack_parameters(model):
  params = []
  for param in model.parameters():
    params.append(param.ravel())
  return torch.hstack(params)

def get_metrics(cur_model, train_loader, max_iter = 200):
    m_acc1, m_acc5 = estimate_accuracy(cur_model, train_loader, max_iter)
    stacked_param = stack_parameters(cur_model)
    pruned_weights = (stacked_param == 0).sum()
    weight_ratio = 1 - pruned_weights / len(stacked_param)
    return weight_ratio.item(), m_acc1, m_acc5

def threshold_prune(model, gamma = 0.1):
  res = deepcopy(model)
  with torch.no_grad():
    for param in res.parameters():
      param *= torch.where(torch.abs(param) < gamma, 0, 1)
  return res

def plot_threshold_pruning(model, train_loader,
 thresh_range = np.linspace(0, 0.001, 10), max_iter = 1000):
  weights = []
  acc1_hist = []
  acc5_hist = []
  for gamma in tqdm(thresh_range):
    pruned = threshold_prune(model, gamma)
    w, acc1, acc5 = get_metrics(pruned, train_loader, max_iter = max_iter)
    weights.append(w)
    acc1_hist.append(acc1.item())
    acc5_hist.append(acc5.item())
  plt.figure(figsize = (18, 9))
  plt.subplot(1, 2, 1)
  sns.lineplot(x = thresh_range, y = weights, label = 'Доля весов')
  sns.lineplot(x = thresh_range, y = acc1_hist, label = 'Точность топ 1')
  sns.lineplot(x = thresh_range, y = acc5_hist, label = 'Точность топ 5')
  plt.xlabel('Граница')
  plt.title('Метрики')
  plt.subplot(1, 2, 2)
  weights = np.array(weights)
  sns.lineplot(x = 1 - weights, y = acc5_hist, label = 'Точность топ 5')
  sns.lineplot(x = 1 - weights, y = acc1_hist, label = 'Точность топ 1')
  plt.xlabel('Доля удаленных весов')
  plt.ylabel('Точность')
  plt.title('Точность в зависимости от доли удаленных весов')


