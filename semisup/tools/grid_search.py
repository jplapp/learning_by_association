# basic hyperparameter grid search
# just calls scripts using exec, and logs result
import os
import subprocess
import ast
from time import time

import numpy as np
from concurrent.futures import *

from multiprocessing import current_process

PYTHON_NAME = 'python3'
NUM_THREADS = 4

used_gpus = np.zeros(NUM_THREADS)

def launchRun(name, params):
  train_scores = 0
  test_scores = 0

  start = time()

  params_list = []
  for a in params.items():
    params_list = params_list + ['--'+str(a[0]), str(a[1])]

  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = current_process()._identity[0]-1

  proc = subprocess.Popen([PYTHON_NAME, "../"+name+'.py'] + params_list, env=env, stdout=subprocess.PIPE)
  #proc = subprocess.Popen(['echo'] + params_list, stdout=subprocess.PIPE)
  while True:
    line = proc.stdout.readline()
    if line != '' and len(line) > 0:  # todo quits on all empty lines, improve detection
      res = line.rstrip().decode()
      print(res)
      type, scores = getAccuracy(res)
      if type is 'train':
        train_scores = scores
      elif type is 'test':
        test_scores = scores
    else:
      break

  best_train_score = np.min(train_scores)
  best_test_score = np.min(test_scores)

  duration = time() - start

  return best_train_score, best_test_score, train_scores, test_scores, duration


def getAccuracy(line):
  """
  parse result to find accuracy scores
  :param line: 
  :return: type, and array of accuracies
  """
  if line.startswith("train accuracies"):
    return 'train', ast.literal_eval(line[17:])
  elif line.startswith("test accuracies"):
    return 'test', ast.literal_eval(line[16:])

  return None, None


def create_task_list(combine_params):
  """
  recursively creates a grid-search task list
  params should be a dictionary
  """
  keys = list(combine_params.keys())

  task_list = []

  firstKey = keys[0]
  firstKeyValues = combine_params[firstKey]

  if len(keys) > 1:
    dict_without_first_key = combine_params.copy()
    del dict_without_first_key[firstKey]

    subtasks = create_task_list(dict_without_first_key)

    for v in firstKeyValues:
      new_dict = {firstKey: v}

      for task in subtasks:
        merged = new_dict.copy()
        merged.update(task)
        task_list = task_list + [merged]

  else:
    task_list = [{firstKey: v} for v in firstKeyValues]

  return task_list




def make_call(name, item, index):
  print('launching '+name, item)

  a,b,c,d,e = launchRun(name, item)

  return index, [a,b,c,d,e]

def run(name, params):
  executor = ProcessPoolExecutor(NUM_THREADS)
  tasks = create_task_list(params)

  futures = []
  index = 0
  for task in tasks:
    futures.append(executor.submit(make_call, name, task, index))
    index += 1

  for x in as_completed(futures):
    index, result = x.result()
    task = tasks[index]
    print(task, result)



if __name__ == '__main__':
  run("cifar100_train_eval", {
    "learning_rate": [1e-3, 2e-3],
    "eval_interval": [1000],
    "max_steps": [2000]
  })