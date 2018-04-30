import argparse
import os
import sys

import numpy as np
import theano

from sampling import samplers

import utils
import operator
import time

rng = np.random.RandomState(31)
sharedX = (lambda X: theano.shared(np.asarray(X, dtype=theano.config.floatX)))


def main(exps, model_type, aux, data, num_steps, num_samples=16, hdim=10, num_test=100, sigma=0.03, prior="normal",
         reverse=False, evalu=False, plot_posterior=False):
  run = True
  # load model
  print('model: ' + model_type + aux)
  print('data: ' + data)
  if 'data' == 'continuous':
    print('sigma: ' + str(sigma))
  permute = True
  model, generator = utils.load_model(model_type, aux)

  # get data
  if exps == "train":
    print('run train')
    if data == 'continuous':
      X = utils.load_mnist('train')[:50000]
      X += rng.uniform(0, 1, size=X.shape) / 256
    if data == 'binary':
      X = utils.fixbinary_mnist('train')[:50000]
  elif exps == "valid":
    print('run valid')
    if data == 'continuous':
      X = utils.load_mnist('test')[:5000]
      X += rng.uniform(0, 1, size=X.shape) / 256
    if data == 'binary':
      X = utils.fixbinary_mnist('test')[:5000]
  elif exps == "test":
    print('run test')
    if data == 'continuous':
      X = utils.load_mnist('test')[5000:]
      X += rng.uniform(0, 1, size=X.shape) / 256
    if data == 'binary':
      X = utils.fixbinary_mnist('test')

  if plot_posterior:
    directory_name = "vis/" + model_type + aux + 'num_steps' + str(num_steps) + 'sigma' + str(
      sigma) + exps + 'posterior_sample/'
    if os.path.exists(directory_name):
      finalstate = np.load(directory_name + 'final_state.npy')
      pf = np.load(directory_name + 'pf.npy')
      lld = 0
      run = False
  ##Shuffle the data
  if permute:
    permutation = np.random.RandomState(seed=2919).permutation(X.shape[0])
    X = X[permutation][:num_test]
  results = {'lld': [], 'pf': []}
  iwae_time = None
  if evalu:
    print('IWAE evalution...')
    t_start = time.time()
    batch = np.reshape(X, (num_test, 28 * 28))
    utils.estimate_lld(model, batch, in_sam)
    t_end = time.time()
    iwae_time = t_end - t_start
    print('IWAE Eval time: ' + str(iwae_time) + ' seconds')
  elif run:
    print('num_test: ' + str(num_test))
    t_start = time.time()
    lld, pf, finalstate = samplers.run_ais(generator, X, num_samples, num_steps, sigma, hdim, L, eps, data, prior, model_name=model_type)
    t_end = time.time()
    ais_time = t_end - t_start
    print('AIS forward Eval time: ' + str(ais_time) + ' seconds')
    results['lld'].append(lld)
    results['pf'].append(pf)
  if plot_posterior:
    directory_name = "vis/" + model_type + aux + 'num_steps' + str(num_steps) + 'sigma' + str(
      sigma) + exps + 'posterior_sample/'
    if not os.path.exists(directory_name):
      os.makedirs(directory_name)
    post_img = (generator(finalstate)).eval()
    post_img = post_img.reshape(num_samples, num_test, 28, 28)
    img_size = int(np.sqrt(num_test))
    exppf = np.exp(pf - np.max(pf, axis=0))
    sampling_prob = exppf / np.sum(exppf, axis=0)
    choices = []
    for i in range(num_test):
      choices.append(rng.choice(num_samples, 3, p=sampling_prob[:, i]))
    choices = np.vstack(choices)
    for i in range(3):
      utils.plot_gen(post_img[choices[:, i], np.arange(num_test)],
                     directory_name + model_type + aux + 'posterior_sample' + str(i) + "num_steps" + str(
                       num_steps) + 'sigma' + str(sigma), n_ex=num_test, dim=(10, 3))
    np.save(directory_name + 'final_state.npy', finalstate)
    np.save(directory_name + 'pf.npy', pf)
    if exps == 'post2tra':
      return X, post_img[choices[:, 0], np.arange(num_test)]
  return {'res': results, 'ais_time': ais_time, 'iwae_time:': iwae_time}


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--exps", default="valid",
                      type=str)  ## 'train'|'valid'|'test'
  parser.add_argument("--model", default="vae50", type=str)
  parser.add_argument("--hdim", default=10, type=int)  ## 10 | 50, hidden code dimension

  parser.add_argument("--prior", default="normal",
                      type=str)  ## 'normal'|'recog1'|'recog2', if recog1 then we use VAE q-dist as initial dist of AIS chain, recog2 for IWAE q-dist.

  parser.add_argument("--sigma", default=0.025, type=float)  ## variance hyperparameter
  parser.add_argument("--num_test", default=36, type=int)  ## num of examples for evaluation
  parser.add_argument("--num_steps", default=200, type=int)  ## num of intermediate distributions/ schedule size
  parser.add_argument("--num_samples", default=16, type=int)  ## num of AIS chains


  # ---------- useless -----------------
  parser.add_argument("--data", default="continuous",
                      type=str)  ##'binary'|'continuous', the data type, which also decides the observation model p(x|h)
  parser.add_argument("--aux", default="c", type=str)  ##'c' for continuous, we stick to it
  parser.add_argument("--in_sam", default=16, type=int)  ##num of IWAE bound samples
  parser.add_argument("--num_simulated_samples", default=100, type=int)  ## num of simulated samples for BDMC
  parser.add_argument("--eps", default=0.01, type=float)  ##starting stepsize
  parser.add_argument("--L", default=10, type=int)  ##num of leapfrog steps
  parser.add_argument("--R", default=False, type=bool)  ##whether do BDMC reverse chain
  parser.add_argument("--evalu", default=False, type=bool)  ##whether use IWAE bound evaluation
  parser.add_argument("--plot_posterior", default=False, type=bool)  ##whether plot posterior samples
  # ------------------------------------

  args = parser.parse_args()
  args_dict = vars(args)
  locals().update(args_dict)
  result = main(exps,
                model,
                aux,
                data,
                num_steps,
                num_samples=num_samples,
                num_test=num_test,
                hdim=hdim,
                evalu=evalu,
                reverse=R,
                sigma=sigma,
                prior=prior,
                plot_posterior=plot_posterior)
