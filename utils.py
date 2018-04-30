import numpy as np
import theano
import theano.tensor as T
import os
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
#import mnist
import _pickle as pickle
from theano.sandbox.rng_mrg import MRG_RandomStreams
import operator
import sys
sys.path.append("/u/ywu/Documents/eval_GAN/training_GAN/iwae/")
import lasagne
import progressbar
from nn import*

np.random.seed(123)

sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))

DATASETS_DIR = '/u/ywu/Documents/eval_GAN/training_GAN/iwae/datasets'


def fixbinary_mnist(data):
  def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])
  with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_train.amat')) as f:
    lines = f.readlines()
  train_data = lines_to_np_array(lines).astype('float32')
  with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_valid.amat')) as f:
    lines = f.readlines()
  validation_data = lines_to_np_array(lines).astype('float32')
  with open(os.path.join(DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_test.amat')) as f:
    lines = f.readlines()
  test_data = lines_to_np_array(lines).astype('float32')
  permutation = np.random.RandomState(seed=2919).permutation(train_data.shape[0])
  train_data = train_data[permutation]
  if data == "train":
    return np.concatenate([train_data, validation_data], axis=0)
  elif data == "test":
    return test_data

def load_mnist(data,label=False):
  (X_train,y_train),(X_test,y_test) = mnist.load_data()
  permutation = np.random.RandomState(seed=2919).permutation(X_test.shape[0])
  X_test = X_test[permutation].astype(np.float32)
  y_test = y_test[permutation].astype(np.int32)
  ind2 = np.random.RandomState(seed=2919).permutation(X_train.shape[0])
  X_train = X_train[ind2].astype(np.float32)
  y_train = y_train[ind2].astype(np.float32)
  X_train /= 256
  X_test /= 256
  if data == "train":
    if label:
      return X_train,y_train
    return X_train
  elif data == "test":
    if label:
      return X_test,y_test
    return X_test

def load_simulated(directory_name):
  exact_h = np.load(os.path.join(directory_name,'noise.npy'))
  X_test = np.load(os.path.join(directory_name,'gen.npy'))

  return X_test,exact_h


def load_model(model_type,aux):
  #load_model returns a generator, which is a python function that takes a input (latent variable) and returns a sample.

  if model_type == 'gan10':
    gen = gan_gen_net10()
    SAVEPATH = './models/'
    filename = model_type
    filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
    print ('load model '+filename)
    with open(filename, 'rb') as f:
      data = pickle.load(f, encoding='latin1')
    lasagne.layers.set_all_param_values(gen, data)
    def generator(z):
      return lasagne.layers.get_output(gen,z)
    return None, generator


  elif model_type == 'gan50':
    gen = gen_net50()
    SAVEPATH = './models/'
    filename = model_type
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
    print ('load model '+filename)
    with open(filename, 'rb') as f:
      data = pickle.load(f, encoding='latin1')
    lasagne.layers.set_all_param_values(gen, data)
    def generator(z):
      return lasagne.layers.get_output(gen,z)
    return None, generator


  elif model_type =='vae10':
    gen = vae_gen_net10()
    if aux[0] == 'c':
      SAVEPATH = './models/'
      filename = model_type
    elif aux[0] == 'b':
      SAVEPATH = './models/'
      filename = model_type
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
    print ('load model '+filename)
    with open(filename, 'rb') as f:
      data = pickle.load(f, encoding='latin1')
    lasagne.layers.set_all_param_values(gen, data)
    def generator(z):
      return lasagne.layers.get_output(gen,z)[:,:784]
    return None, generator


  elif model_type == 'vae50':
    gen = gen_net50()
    SAVEPATH = './models/'
    filename = model_type
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
    print ('load model '+filename)
    with open(filename, 'rb') as f:
      data = pickle.load(f, encoding='latin1')
    data = data[:-1]
    lasagne.layers.set_all_param_values(gen, data)
    def generator(z):
      return lasagne.layers.get_output(gen,z)
    return None, generator


def sampler(mu, log_sigma,std=False):
  seed = 132
  if "gpu" in theano.config.device:
    #from theano.sandbox.rng_mrg import MRG_RandomStreams
    #srng = MRG_RandomStreams(seed=132)
    srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
  else:
    srng = T.shared_randomstreams.RandomStreams(seed=seed)
  eps = srng.normal(mu.shape)
  # Reparametrize
  if std:
    z = mu + T.exp(log_sigma) * eps
  else:
    z = mu + T.exp(0.5 * log_sigma) * eps

  return z


def load_encoder(model_type,aux,eval_np=False): ##for VAE
  gen = vae_gen_net10() if model_type == 'vae10' else gen_net50()
  enc = enc_net10() if model_type == 'vae10' else enc_net50()
  if 'vae' in model_type:
    SAVEPATH = './models/'
    filename = model_type
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join(SAVEPATH, '%s.pkl' % (filename))
    print('load model '+filename)
    with open(filename, 'rb') as f:
      data = pickle.load(f, encoding='latin1')
    lasagne.layers.set_all_param_values(gen, data)

    def encoder(x):
      hid_gen = lasagne.layers.get_output(enc,x)
      mean = hid_gen[:,:10]
      log_sigma = hid_gen[:,10:]
      h_sample = sampler(mean,log_sigma)
      if eval_np:
        return h_sample.eval(),mean.eval(),log_sigma.eval()
      return mean,log_sigma

    return encoder


def estimate_lld(model,minibatch,num_sam,size=1):
  n_examples = minibatch.shape[0]
  num_minibatches = n_examples/size
  minibatch = minibatch.astype(np.float32)
  srng = MRG_RandomStreams(seed=132)
  batch = T.fmatrix()
  index = T.lscalar('i')
  mini = sharedX(minibatch)
  print('num_samples: '+str(num_sam))
  lld = model.log_marginal_likelihood_estimate(batch,num_sam,srng)

  get_log_marginal_likelihood = theano.function([index], T.sum(lld),givens = {batch:mini[index*size:(index+1)*size]})

  pbar = progressbar.ProgressBar(maxval=num_minibatches).start()
  sum_of_log_likelihoods = 0.
  for i in range(num_minibatches):
    summand = get_log_marginal_likelihood(i)
    sum_of_log_likelihoods += summand
    pbar.update(i)
  pbar.finish()

  marginal_log_likelihood = sum_of_log_likelihoods/n_examples
  print("estimate lld: "+str(marginal_log_likelihood))

def plot_gen(generated_images, name, n_ex=16,dim=(6,6), figsize=(10,10)):
  plt.figure(figsize=figsize)
  generated_images = generated_images.reshape(generated_images.shape[0],28,28)
  for i in range(generated_images.shape[0]):
    plt.subplot(dim[0],dim[1],i+1)
    img = generated_images[i,:,:]
    plt.imshow(img,cmap='Greys')
    plt.axis('off')
  plt.tight_layout()
  plt.show()
  plt.savefig('./'+name+'.pdf',format='pdf')
  plt.close()

def plot_real(X,name,n_ex=36,dim=(6,6), figsize=(10,10) ):
  generated_images = X.reshape(X.shape[0],28,28)
  plt.figure(figsize=figsize)
  for i in range(generated_images.shape[0]):
    plt.subplot(dim[0],dim[1],i+1)
    img = generated_images[i,:,:]
    plt.imshow(img,cmap='Greys')
    plt.axis('off')
  plt.tight_layout()
  plt.show()
  plt.savefig('./'+name+'real_img.pdf',format='pdf')
  plt.close()


