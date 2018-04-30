### Likelihood Estimation of Deep Generative Models with Annealed Importance Sampling

---

#### Requirements

Theano 1.0.1

Lasagne 0.2.dev1

Keras 2.0.6

Tensorflow 1.0.1

#### Code Structure

- section 4.1 related code:

  all the `./*.py` but `./util.py` and `dgm_tester.py`

  to run the experiments, see `property_tester.ipynb`

- section 4.2 related code:

  all the directories but ./assets + `./util.py` + `dgm_tester.py`

  to run the experiments, see `dgm_tester.py` and `run_dgm.sh`

- sample plots and experiment log

  see ./assets

#### References

[1] Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint

arXiv:1509.00519, 2015.

[2] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron

Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing

systems, pages 2672–2680, 2014.

[3] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[4] Yann LeCun. Gradient-based learning applied to document recognition. 1998.

[5] Yujia Li, Kevin Swersky, and Rich Zemel. Generative moment matching networks. In International

Conference on Machine Learning, pages 1718–1727, 2015.

[6] Radford M Neal. Annealed importance sampling. Statistics and computing, 11(2):125–139, 2001.5

[7] Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. arXiv

preprint arXiv:1601.06759, 2016.

[8] Emanuel Parzen. On estimation of a probability density function and mode. The annals of mathematical

statistics, 33(3):1065–1076, 1962.

[9] Lucas Theis, Aäron van den Oord, and Matthias Bethge. A note on the evaluation of generative models.

arXiv preprint arXiv:1511.01844, 2015.

[10] https: //github.com/tonywu95/eval_gen

[11] Yuhuai Wu, Yuri Burda, Ruslan Salakhutdinov, and Roger Grosse. On the quantitative analysis of

decoder-based generative models. arXiv preprint arXiv:1611.04273, 2016.6

[12] https: //github.com/jiamings/ais