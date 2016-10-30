Character-level CNNs in NLP
========

This packages contains the Theano-based Keras implementation of "Very Deep Convolutional Networks for Natural Language Processing" [Conneau et al. (2016)](https://arxiv.org/abs/1606.01781). This Keras implemenation contains the first type of deep model with 9 layers and uses max-pooling with kernel size 3 and stride 2. The code is easily adaptable to the deeper models.
Also this package contains the Keras implementation of "Character-level Convolutional Networks for Text
Classificationin" neural network model by [Zhang et al. (2015)](https://arxiv.org/abs/1509.01626) that is available at [Crepe](https://github.com/zhangxiangxiao/Crepe), which was originally written with Torch.

Download and Installation
-------

- Install [Theano](http://deeplearning.net/software/theano/install.html)
- Install [Keras](https://github.com/fchollet/keras#installation)
- Clone the repository
```
git clone https://github.com/sci-lab-uoit/cnn-nlp
```

Details
-------

To use the very deep CNN model [Conneau et al. (2016)](https://arxiv.org/abs/1606.01781), change deepFlag to 1 and to use the simpler CNN model [Zhang et al. (2015)](https://arxiv.org/abs/1509.01626),  set deepFlag to 0.

Please set the input data path to your directory.

Running
-------

Then run the model using the following command:

```
python py-cdnn.py -f 1 -m 1300 -e 10 -b 64 --z1 1024 --z2 1024 -p /home/neil/projects/py-cdnn-text
```

Citation
--------

To cite Character-level CNNs in NLP for publications use:

Eren Gultepe, Neil Seward, and Masoud Makrehchi (2016). Character-level CNNs in NLP: A Python package for convolutaionl neural networks in natural language processing
