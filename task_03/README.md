## Overfitting in RNNs

RNNs can overfit very well as we will
see. As they continue to fit to training
dataset, their performance on test data
will plateau or even worsen.


Keep track of it using a validation set,
save model at each iteration over
training data and pick the earliest, best,
validation performance.


bez wybierania właściwego wiersza - 50k (64) - >95.6%
wybieranie właściwego, dwie warstwy - 30k (64) - 97% (loss 0.016-0.017)