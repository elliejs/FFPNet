# FFPNet
A [feed forward perceptron network](https://en.wikipedia.org/wiki/Feedforward_neural_network) written in raw C89. It scores 86% accuracy on the MNIST database of handwritten numbers (included).

sample execution: `./ffpnet 10 0.0002`

This execution will both train and test the net, as well as exporting the net for later use.

This project was previously affectionately known as Arwen

Sample output:
```
./ffpnet 10 0.0002
percent correct So Far:-nan  this batch: 0.00
percent correct So Far:0.14  this batch: 0.14
percent correct So Far:0.14  this batch: 0.15
percent correct So Far:0.17  this batch: 0.23
percent correct So Far:0.25  this batch: 0.49
percent correct So Far:0.29  this batch: 0.46
percent correct So Far:0.32  this batch: 0.42
percent correct So Far:0.35  this batch: 0.54
percent correct So Far:0.37  this batch: 0.54
percent correct So Far:0.40  this batch: 0.61
percent correct So Far:0.43  this batch: 0.72
...
percent correct So Far:0.84  this batch: 0.98
percent correct So Far:0.84  this batch: 0.95
percent correct So Far:0.84  this batch: 0.79
percent correct So Far:0.84  this batch: 0.98

percent correct (training): 0.84

percent correct (testing): 0.86
(8583 out of 10000)
```
