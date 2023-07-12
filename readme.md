## Classification with LSTM


We will be using LSTM (Long-Short Term Memory) model for classification. The architecture of our model looks like :

<p align="center"><img src="https://www.tensorflow.org/static/text/tutorials/images/bidirectional.png" width="75%" align="center"></p>

For more details on LSTM, please refer to class lectures.

Use an Embedding layer, followed by a LSTM layer, and a linear layer.

We will then classify the Clickbait and Web of science dataset for this task.


## Classification with LSTM + Attention


A potential issue with vanilla LSTM approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus. Attention mechanism helps to look at all hidden states from sequence for making predictions unlike vanilla approach.

In this task, we will be implementing LSTM with Attention.

<p align="center"><img src="https://miro.medium.com/max/1400/1*YM4T-QSJIIPQUlMOO_gnzw.png" width="75%" align="center"></p>

Please refer to lecture for more details.

You will be extending the LSTM model and incorporate attention on top of it.

We will then classify the Clickbait and Web of science dataset for this task.

