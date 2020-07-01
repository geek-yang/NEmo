# NEmo :fish:
**Neural network for Emotion detection**, in short as **NEmo**, is a python library designed to implement Bayesian deep learning algorisms to emotion database for human emotion detection. It offers an easy construction of two types of deep neural network to incorporate the eletronic signal from sensor and predict human emotion, like valance and arousal.<br/>

The neural networks adopted by this library are:
* Bayesian Convolutional Long-Short Term Memory neural network approximated by Bernoulli distribution (BBConvLSTM)
* Convolutional Long-Short Term Memory neural network (ConvLSTM)

These two neural networks are good at dealing with spatial temporal sequence tasks and therefore very useful for emotion detection with a strategy of time series folding. Moreover, BBConvLSTM can address uncertainty in the input data and model parameter and therefore it is facilitated with L2 regularization and it could distill more information with data from different tests. <br/>

The library is built ontop of pytorch. <br/> 

Reference:<br/> 
Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning (pp. 1050-1059). <br/>
Xingjian, S. H. I., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems (pp. 802-810). <br/>

