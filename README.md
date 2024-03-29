# complementary-label-learning

This code gives the implementation of the paper ''Discriminative Complementary-Label Learning with Weighted Loss''.


Requirements

-Python 3.6

-PyTorch 1.1

or using Colab to implement

demo.py

This is main function. After running the code, you should see a text file with the results saved in the same directory. 
The results will have four columns: epoch number, training accuracy, test accuracy, train loss.


python demo.py --me  \<method name\> --mo \<model name\>

  
Methods and models
  
In demo.py, specify the method argument to choose one of the 2 methods available:
  
-w_loss: L-W risk estimator is defined by Equation(8) in the paper
  
-non_k_softmax: L-UW loss is defined by Equation(7) in the paper
  
Specify the model argument:
  
-linear: linear model
  
-MLP: multi-layer perceptron with one hidden layer (500 units)

## citation
<code data-enlighter-language="raw" class="EnlighterJSRAW">@inproceedings{DBLP:conf/icml/GaoZ21,
  author = {Yi Gao and Min{-}Ling Zhang},
  title = {Discriminative Complementary-Label Learning with Weighted Loss},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning, {ICML} 2021, 18-24 July 2021, Virtual Event},
  volume = {139},
  year = {2021}
  }</code>
