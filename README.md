<img src="./images/example_image.png" align="left" height="600" width="350" ></a>


Observations

- Use cross entropy loss function with softmax activation function in the output layer only. It doesn't make much sense otherwise. And also use of logistic in the other layers gives better results.
- Different activation function may require different learning rates. And then also different amount of training epochs.
- If the error over time seems to make spiky oscillations, probably it's stuck on the sides of a minimum and the large learning rate does not allow it to go down to the minimum. Reducing learning rate will probably help, increasing minibatch size might also help (more samples seek for a smoother minimum)

To implement next:
- L1 regularization
- L2 regularization
- Total variation regularization
- Learning rate matrices (i.e. not same learning rate for all the weights in a layer)
- Convolutional layers
- Dropout


Attention: The classification script (deeper_classification.m) combines ReLU+Softmax+Cross-entropy which can cause exploding gradient. The gradient explosion depends on how the parameters are initialized. Therefore, it might explode when running the script which leads to nonsense models. It is left like this for demonstration purposes.
