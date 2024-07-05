# gesture_recognition
	Task Description: We aim to train models using a video training dataset to predict or classify actions performed in the videos.

	Model Types: We will build and train two different types of models using Neural Networks:

o	3D Convolution Models (3D CNN): These models process video frames using 3D convolutions.
o	CNN + RNN Architecture: In this approach, we pass video frames through a Convolutional Neural Network (CNN) to extract feature vectors for each frame. Then, we feed the sequence of these feature vectors into a Recurrent Neural Network (RNN).
	Feature Extraction:
o	For the CNN + RNN architecture, the Conv2D network extracts feature vectors from each video frame.
o	These feature vectors are then passed to an RNN-based network.
o	The RNN output is a regular softmax.
	Transfer Learning:
o	Instead of training our own CNN from scratch, we can use transfer learning.
o	Popular pre-trained models like VGG or MobileNet can be used as the base architecture for the 2D CNN layer.
	Model Objective: Our goal is to train a model with the highest accuracy while minimizing the number of parameters.

Experiment Number	Model	Result 	Decision + Explanation	Trainable Parameter
1 
(model) 	Conv3D	Throws Generator error	Crop the images correctly, try to overfit on less amount of data	
2 
(model) 	Conv3D	Training Accuracy : 0.99 
Validation Accuracy : 0.72 
(Best weight Accuracy, Epoch: 17/50) 	Change the Architecture of the model as 
It is Overfitting 	3,237,125
3 
(model1) 	Conv3D	Training Accuracy : 0.86 
Validation Accuracy : 0.75 
(Best weight Accuracy, Epoch: 40/50) 
(Optimizer: SGD) 	This model is able to generalize better than the previous model, but the accuracy is still low. Trying to get good accuracy. 
	30,152,677
4 
(model2) 	Conv3D	Training Accuracy : 0.62 
Validation Accuracy : 0.62 
(Best weight Accuracy, Epoch: 34/50 ) 
(Optimizer: Adam) 	Model is generalizing but the model accuracy has gone further down on changing the optimizer and learning rate. 
Increasing the Learning Rate, to reach the global minima. 	30,152,677
5 
(model3) 	Conv3D	Training Accuracy : 0.54 
Validation Accuracy : 0.67 
(Best weight Accuracy, Epoch: 48/50 ) 
(Optimizer: Adam) 	Reduced filters in the Dense layer, but did not see much performance improvement. 
	15,222,501
6 
(model4) 	Conv3D	Training Accuracy : 0.89 
Validation Accuracy : 0.80 
(Best weight Accuracy, Epoch: 46/50 ) 
(Optimizer: Adam) 	Reduced Batch size and learning rate, which increased the accuracy as well as the generalizability of the model, also have comparatively smaller number of parameters. 
Best model with the Conv-3D approach. 	3,057,381
7 (model5)	LSTM 

base model with 20 frames, 30 epochs, 64 batch size and 120x120 image size
	Max. Training Accuracy 1.0

Max. Validation Accuracy 1.0	We notice wide fluctuations in validation-accuracy while train accuracy reaches 1.0 very early
This is a sign of overfitting.
	4,587,845
8 (model6)	LSTM

LSTM 2nd model with 20 frames, reduced epochs to 20 from 30 and reduced batch size to 50 from 64	Max. Training Accuracy 0.92
Max. Validation Accuracy 0.88

We see a consistent pattern in accuracy improvement in both train and validation set. This is a more reliable model.

We achieved Validation accuracy of 88% and Train accuracy of 91% at Epoch 18. model-00018-0.26692-0.90659-0.31584-0.88000.h5 is the best model so far.	Small batch sizes often provide a regularizing effect, as the model updates its weights more frequently with noisier gradient estimates. Fewer epochs reduce the risk of overfitting
	4,587,845
9 (model7)	GRU 

Start with a base GRU model with same hyper parameters as LSTM 

frames = 20
epochs = 20
batch_size = 50
Image size = 120x120
GRU units = 256
Dropout Rate = 0.2
 	With GRU trainable param get reduced to 1 million from 4 million params in LSTM
	We now want to get the same level of accuracy with reduced trainable parameters.	1,053,701

10 (model8)	GRU 

Increase the number of frames to see if more data adds to model improvement:
 
frames = 25
epochs = 20
batch size = 50
Image size = 120x120
GRU units = 256
Dropout Rate = 0.2
	Validation Accuracy 0.83

We notice only a minor improvement in validation accuracy while train accuracy is 100%. This is possibly an overfit.
	Increasing number of frames did not result in a better validation accuracy instead we got an overfitted model. Change frame size back to 20	1,053,701

11 (model9)	GRU 

Let us try reducing the image size to 100x100

Frames = 20
Epochs = 20
batch size = 50
Image size = 100x100
GRU units = 256
Dropout Rate = 0.2	We got a significant reduction in validation accuracy with maximum value of only 69%
	Reducing image size is resulting is critical data loss for the model. Change image size back to 120x120
	1,053,701

12 (model10)	GRU 

Try with reduced batch size of 30

Frames = 20
Epochs = 20
batch size = 30
Image size = 120x120
GRU units = 256
Dropout Rate = 0.2	We get Training Accuracy of 86% and Validation Accuracy of 77%.	This seems reasonable trade-off between training and validation accuracies. Let us focus on reducing trainable parameters.
	1,053,701

13 (model11)	GRU 

Let us try to reduce the trainable params by reducing GRU and Dense layer units to 128

Frames = 20
Epochs = 20
batch size = 30
Image size = 120x120
GRU units = 128
Dropout Rate = 0.2	We see that the trainable params reduced to 400k from 1 million but the model is again overfitting with Training accuracy of 100%

Validation Accuracy is at 79%	We should now try to reduce overfitting while keeping the same Units for GRU and Dense layers so our Trainable parameters do not increase. 	462,341
14 (model12)	GRU

Let us use regularization in GRU and dense layers and increase dropout rate to 0.5 to reduce overfitting.

Frames = 20
Epochs = 20
batch size = 30
Image size = 120x120
GRU [with regularizer l2 0.001 ] units = 128
Dropout Rate = 0.5
	We get Validation accuracy of 80% with Training accuracy of 86%. 	This is a best GRU model we have and with only 400k trainable parameters. However, it lags behind LSMT (model6) which had 88% validation accuracy.

Let us see if we can further reduce the trainable parameters.
	462,341
15 (model13)	GRU 

Reduce GRU and Dense Units to 64

Frames = 20
Epochs = 20
batch size = 30
Image size = 120x120
GRU [with regularizer l2 0.001 ] units = 64
Dropout Rate = 0.5
	We see that trainable params got reduced to 200k from 400k but both training and validation accuracies have gone down to 70s. 	The accuracy trends however, look to be moving in the right direction.
Let's increase epochs to see if we get a better accuracy score at higher epochs	215,813

16 (model14)	GRU

Increased Epochs to 30 

Frames = 20
Epochs = 30
batch size = 30
Image size = 120x120
GRU [with regularizer l2 0.001 ] units = 64
Dropout Rate = 0.5
	On increasing epochs to 30 we do not see any significant improvement in validation accuracy. In fact Validation accuracy seems to be plateauing.
	We conclude that LSTM model6 is the most accurate model we have with 88% Validation accuracy.
However, It uses 4m trainable parameters.

While GRU model12 gets 80% accuracy with only 400k parameters.

We select GRU model12 as our final model. 	215,813



