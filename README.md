**CAPUCHIN BIRD DETECTOR**

This project's main objective is to classify a capuchin birds call from a other noise.
I had used this video for main inspiration for this project, and how to write and structure the code https://www.youtube.com/watch?v=ZLIPkmmDJAc&t=3380s

**Tech stack**

1)numpy

2)tensorflow 

3)librosa

4)matplotlib

**Features**

In the video he used the features from spectorgrams, but I went for MFCC's as I found out that they are better for voice detection.

**Model**

I have used a 2D convolutional neural network to classify , I had used it becuase as my research into what models are better for audio detection and also work well with quantization CNN kept comping up, I have mainly used ReLU as the activation function between all the hidden layers and a sigmoid for my out put layer as its a binary classsfication problem.

**Quantization**

I had found a good article from google describing quantization amnd had quantized the model into a int8 from , using full integer quantization.
