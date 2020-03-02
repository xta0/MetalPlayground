//
//  MNIST_Classifier.swift
//  MPSCNNHelloWorld
//
//  Created by Tao Xu on 2/28/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

/*
 # Construct NN
 input = [28x28] grayscale
 class Classifier(nn.Module):
 def __init__(self):
 super().__init__()
 self.fc1 = nn.Linear(784, 256)
 self.fc2 = nn.Linear(256, 10)
 # Dropout module with 0.2 drop probability
 self.dropout = nn.Dropout(p=0.2)
 
 def forward(self, x):
 # make sure input tensor is flattened
 x = x.view(x.shape[0], -1)
 # Now with dropout
 x = self.dropout(F.relu(self.fc1(x)))
 # output so no dropout here
 x = F.log_softmax(self.fc2(x), dim=1)
 
 return x
 */
import Foundation
import MetalPerformanceShaders

class MNIST: NeuralNetwork {
    typealias PredictionType = (label: String, probability: Float)
    
    var fc1,fc2      : MPSCNNFullyConnected
    var relu         : MPSCNNNeuronReLU
    var softmax      : MPSCNNSoftMax

    let fc1id = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 256)
    let fc2id = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 10) //output
    
    var outputImg: MPSImage
    var inputImg:  MPSImage!
    
    init(device: MTLDevice, inflightBuffers: Int)  {
        weightsLoader   = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_W", ext: "txt")}
        biasLoader      = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_b", ext: "txt")}
        
        relu            = MPSCNNNeuronReLU(device: device, a: 0)
        fc1             = dense(device: device, fanIn: 784, fanOut: 256, activation: relu, name: "fc1")
        fc2             = dense(device: device, fanIn: 256, fanOut: 10, activation: nil, name: "fc2")
        softmax         = MPSCNNSoftMax(device: device)
        outputImg       = MPSImage(device: device, imageDescriptor: fc2id)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture, inflightIndex: Int) {
        
        //fc1
        let fc1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1id)
        fc1.encode(commandBuffer: commandBuffer, sourceImage: inputImg, destinationImage: fc1Img)

        //fc2
        let fc2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc2id)
        fc2.encode(commandBuffer: commandBuffer, sourceImage: fc1Img, destinationImage: fc2Img)

        //softmax
        softmax.encode(commandBuffer: commandBuffer, sourceImage: fc2Img, destinationImage: outputImg)
    }
    
    func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<MNIST.PredictionType> {
        let results = outputImg.toFloatArray()
        let (maxIndex, maxValue) = results.argmax()
        var result = NeuralNetworkResult<MNIST.PredictionType>()
        result.predictions.append((label: "\(maxIndex)", probability: maxValue))
        return result
    }
}
