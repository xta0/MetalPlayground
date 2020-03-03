//
//  CIFARDeepCNN.swift
//  MPSCNNHelloWorld
//
//  Created by Tao Xu on 2/27/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

/*
 // model structure
 // input 32x32
 class Classifier(nn.Module):
 def __init__(self):
 super().__init__()
 self.conv1 = nn.Conv2d(3,16,3,padding=1)
 self.conv2 = nn.Conv2d(16,32,3,padding=1)
 self.conv3 = nn.Conv2d(32,64,3,padding=1)
 self.pool  = nn.MaxPool2d(2,2)
 self.dropout = nn.Dropout(0.2)
 self.fc1   = nn.Linear(64*4*4,128)
 self.fc2   = nn.Linear(128,10)
 def forward(self, x):
 x = self.pool(F.relu(self.conv1(x)))
 x = self.pool(F.relu(self.conv2(x)))
 x = self.pool(F.relu(self.conv3(x)))
 #flatten the input
 x = x.view(-1,64*4*4)
 x = self.dropout(x)
 x = F.relu(self.fc1(x))
 x = self.dropout(x)
 x = F.relu(self.fc2(x))
 x = F.log_softmax(x, dim=1)
 return x
 */

// input tensor size:
import Foundation
import MetalPerformanceShaders
import Accelerate

@available(iOS 11.3, *)
class CIFAR10_Classifier : NeuralNetwork {
    typealias PredictionType = (label: String, probability: Float)
    
    let c1id    = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 16)
    let p1id    = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 16)
    let c2id    = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 32)
    let p2id    = MPSImageDescriptor(channelFormat: .float16, width: 8, height: 8, featureChannels: 32)
    let c3id    = MPSImageDescriptor(channelFormat: .float16, width: 8, height: 8, featureChannels: 64)
    let p3id    = MPSImageDescriptor(channelFormat: .float16, width: 4, height: 4, featureChannels: 32)
    let fc1id   = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 128)
    let fc2id   = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 10)
    
    //network structure
    var conv1,conv2,conv3 : MPSCNNConvolution
    var pool: MPSCNNPoolingMax
    var relu: MPSCNNNeuronReLU
    var fc1,fc2: MPSCNNFullyConnected
    var softmax: MPSCNNSoftMax
    
    var outputImg: MPSImage!
    var inputImg: MPSImage!
    
    init(device: MTLDevice, inflightBuffers: Int) {
        pool    = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2)
        relu    = MPSCNNNeuronReLU(device: device, a: 0)
        softmax = MPSCNNSoftMax(device: device)
        
        weightsLoader = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_W", ext: "txt") }
        biasLoader = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_b", ext: "txt") }
        
        conv1 = convolution(device: device, kernel: (3,3), inChannels: 3, outChannels: 16, activation: relu, name: "conv1")
        conv2 = convolution(device: device, kernel: (3,3), inChannels: 3, outChannels: 32, activation: relu, name: "conv2")
        conv3 = convolution(device: device, kernel: (3,3), inChannels: 3, outChannels: 64, activation: relu, name: "conv3")
        pool = maxPooling(device: device, kernel: (2,2), stride: (1,1))
        fc1 = dense(device: device, shape: (4,4), inChannels: 16, fanOut: 128, activation: relu, name: "fc1")
        fc2 = dense(device: device, shape: (1,1), inChannels: 128, fanOut: 10, activation: nil, name: "fc2")
        softmax         = MPSCNNSoftMax(device: device)
        outputImg       = MPSImage(device: device, imageDescriptor: fc2id)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture, inflightIndex: Int) {
        
        let conv1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1id)
        conv1.encode(commandBuffer: commandBuffer, sourceImage: inputImg, destinationImage: conv1Img)
        
        let p1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: p1id)
        pool.encode(commandBuffer: commandBuffer, sourceImage: conv1Img, destinationImage: p1Img)
        
        let conv2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1id)
        conv2.encode(commandBuffer: commandBuffer, sourceImage: p1Img, destinationImage: conv2Img)
        
        let p2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: p2id)
        pool.encode(commandBuffer: commandBuffer, sourceImage: conv2Img, destinationImage: p2Img)
        
        let conv3Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1id)
        conv3.encode(commandBuffer: commandBuffer, sourceImage: p2Img, destinationImage: conv3Img)
        
        let p3Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: p3id)
        pool.encode(commandBuffer: commandBuffer, sourceImage: conv2Img, destinationImage: p3Img)
        
        let fc1Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1id)
        fc1.encode(commandBuffer: commandBuffer, sourceImage: p3Img, destinationImage: fc1Img)
        
        let fc2Img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc2id)
        fc2.encode(commandBuffer: commandBuffer, sourceImage: fc1Img, destinationImage: fc2Img)
        
        softmax.encode(commandBuffer: commandBuffer, sourceImage: fc2Img, destinationImage: outputImg)
        
    }
    func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<CIFAR10_Classifier.PredictionType> {
        let probabilities = outputImg.toFloatArray()
        let (maxIndex, maxValue) = probabilities.argmax()
        var result = NeuralNetworkResult<CIFAR10_Classifier.PredictionType>()
        result.predictions.append((label: "\(maxIndex)", probability: maxValue))
        return result
    }
}

