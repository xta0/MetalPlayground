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
    
    init(device: MTLDevice, inflightBuffers: Int) {
        pool    = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2)
        relu    = MPSCNNNeuronReLU(device: device, a: 0)
        softmax = MPSCNNSoftMax(device: device)
        
        weightsLoader = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_W", ext: "txt") }
        biasLoader = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_b", ext: "txt") }
        
        conv1 = convolution(device: device, kernel: (3,3), inChannels: 3, outChannels: 16, activation: relu, name: "conv1")
        conv1.padding = .same
    }
    
    init(withCommandQueue commandQueueIn: MTLCommandQueue!) {
        commandQueue = commandQueueIn
        device = commandQueue.device
        pool = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2)
        relu = MPSCNNNeuronReLU(device: device, a: 0)
        softmax = MPSCNNLogSoftMax(device: device)
        conv1 = MPSCNN_Conv2D(kernelWidth: 3,
                                kernelHeight: 3,
                                inputFeatureChannels: 3,
                                outputFeatureChannels: 16,
                                neuronFilter: relu,
                                device: device,
                                kernelParamsBinaryName: "conv1_w")
        
        conv2 = MPSCNN_Conv2D(kernelWidth: 3,
                                kernelHeight: 3,
                                inputFeatureChannels: 16,
                                outputFeatureChannels: 32,
                                neuronFilter: relu,
                                device: device,
                                kernelParamsBinaryName: "conv2_w")
    
        conv3 = MPSCNN_Conv2D(kernelWidth: 3,
                                kernelHeight: 3,
                                inputFeatureChannels: 32,
                                outputFeatureChannels: 64,
                                neuronFilter: relu,
                                device: device,
                                kernelParamsBinaryName: "conv3_w")
        
        fc1 = MPSCNN_FC(kernelWidth: 1,
                            kernelHeight: 1,
                            inputFeatureChannels: 1024,
                            outputFeatureChannels: 128,
                            device: device,
                            kernelParamsBinaryName: "fc1_w")
        
        fc2 = MPSCNN_FC(kernelWidth: 1,
                            kernelHeight: 1,
                            inputFeatureChannels: 128,
                            outputFeatureChannels: 10,
                            device: device,
                            kernelParamsBinaryName: "fc2_w")
    }
    func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture, inflightIndex: Int) {
        
    }
    func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<CIFAR10_Classifier.PredictionType> {
        var result = NeuralNetworkResult<Prediction>()
        return result
    }
}

