//
//  CIFARDeepCNN.swift
//  MPSCNNHelloWorld
//
//  Created by Tao Xu on 2/27/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

/*
 // model structure
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
class CIFARClassifier {
    
    //metal
    var commandQueue: MTLCommandQueue
    var device: MTLDevice
//    var srcImage: MPSImage
//    var dstImage: MPSImage
    
    let c1id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 32, height: 32, featureChannels: 16)
    let p1id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 16, height: 16, featureChannels: 16)
    let c2id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 16, height: 16, featureChannels: 32)
    let p2id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 8, height: 8, featureChannels: 32)
    let c3id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 8, height: 8, featureChannels: 64)
    let p3id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 4, height: 4, featureChannels: 32)
    let fc1id = MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float16, width: 1, height: 1, featureChannels: 128)
    
    //network structure
    var conv1,conv2,conv3 : MPSCNN_Conv2D
    var pool: MPSCNNPoolingMax
    var relu: MPSCNNNeuronReLU
    var fc1,fc2: MPSCNN_FC
    var softmax: MPSCNNLogSoftMax
    
    //intermediate results
//    var c1Image, c2Image, c3Image, p1Image, p2Image, p3Image, fc1Image: MPSImage
    
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
    
    func forward(inputImage: MPSImage?=nil) -> UInt {
        autoreleasepool {
            let commandBuffer = commandQueue.makeCommandBuffer()
            // output will be stored in this image
//            let finalLayer = MPSImage(device: commandBuffer.device, imageDescriptor: did)
        }
        return 1;
    }
}

