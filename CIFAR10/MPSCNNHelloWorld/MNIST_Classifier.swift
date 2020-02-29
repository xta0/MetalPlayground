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

class MNIST_Classifier {
    var commandQueue : MTLCommandQueue!
    var device       : MTLDevice
    var fc1,fc2      : MPSCNN_FC
    var softmax      : MPSCNNLogSoftMax
    
    let fc1id = MPSImageDescriptor(channelFormat: .float16, width: 25, height: 25, featureChannels: 1)
    let fc2id = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 10)
    
    init(withCommandQueue commandQueueIn: MTLCommandQueue!) {
        device          = commandQueueIn.device
        commandQueue    = device.makeCommandQueue()
        
        softmax         = MPSCNNLogSoftMax(device: device)
        fc1             = MPSCNN_FC(kernelWidth: 1,
                                    kernelHeight: 1,
                                    inputFeatureChannels: 784,
                                    outputFeatureChannels: 256,
                                    device: device ,
                                    kernelParamsBinaryName: "mnist_fc1")
        fc2             = MPSCNN_FC(kernelWidth: 1,
                                    kernelHeight: 1,
                                    inputFeatureChannels: 256,
                                    outputFeatureChannels: 10,
                                    device: device ,
                                    kernelParamsBinaryName: "mnist_fc2")
    }
    func forward(input: MPSImage) {
        autoreleasepool {
            let commandBuffer = commandQueue.makeCommandBuffer()
            
            let fc1Img = MPSTemporaryImage(commandBuffer: commandBuffer!, imageDescriptor: fc1id)
            fc1.encode(commandBuffer: commandBuffer!, sourceImage: input, destinationImage: fc1Img)
            
            let fc2Img = MPSTemporaryImage(commandBuffer: commandBuffer!, imageDescriptor: fc2id)
            fc2.encode(commandBuffer: commandBuffer!, sourceImage: fc1Img, destinationImage: fc2Img)
            
//            softmax.encode(commandBuffer: commandBuffer!, sourceImage: fc2Img, destinationImage: <#T##MPSImage#>)
            
            commandBuffer?.addCompletedHandler { commandBuffer in
                
            }
            commandBuffer?.commit()
            commandBuffer?.waitUntilCompleted()
        }
    }
    
    
}
