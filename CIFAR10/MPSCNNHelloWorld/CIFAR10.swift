//
//  CIFAR10.swift
//  MPSCNNHelloWorld
//
//  Created by Tao Xu on 3/26/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class CIFAR10 {
    
    var device: MTLDevice
    var conv1, conv2, conv3: Conv2d
    var fc1,fc2: FC
    var mp: MaxPooling
    var relu: MPSCNNNeuronReLU
    var softmax: MPSCNNSoftMax
    
    let c1id = MPSImageDescriptor(channelFormat: .float16, width: 32, height: 32, featureChannels: 16)
    let p1id = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 16)
    let c2id = MPSImageDescriptor(channelFormat: .float16, width: 16, height: 16, featureChannels: 32)
    let p2id = MPSImageDescriptor(channelFormat: .float16, width: 8, height: 8, featureChannels: 32)
    let c3id = MPSImageDescriptor(channelFormat: .float16, width: 8, height: 8, featureChannels: 64)
    let p3id = MPSImageDescriptor(channelFormat: .float16, width: 4, height: 4, featureChannels: 64)
    let fc1d = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 128)
    let fc2d = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 10)
    
    init(device: MTLDevice) {
        self.device = device
        relu    = MPSCNNNeuronReLU(device: device, a: 0)
        conv1 = Conv2d(device: device, dataSource: Conv2dDataSource(
            device: device,
            name: "conv1",
            kernel: (3,3),
            inputFeatureChannels: 3,
            outputFeatureChannels: 16,
            activation: relu,
            padding: (1,1),
            bias: true
        ))
        conv2 = Conv2d(device: device, dataSource: Conv2dDataSource(
            device: device,
            name: "conv2",
            kernel: (3,3),
            inputFeatureChannels: 16,
            outputFeatureChannels: 32,
            activation: relu,
            padding: (1,1),
            bias: true
        ))
        conv3 = Conv2d(device: device, dataSource: Conv2dDataSource(
            device: device,
            name: "conv3",
            kernel: (3,3),
            inputFeatureChannels: 32,
            outputFeatureChannels: 64,
            activation: relu,
            padding: (1,1),
            bias: true
        ))
        fc1 = FC(device: device, dataSource: FCDataSource(
            device: device,
            name: "fc1",
            inputShape: (4,4),
            inputFeatureChannels: 64,
            outputFeatureChannels: 128,
            useBias: true,
            activation: relu
        ))
        fc2 = FC(device: device, dataSource: FCDataSource(
            device: device,
            name: "fc2",
            inputShape: (1,1),
            inputFeatureChannels: 128,
            outputFeatureChannels: 10,
            useBias: true,
            activation: relu
        ))
        mp = MaxPooling(device: device, kernel: (2,2), stride: (2,2))
        softmax = MPSCNNSoftMax(device: device)
    }
    func forward(commandBuffer: MTLCommandBuffer,input: MPSImage) -> [Float] {
        //conv
        let conv1_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: c1id)
        conv1.kernel.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: conv1_out)
        //pooling
        let p1_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: p1id)
        mp.kernel.encode(commandBuffer: commandBuffer, sourceImage: conv1_out, destinationImage: p1_out)
        //conv
        let conv2_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: c2id)
        conv2.kernel.encode(commandBuffer:commandBuffer, sourceImage: p1_out, destinationImage: conv2_out)
        //pooling
        let p2_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: p2id)
        mp.kernel.encode(commandBuffer: commandBuffer, sourceImage: conv2_out, destinationImage: p2_out)
        //conv
        let conv3_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: c3id)
        conv3.kernel.encode(commandBuffer: commandBuffer, sourceImage: p2_out, destinationImage: conv3_out)
        //pooling
        let p3_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: p3id)
        mp.kernel.encode(commandBuffer: commandBuffer, sourceImage: conv3_out, destinationImage: p3_out)
        //fc1
        let fc1_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc1d)
        fc1.kernel.encode(commandBuffer: commandBuffer, sourceImage: p3_out, destinationImage: fc1_out)
        //fc2
        let fc2_out = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: fc2d)
        fc2.kernel.encode(commandBuffer: commandBuffer, sourceImage: fc1_out, destinationImage: fc2_out)
        //softmax
        let output: MPSImage = MPSImage(device: device, imageDescriptor: fc2d)
        softmax.encode(commandBuffer: commandBuffer, sourceImage: fc2_out, destinationImage: output)
        
        //commit
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
//        print(fc2_out.toFloatArray().count)
//        print(fc2_out.toFloatArray())
        return output.toFloatArray()
    }
}
