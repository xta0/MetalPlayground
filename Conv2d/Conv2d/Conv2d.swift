//
//  Conv2d.swift
//  Conv2d
//
//  Created by Tao Xu on 3/5/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class Conv2d : NeuralNetwork {
    typealias PredictionType = Float16
    
    var inputImg: MPSImage!
    var outputImg: MPSImage!
    var oid = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 2)
    var conv2d: MPSCNNConvolution
    
    init(device: MTLDevice, inflightBuffers: Int) {
        weightsLoader   = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_W", ext: "txt") }
//        biasLoader      = { name, count in ParameterLoaderBundle2(name: name, count: count, suffix: "_b", ext: "txt") }
        outputImg       = MPSImage(device: device, imageDescriptor: oid)
        conv2d          = convolution(device: device, kernel: (2, 2), inChannels: 3, outChannels: 2, activation: nil, name: "conv", useBias: false)
    }
    
    func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture, inflightIndex: Int) {
        conv2d.encode(commandBuffer: commandBuffer, sourceImage: inputImg, destinationImage: outputImg)
    }
    func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Float16> {
        let probabilities = outputImg.toFloatArray()
        print(probabilities)
        return NeuralNetworkResult<Float16>()
    }
}
