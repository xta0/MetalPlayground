//
//  Conv2dDataSource.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/14/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class Conv2dDataSource: NSObject, MPSCNNConvolutionDataSource {
    let device: MTLDevice
    let name: String
    let kernel: (Int,Int)
    let stride: (Int,Int)
    let padding: (Int,Int)
    let inputFeatureChannels: Int
    let outputFeatureChannels: Int
    let useBias: Bool
    let activation: MPSCNNNeuron?
    var weight: UnsafeMutableRawPointer!
    var bias: UnsafeMutablePointer<Float>!
    
    init( device: MTLDevice,
          name: String,
          kernel: (Int, Int),
          inputFeatureChannels: Int,
          outputFeatureChannels: Int,
          activation: MPSCNNNeuron? = nil,
          stride:(Int, Int) = (1,1),
          padding: (Int, Int) = (0,0),
          bias: Bool = false
         ) {
        self.device     = device
        self.name       = name
        self.kernel     = kernel
        self.stride     = stride
        self.padding    = padding
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.useBias    = bias
        self.activation = activation
    }
    
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: kernel.0,
                                                   kernelHeight: kernel.1,
                                                   inputFeatureChannels: inputFeatureChannels,
                                                   outputFeatureChannels: outputFeatureChannels,
                                                   neuronFilter: activation)
        convDesc.strideInPixelsX = stride.0
        convDesc.strideInPixelsY = stride.1
        return convDesc
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return weight
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        if ( useBias ) {
            return bias
        }
        return nil
    }
    
    func load() -> Bool {
        let weight = self.name + "_W"
        let bias = self.name + "_b"
        self.weight = MPSUtils.loadData(name: weight, ext: "txt").0!
        if (self.useBias) {
            let (rawPointer,count) = MPSUtils.loadData(name: bias, ext: "txt")
            self.bias = rawPointer!.bindMemory(to: Float.self, capacity: count)
        }
        return true
    }
    func purge() {
        self.weight = nil
        self.bias = nil;
    }
    
    func label() -> String? {
        return name
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        fatalError("copy is not implemented")
    }
    
    func offset() -> MPSOffset {
        return MPSUtils.offsetForConvolution(self.kernel, self.padding)
    }
}
