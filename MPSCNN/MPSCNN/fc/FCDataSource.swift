//
//  FCDataSource.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/15/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class FCDataSource:NSObject, MPSCNNConvolutionDataSource {
    let device: MTLDevice
    let name: String
    let inputShape: (Int,Int)
    let inputFeatureChannels: Int
    let outputFeatureChannels: Int
    let useBias: Bool
    let activation: MPSCNNNeuron?
    var weight: UnsafeMutableRawPointer!
    var bias: UnsafeMutablePointer<Float>!
    init(device: MTLDevice,
         name: String,
         inputShape:(Int,Int),
         inputFeatureChannels: Int,
         outputFeatureChannels: Int,
         useBias: Bool = false,
         activation: MPSCNNNeuron? = nil) {
        self.device = device
        self.name = name
        self.inputShape = inputShape
        self.inputFeatureChannels = inputFeatureChannels
        self.outputFeatureChannels = outputFeatureChannels
        self.useBias = useBias
        self.activation = activation
    }
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: inputShape.0,
                                                   kernelHeight: inputShape.1,
                                                   inputFeatureChannels: inputFeatureChannels,
                                                   outputFeatureChannels: outputFeatureChannels,
                                                   neuronFilter: activation)
        convDesc.strideInPixelsX = 1
        convDesc.strideInPixelsY = 1
        convDesc.groups = 1
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
}
