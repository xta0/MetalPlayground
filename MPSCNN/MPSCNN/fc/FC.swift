//
//  FC.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/15/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class FC {
    var device: MTLDevice
    var fc: MPSCNNFullyConnected
    var dataSource: FCDataSource
    
    init(device: MTLDevice, dataSource:FCDataSource) {
        self.device          = device
        self.dataSource      = dataSource
        self.fc              = MPSCNNFullyConnected(device: device, weights: dataSource)
        self.fc.clipRect     = MTLRegionMake2D(0, 0, 1, 1)
    }
    func run(input: MPSImage, output: MPSImage, commandQueue: MTLCommandQueue) -> [Float] {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return  []
        }
        fc.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: output)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output.toFloatArray()
    }
}
