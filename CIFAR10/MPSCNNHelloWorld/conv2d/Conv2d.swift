//
//  Conv2d.swift
//  Conv2d
//
//  Created by Tao Xu on 3/5/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class Conv2d {
    var device: MTLDevice
    var kernel: MPSCNNConvolution
    var dataSource: Conv2dDataSource
    
    init(device: MTLDevice, dataSource:Conv2dDataSource) {
        self.device          = device
        self.dataSource      = dataSource
        self.kernel          = MPSCNNConvolution(device: device, weights: dataSource)
        self.kernel.offset   = dataSource.offset()
        self.kernel.edgeMode = .zero
    }
    func run(input: MPSImage, output: MPSImage, commandQueue: MTLCommandQueue) -> [Float] {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return  []
        }
        kernel.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: output)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output.toFloatArray()
    }
}
