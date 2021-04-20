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
    var conv2d: MPSCNNConvolution
    var dataSource: Conv2dDataSource
    
    init(device: MTLDevice, dataSource:Conv2dDataSource) {
        self.device          = device
        self.dataSource      = dataSource
        self.conv2d          = MPSCNNConvolution(device: device, weights: dataSource)
        self.conv2d.offset   = dataSource.offset()
        self.conv2d.edgeMode = .zero
    }
    func run(input: MPSImage, output: MPSImage, commandQueue: MTLCommandQueue) -> [Float] {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return  []
        }
        conv2d.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: output)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output.toFloatArray()
    }
}
