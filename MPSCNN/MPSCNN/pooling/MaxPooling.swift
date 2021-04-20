//
//  MaxPooling.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/25/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class MaxPooling {
    var pool: MPSCNNPoolingMax
    var kernel:(Int,Int)
    var stride:(Int,Int)
    init(device: MTLDevice, kernel:(Int, Int), stride:(Int, Int)) {
        self.kernel = kernel
        self.stride = stride
        self.pool = MPSCNNPoolingMax(device: device,
                                     kernelWidth: kernel.0,
                                     kernelHeight: kernel.1,
                                     strideInPixelsX: stride.0,
                                     strideInPixelsY: stride.1)
        self.pool.edgeMode = .clamp
        self.pool.offset = MPSOffset(x: kernel.0/2, y: kernel.1/2, z: 0)
//        self.pool.offset = offsetForPooling()
        self.pool.destinationFeatureChannelOffset = 0
    }
    func run(input: MPSImage, output: MPSImage, commandQueue: MTLCommandQueue) -> [Float] {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return  []
        }
        self.pool.encode(commandBuffer: commandBuffer, sourceImage: input, destinationImage: output)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return output.toFloatArray()
    }
    
//    func offsetForPooling() -> MPSOffset {
//        var offset = MPSOffset(x: 0, y: 0, z: 0)
//        let (kernelWidth, kernelHeight) = self.kernel
//        let (strideInPixelsX, strideInPixelsY) = self.stride
//        let (sourceWidth, sourceHeight) = self.source
//        if kernelWidth % 2 == 0 {
//            offset.x += (((sourceWidth - 1) % strideInPixelsX) / 2) + 1
//        } else {
//            offset.x += (((sourceWidth - 1) % strideInPixelsX) + 1) / 2
//        }
//        if kernelHeight % 2 == 0 {
//            offset.y += (((sourceHeight - 1) % strideInPixelsY) / 2) + 1
//        } else {
//            offset.y += (((sourceHeight - 1) % strideInPixelsY) + 1) / 2
//        }
//        return offset
//    }
    
}
