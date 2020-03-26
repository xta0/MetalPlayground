//
//  PoolingTests.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/25/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

enum PoolingTests {
    static func test_12x12_2x2_maxpooling(device: MTLDevice, commandQueue: MTLCommandQueue) {
        let buffer = MPSUtils.loadInput(name: "1144x22")
        let mp = MaxPooling(device: device, source: (4,4), kernel: (2,2), stride: (2,2))
        let input = MPSImage(device: device,
                             numberOfImages: 1,
                             width: 4,
                             height: 4,
                             featureChannels: 3,
                             array: buffer!,
                             count: 4*4*4)
        let output = MPSImageWrapper(device: device, n: 1, c: 3, h: 2, w: 2)
        let result = mp.run(input: input, output: output.image!, commandQueue: commandQueue)
        print(result)
    }
}
