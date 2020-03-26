//
//  FCTests.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/15/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

enum FCTests {
    static func test_18x8(device: MTLDevice, commandQueue: MTLCommandQueue) {
        let inputBuffer = MPSUtils.loadInput(name: "18_8_X")
        let inputImage: MPSImage! = MPSImage(device: device,
                                             numberOfImages: 1,
                                             width: 1,
                                             height: 1,
                                             featureChannels: 18,
                                             array: inputBuffer!,
                                             count: 18)
        let outputImage = MPSImageWrapper(device: device,
                                          n: 1,
                                          c: 8,
                                          h: 1,
                                          w: 1).image
        let ds = FCDataSource(device: device,
                              name: "18_8",
                              inputShape: (1,1),
                              inputFeatureChannels: 18,
                              outputFeatureChannels: 8,
                              useBias: true)

        let fc = FC(device: device, dataSource: ds)
        let result = fc.run(input: inputImage, output: outputImage!, commandQueue: commandQueue)
        print(result)
        print(result.count)
    }
}
