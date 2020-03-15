//
//  Conv2dInputs.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/14/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

enum MPSUtils {
    static func loadData(name: String, ext: String) -> (UnsafeMutableRawPointer?, Int)  {
        guard let path = Bundle.main.path(forResource: name, ofType: ext) else {
            print("Error: resource \"\(name)\" not found")
            return (nil,0)
        }
        guard let content = try? String(contentsOfFile:path, encoding: String.Encoding.utf8) else {
            return (nil,0)
        }
        let strs = content.components(separatedBy: ",")
        var nums: [Float] = strs.map { str -> Float in
            let trimmed = str.replacingOccurrences(of: "\n", with: "")
            return Float(trimmed)!
        }
        let fileSize = nums.count * MemoryLayout<Float>.size
        guard let buffer = malloc(fileSize)else {
            return (nil,0)
        }
        memcpy(buffer, &nums, fileSize)
        return (buffer, nums.count)
    }
    static func offsetForConvolution(_ kernel: (Int, Int), _ padding: (Int, Int)) -> MPSOffset {
        // To set the offset, we can just match the top-left pixel (in the input
        // image, with negative values for padding) that we look at. For 3x3s1p1, we
        // look at the (-1, -1) pixel in the original impl. For 3x3s1p0, we look at
        // (0, 0) pixel. For 3x3s1p2, look at (-2, -2)
        // MPSCNN always looks at (-floor(kernel_size - 1 / 2), -floor(kernel_size - 1 / 2)) Thus, we just
        // need to match this up.
        
        // For 3x3s1p1, offset should be (0, 0)
        // For 3x3s1p0, offset should be (1, 1)
        // For 3x3s1p2, offset should be (-1, -1)
        let mps_offsetX: Int  = kernel.0 / 2;
        let mps_offsetY: Int  = kernel.1 / 2;
        let paddingX = padding.0;
        let paddingY = padding.1;
        return MPSOffset(
            x: mps_offsetX - paddingX,
            y: mps_offsetY - paddingY,
            z: 0)
    }
    static func loadInput(name: String) -> UnsafeMutablePointer<Float>? {
        let (rawPointer, count) = MPSUtils.loadData(name: name, ext: "txt")
        return rawPointer?.bindMemory(to: Float.self, capacity: count)
    }
    static func loadInput2(name: String) -> [Float] {
        let (rawPointer, count) = MPSUtils.loadData(name: name, ext: "txt")
        let floatPtr = rawPointer?.bindMemory(to: Float.self, capacity: count)
        var result: [Float] = []
        for i in 0..<count {
            result.append(floatPtr![i])
        }
        return result
    }
}

struct MPSImageWrapper {
    var numberOfImages: Int
    var channels: Int
    var height: Int
    var width: Int
    var device: MTLDevice
    var image: MPSImage?
    var desc: MPSImageDescriptor
    
    init(device: MTLDevice, n: Int, c: Int, h: Int, w:Int) {
        self.device = device
        self.numberOfImages = n
        self.channels = c
        self.height = h
        self.width = w
//        self.desc = MPSImageDescriptor(channelFormat: .float16, width: w, height: h, featureChannels: c)
        self.desc = MPSImageDescriptor(channelFormat: .float16, width: w, height: h, featureChannels: c, numberOfImages: n, usage:  [.shaderRead, .shaderWrite])
        self.image = MPSImage(device: device, imageDescriptor: desc)

    }
}
