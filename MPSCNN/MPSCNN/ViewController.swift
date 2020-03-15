//
//  ViewController.swift
//  MPSCNN
//
//  Created by Tao Xu on 3/14/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import UIKit
import MetalPerformanceShaders

class ViewController: UIViewController {
    
    var commandQueue : MTLCommandQueue!
    var device       : MTLDevice!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = .white
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
        commandQueue = device.makeCommandQueue()
        
        //run test cases
//        test_1x2x2x3_s1p0_no_bias_no_activation()
//        test_1x2x2x3_s1p0_no_activation()
        test_1x4x4x3_s1p1()
    }
    
    func test_1x2x2x3_s1p0_w() {
        let inputBuffer = MPSUtils.loadInput(name: "1x2x2x3_s1p0_no_bias_no_activation_X")
        let inputImage: MPSImage! = MPSImage(device: device,
                                             numberOfImages: 1,
                                             width: 2,
                                             height: 2,
                                             featureChannels: 3,
                                             array: inputBuffer!,
                                             count: 2*2*4)
        let outputImage = MPSImageWrapper(device: device,
                                          n: 1,
                                          c: 1,
                                          h: 1,
                                          w: 1).image
        let ds = Conv2dDataSource(device: self.device,
                                  name: "1x2x2x3_s1p0_no_bias_no_activation",
                                  kernel: (2,2),
                                  inputFeatureChannels: 3,
                                  outputFeatureChannels: 1)
        let conv2d = Conv2d(device: device, dataSource: ds)
        let result = conv2d.run(input: inputImage, output: outputImage!, commandQueue: commandQueue)
        print(result)
    }
    
    func test_1x2x2x3_s1p0() {
        let inputBuffer = MPSUtils.loadInput(name: "1x2x2x3_s1p0_bias_no_activation_X")
        let inputImage: MPSImage! = MPSImage(device: device,
                                             numberOfImages: 1,
                                             width: 2,
                                             height: 2,
                                             featureChannels: 3,
                                             array: inputBuffer!,
                                             count: 2*2*4)
        let outputImage = MPSImageWrapper(device: device,
                                          n: 1,
                                          c: 2,
                                          h: 1,
                                          w: 1).image
        let ds = Conv2dDataSource(device: self.device,
                                  name: "1x2x2x3_s1p0_bias_no_activation",
                                  kernel: (2,2),
                                  inputFeatureChannels: 3,
                                  outputFeatureChannels: 2,
                                  bias: true)
        let conv2d = Conv2d(device: device, dataSource: ds)
        let result = conv2d.run(input: inputImage, output: outputImage!, commandQueue: commandQueue)
        print(result)
    }
    func test_1x4x4x3_s1p0() {
        let inputBuffer = MPSUtils.loadInput(name: "1x4x4x3_s1p0_X")
        let inputImage: MPSImage! = MPSImage(device: device,
                                             numberOfImages: 1,
                                             width: 4,
                                             height: 4,
                                             featureChannels: 3,
                                             array: inputBuffer!,
                                             count: 4*4*4)
        let outputImage = MPSImageWrapper(device: device,
                                          n: 1,
                                          c: 2,
                                          h: 2,
                                          w: 2).image
        let ds = Conv2dDataSource(device: self.device,
                                  name: "1x4x4x3_s1p0",
                                  kernel: (3,3),
                                  inputFeatureChannels: 3,
                                  outputFeatureChannels: 2,
                                  activation: MPSCNNNeuronSigmoid(device: device),
                                  bias: true)
        let conv2d = Conv2d(device: device, dataSource: ds)
        let result = conv2d.run(input: inputImage, output: outputImage!, commandQueue: commandQueue)
        print(result)
    }
    func test_1x4x4x3_s1p1() {
           let inputBuffer = MPSUtils.loadInput(name: "1x4x4x3_s1p1_X")
           let inputImage: MPSImage! = MPSImage(device: device,
                                                numberOfImages: 1,
                                                width: 4,
                                                height: 4,
                                                featureChannels: 3,
                                                array: inputBuffer!,
                                                count: 4*4*4)
           let outputImage = MPSImageWrapper(device: device,
                                             n: 1,
                                             c: 2,
                                             h: 4,
                                             w: 4).image
           let ds = Conv2dDataSource(device: self.device,
                                     name: "1x4x4x3_s1p1",
                                     kernel: (3,3),
                                     inputFeatureChannels: 3,
                                     outputFeatureChannels: 2,
                                     padding: (1,1),
                                     bias: true)
           let conv2d = Conv2d(device: device, dataSource: ds)
           let result = conv2d.run(input: inputImage, output: outputImage!, commandQueue: commandQueue)
           print(result)
       }
}

