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
//        self.testConv2d()
//        self.testFC()
        self.testPooling()
        
    }
    
    func testConv2d() {
//        Conv2dTests.test_1x2x2x3_s1p0_w(device: self.device, commandQueue: self.commandQueue)
//        Conv2dTests.test_1x2x2x3_s1p0(device: self.device, commandQueue: self.commandQueue)
//        Conv2dTests.test_1x4x4x3_s1p0(device: self.device, commandQueue: self.commandQueue)
//        Conv2dTests.test_1x4x4x3_s1p1(device: self.device, commandQueue: self.commandQueue)
        Conv2dTests.test_32x32x3_s1p1(device: self.device, commandQueue: self.commandQueue)
    }
    func testFC(){
        FCTests.test_18x8(device: device, commandQueue: commandQueue)
    }
    func testPooling(){
//        PoolingTests.test_12x12_2x2_maxpooling(device: device, commandQueue: commandQueue)
        PoolingTests.test_32x32x16_2x2_maxpooling(device: device, commandQueue: commandQueue)
        
    }

}

