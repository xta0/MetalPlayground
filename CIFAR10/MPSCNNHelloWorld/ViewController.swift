//
//  ViewController.swift
//  MPSCNNHelloWorld
//
//  Created by Tao Xu on 2/27/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import UIKit
import MetalPerformanceShaders

@available(iOS 11.3, *)
class ViewController: UIViewController {
    var commandQueue : MTLCommandQueue!
    var device       : MTLDevice!
    var network      : CIFAR10!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = .white
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
//        let inputs = ["0_6","1_9","2_9","3_4","4_1","5_1","6_2","7_7","8_8","9_3"]
        let inputs = ["1_9"]
        commandQueue = device.makeCommandQueue()
        print("waiting...")
        let deadlineTime = DispatchTime.now() + .seconds(5)
        DispatchQueue.main.asyncAfter(deadline: deadlineTime) {
            print("running...")
            self.createNeuralNetwork {
                for input in inputs {
                    let srcImage = self.getInputImage(name: input)!
                    let capManager = MTLCaptureManager.shared()
                    capManager.startCapture(commandQueue: self.commandQueue)
                    let result = self.network.forward(commandBuffer: self.commandQueue.makeCommandBuffer()!, input: srcImage)
                    let (index, score) = result.argmax()
                    print("\(index),\(score)")
                    capManager.stopCapture()
//                    sleep(2)
                }
            }
        }
    }
    
    func createNeuralNetwork(completion: @escaping () -> Void) {
        // Make sure the current device supports MetalPerformanceShaders.
        guard MPSSupportsMTLDevice(device) else {
            print("Error: this device does not support Metal Performance Shaders")
            return
        }
        // Because it may take a few seconds to load the network's parameters,
        // perform the construction of the neural network in the background.
        DispatchQueue.global().async {
            self.network = CIFAR10(device: self.device)
            DispatchQueue.main.async(execute: completion)
        }
    }
    func getInputImage(name:String) -> MPSImage? {
        let buffer = MPSUtils.loadInput(name: name)
        return MPSImage(device: device,
                        numberOfImages: 1,
                        width: 32,
                        height: 32,
                        featureChannels: 3,
                        array: buffer!,
                        count: 32*32*4 )
    }
    
    
}

