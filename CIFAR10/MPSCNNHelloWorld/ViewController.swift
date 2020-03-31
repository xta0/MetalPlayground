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
    var srcImage     : MPSImage!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = .white
        
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
//        srcImage = getInputImage(name: "0_6")
        srcImage = getInputImage(name: "9_3")
        commandQueue = device.makeCommandQueue()
        createNeuralNetwork {
            let result = self.network.forward(commandBuffer: self.commandQueue.makeCommandBuffer()!, input: self.srcImage)
            let (index, score) = result.argmax()
            print("\(index),\(score)")
            
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

