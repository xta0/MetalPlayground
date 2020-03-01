//
//  ViewController.swift
//  MNIST
//
//  Created by taox on 2/29/20.
//  Copyright © 2020 taox. All rights reserved.
//

import UIKit
import MetalPerformanceShaders

let MaxBuffersInFlight = 3   // use triple buffering


class ViewController: UIViewController {
    
    var inputImage   : MPSImage!
    var commandQueue : MTLCommandQueue!
    var device       : MTLDevice!
    var runner       : Runner!
    var network      : MNIST!
    var startupGroup = DispatchGroup()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
        
        commandQueue = device.makeCommandQueue()
        inputImage = getInputImage()!
        startupGroup.enter()
        createNeuralNetwork {
            self.startupGroup.leave()
        }
        startupGroup.notify(queue: .main) {
            self.predict(texture: self.inputImage.texture)
        }
    }
    
    func createNeuralNetwork(completion: @escaping () -> Void) {
        // Make sure the current device supports MetalPerformanceShaders.
        guard MPSSupportsMTLDevice(device) else {
            print("Error: this device does not support Metal Performance Shaders")
            return
        }
        
        runner = Runner(commandQueue: commandQueue, inflightBuffers: MaxBuffersInFlight)
        
        // Because it may take a few seconds to load the network's parameters,
        // perform the construction of the neural network in the background.
        DispatchQueue.global().async {
            
            timeIt("Setting up neural network") {
                self.network = MNIST(device: self.device, inflightBuffers: MaxBuffersInFlight)
            }
            DispatchQueue.main.async(execute: completion)
        }
    }
    func predict(texture: MTLTexture) {
      // Since we want to run in "realtime", every call to predict() results in
      // a UI update on the main thread. It would be a waste to make the neural
      // network do work and then immediately throw those results away, so the
      // network should not be called more often than the UI thread can handle.
      // It is up to VideoCapture to throttle how often the neural network runs.

      runner.predict(network: network, texture: texture, queue: .main) { result in
       
      }
    }
    
    func getInputImage() -> MPSImage? {
        guard let path = Bundle.main.path(forResource: "input", ofType: "txt") else {
            return nil
        }
        if let content = try? String(contentsOfFile:path, encoding: String.Encoding.utf8) {
            let strs = content.components(separatedBy: ",")
            var nums: [Float32] = strs.map { str -> Float32 in
                let trimmed = str.replacingOccurrences(of: "\n", with: "")
                return Float32(trimmed)!
            }
            return MPSImage(device: device,
                            numberOfImages: 1,
                            width: 28,
                            height: 28,
                            featureChannels: 1,
                            array: &nums,
                            count: 28*28 )
            
        }
        return nil;
    }
}

