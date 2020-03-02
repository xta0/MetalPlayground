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

extension String  {
    var isNumber: Bool {
        return !isEmpty && rangeOfCharacter(from: CharacterSet.decimalDigits.inverted) == nil
    }
}

class ViewController: UIViewController {
    var commandQueue : MTLCommandQueue!
    var device       : MTLDevice!
    var runner       : Runner!
    var network      : MNIST!
    var startupGroup = DispatchGroup()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.backgroundColor = .white
        
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
        
        commandQueue = device.makeCommandQueue()
        startupGroup.enter()
        createNeuralNetwork {
            self.startupGroup.leave()
        }
        startupGroup.notify(queue: .main) {
            self.predict()
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
    func predict() {
        // Since we want to run in "realtime", every call to predict() results in
        // a UI update on the main thread. It would be a waste to make the neural
        // network do work and then immediately throw those results away, so the
        // network should not be called more often than the UI thread can handle.
        // It is up to VideoCapture to throttle how often the neural network runs.
        let (tests, labels) = filesInBundle()
        var cn = 0
        for i in 0..<tests.count {
            let label = labels[i]
            let fileName = tests[i]
            network.inputImg = getInputImage(name: fileName)!
            runner.predict(network: network, texture: network.inputImg.texture, queue: .main) { results in
                let (clz, _ ) = results.predictions[0]
                if clz == label {
                    print("True: ( output:\(clz), label:\(label) )")
                    cn += 1
                } else {
                    print("False: ( output:\(clz), label:\(label) )")
                }
                if i == tests.count - 1 {
                    print("Accuracy: \(String(format:"%.3f", Float(cn)/Float(tests.count)))")
                }
            }
        }
    }
    
    func getInputImage(name:String) -> MPSImage? {
        guard let path = Bundle.main.path(forResource: name, ofType: "") else {
            return nil
        }
        if let content = try? String(contentsOfFile:path, encoding: String.Encoding.utf8) {
            let strs = content.components(separatedBy: ",")
            var nums: [Float] = strs.map { str -> Float in
                let trimmed = str.replacingOccurrences(of: "\n", with: "")
                return Float(trimmed)!
            }
            return MPSImage(device: device,
                            numberOfImages: 1,
                            width: 1,
                            height: 1,
                            featureChannels: 784,
                            array: &nums,
                            count: 28*28 )
            
        }
        return nil;
    }
    
    func filesInBundle() -> ([String], [String]) {
        let urls = Bundle.main.paths(forResourcesOfType: "txt", inDirectory: "")
        var tests:[String] = []
        var labels:[String] = []
        for path in urls {
            if path.hasSuffix(".txt") {
                let fileName = (path as NSString).lastPathComponent
                let tmp = fileName.components(separatedBy: "_")
                let name = tmp[0]
                let label = tmp[1].components(separatedBy: ".")[0]
                if name.isNumber {
                    tests.append(fileName)
                    labels.append(label)
                }
            }
        }
        return (tests,labels)
    }
}

