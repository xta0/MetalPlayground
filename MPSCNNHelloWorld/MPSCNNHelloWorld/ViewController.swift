//
//  ViewController.swift
//  MPSCNNHelloWorld
//
//  Created by Tao Xu on 2/27/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import UIKit
import MetalPerformanceShaders

class ViewController: UIViewController {
    
    var commandQueue: MTLCommandQueue!
    var device: MTLDevice!

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        // Load default device.
        device = MTLCreateSystemDefaultDevice()
        // Make sure the current device supports MetalPerformanceShaders.
        guard MPSSupportsMTLDevice(device) else {
            print("Metal Performance Shaders not Supported on current Device")
            return
        }
        
        // Create new command queue.
        commandQueue = device!.makeCommandQueue()
        
    }
    


}

