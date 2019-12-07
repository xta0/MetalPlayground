//
//  ViewController.swift
//  MetalRender
//
//  Created by taox on 10/15/19.
//  Copyright © 2019 taox. All rights reserved.
//

import Cocoa
import MetalKit
class ViewController: NSViewController {
    @IBOutlet var metalView: MTKView!
    var renderer: Renderer?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        renderer = Renderer(view: metalView)
        metalView.device = Renderer.device
        metalView.delegate = renderer
        metalView.clearColor = MTLClearColorMake(1.0, 1.0, 0.8, 1.0)
    }

    override var representedObject: Any? {
        didSet {
        // Update the view, if already loaded.
        }
    }


}

