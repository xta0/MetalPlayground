//
//  Renderer.swift
//  MetalRender
//
//  Created by taox on 10/15/19.
//  Copyright © 2019 taox. All rights reserved.
//

import Cocoa
import MetalKit

class Renderer: NSObject {
    static var device: MTLDevice!
    let commandQueue: MTLCommandQueue
    static var library: MTLLibrary!
    let pipelineState: MTLRenderPipelineState
    
    
    init(view: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
            let commandQueue = device.makeCommandQueue() else {
                fatalError()
        }
        Renderer.device = device
        self.commandQueue = commandQueue
        Renderer.library = device.makeDefaultLibrary()
        self.pipelineState = Renderer.createPipelineState()
        super.init()
    }
    
    static func createPipelineState() -> MTLRenderPipelineState {
        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        
        //set pipeline state properties
        //pipeline states are read only, so we use a descriptor to init them.
        //after we initialize the pipeline obejct, the pipeline won't change
        //so that metal can pre-compile them and make a more efficient use of them
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        let vertextFunc = Renderer.library.makeFunction(name: "vertext_main")
        let fragmentFunc = Renderer.library.makeFunction(name: "fragment_main")
        pipelineStateDescriptor.vertexFunction = vertextFunc
        pipelineStateDescriptor.fragmentFunction = fragmentFunc
        return try! Renderer.device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
        
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let drawable = view.currentDrawable,
            //view contains current texture's descriptor
            let descriptor = view.currentRenderPassDescriptor,
            //each commandEncoder contains the GPU commands and controls a single render pass. It needs to be created from the current texture
            let commandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else{
                return
        }
        //send the pipeline state to GPU
        commandEncoder.setRenderPipelineState(pipelineState)
        //draw call
        commandEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: 1)
        commandEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    func draw(in view: MTKView) {
        print("draw")
    }
    
    
}
