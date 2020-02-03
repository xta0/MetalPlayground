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
    static var library: MTLLibrary! //used to hold the shader functions
    let pipelineState: MTLRenderPipelineState
    
    let positions: [SIMD4<Float>] = [
      SIMD4<Float>(0.5,0.5,0,1),
      SIMD4<Float>(-0.5,0.5,0,1),
      SIMD4<Float>(0.5,-0.5,0,1),
      SIMD4<Float>(-0.5,-0.5,0,1),
      SIMD4<Float>(-0.5,0.5,0,1),
      SIMD4<Float>(0.5,-0.5,0,1),
    ];

    let colors: [SIMD3<Float>] = [
        SIMD3<Float>(1,0,0),
        SIMD3<Float>(0,1,0),
        SIMD3<Float>(0,0,1),
        SIMD3<Float>(1,0,0),
        SIMD3<Float>(0,1,0),
        SIMD3<Float>(0,0,1),
    ];
    let posistionBuffer: MTLBuffer
    let colorBuffer: MTLBuffer
    var timer:Float = 0.0;
    
    init(view: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
            let commandQueue = device.makeCommandQueue() else {
                fatalError()
        }
        Renderer.device     = device
        Renderer.library    = device.makeDefaultLibrary()
        self.commandQueue   = commandQueue
        self.pipelineState  = Renderer.createPipelineState()
        let positionlength  = MemoryLayout<SIMD4<Float>>.stride * positions.count
        posistionBuffer     = device.makeBuffer(bytes: positions, length: positionlength, options: [])!
        let colorLength     = MemoryLayout<SIMD3<Float>>.stride * colors.count
        colorBuffer         = device.makeBuffer(bytes: colors, length: colorLength, options: [])!
        
        super.init()
    }
    
    static func createPipelineState() -> MTLRenderPipelineState {
        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        
        //set pipeline state properties
        //pipeline states are read only, so we use a descriptor to init them.
        //after we initialize the pipeline obejct, the pipeline won't change
        //so that metal can pre-compile them and make a more efficient use of them
        //This all happens before the main runloop, and only be executed once
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        let vertextFunc = Renderer.library.makeFunction(name: "vertex_main")
        let fragmentFunc = Renderer.library.makeFunction(name: "fragment_main")
        pipelineStateDescriptor.vertexFunction = vertextFunc
        pipelineStateDescriptor.fragmentFunction = fragmentFunc
        return try! Renderer.device.makeRenderPipelineState(descriptor: pipelineStateDescriptor)
        
    }
}

extension Renderer: MTKViewDelegate {
    //will be invoked when view's frame changed
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }
    //will be invoked every frame
    //for each frame, we need a command buffer, and a commendEncoder
    func draw(in view: MTKView) {
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            // represents the texture
            let drawable = view.currentDrawable,
            //view contains current texture's descriptor
            let descriptor = view.currentRenderPassDescriptor,
            //each commandEncoder contains the GPU commands and controls a single render pass. It needs to be created from the current texture
            let commandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else{
                return
        }
        
        //put a small variation in the buffer
        timer += 0.1
        var currentTime = sin(timer)
        commandEncoder.setVertexBytes(&currentTime, length: MemoryLayout<Float>.stride, index: 2)
        //send the pipeline state to GPU
        commandEncoder.setRenderPipelineState(pipelineState)
        
        //set vertex buffer
        commandEncoder.setVertexBuffer(posistionBuffer, offset: 0, index: 0)
        commandEncoder.setVertexBuffer(colorBuffer, offset: 0, index: 1)

        
        //draw call
        //triangle is being drawn unti-clockwise
        commandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        commandEncoder.drawPrimitives(type: .triangle, vertexStart: 3, vertexCount: 3)
        commandEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

//1. create the vertexes in CPU, and send them to GPU through vertex buffer
//2. map the MTLBuffer to shdader using argument tables. Depending on the device and resource type, there are 31 slots for each argument table
