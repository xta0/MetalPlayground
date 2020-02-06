//
//  Renderer.swift
//  MetalRender
//
//  Created by taox on 10/15/19.
//  Copyright © 2019 taox. All rights reserved.
//

import Cocoa
import MetalKit

struct Vertex {
    let position: SIMD3<Float>
    let color: SIMD3<Float>
}

class Renderer: NSObject {
    static var device: MTLDevice!
    let commandQueue: MTLCommandQueue
    static var library: MTLLibrary! //used to hold the shader functions
    let pipelineState: MTLRenderPipelineState
    

    let verticies: [Vertex] = [
        Vertex(position: SIMD3<Float>(0.5,0.5,0), color: SIMD3<Float>(1,0,0)),
        Vertex(position: SIMD3<Float>(-0.5,0.5,0), color: SIMD3<Float>(0,1,0)),
        Vertex(position: SIMD3<Float>(0.5,-0.5,0), color: SIMD3<Float>(0,0,1)),
        Vertex(position: SIMD3<Float>(-0.5,-0.5,0), color: SIMD3<Float>(1,1,0)),
    ]
    
    let indexArray: [uint16] = [
        0,1,2,
        1,3,2
    ];
        
    let vertexBuffer: MTLBuffer
    let indexBuffer: MTLBuffer
    
    init(view: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
            let commandQueue = device.makeCommandQueue() else {
                fatalError()
        }
        Renderer.device     = device
        Renderer.library    = device.makeDefaultLibrary()
        self.commandQueue   = commandQueue
        self.pipelineState  = Renderer.createPipelineState()
        
        let indexLength     = MemoryLayout<uint16>.stride * indexArray.count
        indexBuffer         = device.makeBuffer(bytes: indexArray, length: indexLength, options: [])!
        
        let vertexLength    = MemoryLayout<Vertex>.stride * verticies.count
        vertexBuffer        = device.makeBuffer(bytes: verticies, length: vertexLength, options: [])!
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
        pipelineStateDescriptor.vertexDescriptor = MTLVertexDescriptor.defaultVertexDescriptor()
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
        
        //send the pipeline state to GPU
        commandEncoder.setRenderPipelineState(pipelineState)
        
        //set vertex buffer
        commandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

        //draw call
        //triangle is being drawn unti-clockwise
        commandEncoder.drawIndexedPrimitives(type: .triangle,
                                             indexCount: indexArray.count,
                                             indexType: .uint16,
                                             indexBuffer: indexBuffer,
                                             indexBufferOffset: 0);
        commandEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

//1. create the vertexes in CPU, and send them to GPU through vertex buffer
//2. map the MTLBuffer to shdader using argument tables. Depending on the device and resource type, there are 31 slots for each argument table
