//
//  vertext_main.metal
//  MetalRender
//
//  Created by taox on 2/2/20.
//  Copyright © 2020 taox. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;



//constant float4 positions[6] = {
//  float4(0.5,0.5,0,1),
//  float4(-0.5,0.5,0,1),
//  float4(0.5,-0.5,0,1),
//  float4(-0.5,-0.5,0,1),
//  float4(-0.5,0.5,0,1),
//  float4(0.5,-0.5,0,1),
//};
//
//constant float3 colors[6] = {
//    float3(1,0,0),
//    float3(0,1,0),
//    float3(0,0,1),
//    float3(1,0,0),
//    float3(0,1,0),
//    float3(0,0,1),
//};

struct VertexOut {
    float4 position [[position]]; //[[ ]] syntax represents attributes
    float3 color;
};

//shader functions are stored in GPU buffers
vertex VertexOut vertex_main(device const float4 *positionBuffer [[buffer(0)]],
                             device const float3 *colorBuffer [[buffer(1)]],
                             constant float &timer [[buffer(2)]],
                             uint vertexId [[vertex_id]]) {
    auto out =  VertexOut {
        .position = positionBuffer[vertexId],
        .color = colorBuffer[vertexId],
    };
    out.position.x += timer;
    
    return out;
}

//fragment is responsible for coloring pixels in RGBA format
//each fragment represents one pixel
//fragment value is stored in the frame buffer
fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    return float4(in.color,1);
}


