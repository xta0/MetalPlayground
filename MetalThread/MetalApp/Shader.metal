//
//  Shader.metal
//  MetalApp
//
//  Created by taox on 4/19/21.
//

#include <metal_stdlib>
using namespace metal;


kernel void add(device float* input[[buffer(0)]],
                device float* output[[buffer(1)]],
                ushort tids[[threads_per_grid]]) {
    
    output[0] = tids;
    
    
    
}

