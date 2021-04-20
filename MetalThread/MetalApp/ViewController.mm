//
//  ViewController.m
//  MetalApp
//
//  Created by taox on 4/19/21.
//

#import "ViewController.h"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <vector>
#include <iostream>

@interface ViewController ()

@end
/*
 ushort2 gid[[threadgroup_position_in_grid]],
 ushort2 tid[[thread_position_in_threadgroup]],
 */
@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    std::vector<float> src;
    for(int i=0; i<16; i++) {
        src.emplace_back(1.0);
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> lib = [device newDefaultLibrary];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
    
    id<MTLBuffer> buffer = [device newBufferWithBytes:src.data()
                                            length:16*sizeof(float)
                                           options:MTLResourceCPUCacheModeDefaultCache];
    id<MTLBuffer> buffer2 = [device newBufferWithLength:16*sizeof(float)
                                                options:MTLResourceCPUCacheModeDefaultCache];
    id<MTLFunction> func = [lib newFunctionWithName:@"add"];
    NSError* errors;
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:func error:&errors];
    NSLog(@"maxTotalThreadsPerThreadgroup: %d",(int)state.maxTotalThreadsPerThreadgroup);
    NSLog(@"threadExecutionWidth: %d",(int)state.threadExecutionWidth);
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
    MTLSize threadGroups = MTLSizeMake(1, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(16, 1, 1);
    MTLSize threadsPerGrid = MTLSizeMake(4, 4, 1);
    [encoder setComputePipelineState:state];
//    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBuffer:buffer2 offset:0 atIndex:1];
    [encoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    std::vector<float> dst(16, 0.0);
    memcpy(dst.data(), buffer2.contents, 16*sizeof(float));
    for(int i=0; i<16; i++) {
        std::cout<<dst[i]<<", ";
    }
    std::cout<<std::endl;
}


@end
