//
//  CIFAR_ConvLayer.swift
//  MPSCNNHelloWorld
//
//  Created by Tao Xu on 2/27/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class CIFAR_ConvLayer : MPSCNNConvolution {
    init(kernelWidth: UInt,
         kernelHeight: UInt,
         inputFeatureChannels: UInt,
         outputFeatureChannels: UInt,
         neuronFilter: MPSCNNNeuron? = nil,
         device: MTLDevice,
         kernelParamsBinaryName: String,
         padding willPad: Bool = true,
         strideXY: (UInt, UInt) = (1, 1),
         destinationFeatureChannelOffset: UInt = 0,
         groupNum: UInt = 1){
        
        // calculate the size of weights and bias required to be memory mapped into memory
        let sizeBias = outputFeatureChannels * UInt(MemoryLayout<Float>.size)
        let sizeWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannels * UInt(MemoryLayout<Float>.size)
        
        // get the url to this layer's weights and bias
        let wtPath = Bundle.main.path(forResource: "mpscnn_" + kernelParamsBinaryName, ofType: "txt")
        let bsPath = Bundle.main.path(forResource: "mpscnn_" + kernelParamsBinaryName, ofType: "txt")
        
        // open file descriptors in read-only mode to parameter files
        let fd_w = open( wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        let fd_b = open( bsPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        
        assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")
        assert(fd_b != -1, "Error: failed to open output file at \""+bsPath!+"\"  errno = \(errno)\n")
        
        // memory map the parameters
        let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0)
        let hdrB = mmap(nil, Int(sizeBias), PROT_READ, MAP_FILE | MAP_SHARED, fd_b, 0)
        
        // cast Void pointers to Float
        let w = UnsafePointer(hdrW!.bindMemory(to: Float.self, capacity: Int(sizeWeights)))
        let b = UnsafePointer(hdrB!.bindMemory(to: Float.self, capacity: Int(sizeBias)))
        
        assert(w != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
        assert(b != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
        
        // create appropriate convolution descriptor with appropriate stride
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelWidth),
                                                   kernelHeight: Int(kernelHeight),
                                                   inputFeatureChannels: Int(inputFeatureChannels),
                                                   outputFeatureChannels: Int(outputFeatureChannels),
                                                   neuronFilter: neuronFilter)
        convDesc.strideInPixelsX = Int(strideXY.0)
        convDesc.strideInPixelsY = Int(strideXY.1)
        
        assert(groupNum > 0, "Group size can't be less than 1")
        convDesc.groups = Int(groupNum)
        
        // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
        super.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: w,
                   biasTerms: b,
                   flags: MPSCNNConvolutionFlags.none)
        self.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        
        // unmap files at initialization of MPSCNNConvolution, the weights are copied and packed internally we no longer require these
        assert(munmap(hdrW, Int(sizeWeights)) == 0, "munmap failed with errno = \(errno)")
        assert(munmap(hdrB, Int(sizeBias))    == 0, "munmap failed with errno = \(errno)")
        
        // close file descriptors
        close(fd_w)
        close(fd_b)
        
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
