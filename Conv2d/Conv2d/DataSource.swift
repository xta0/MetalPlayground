//
//  DataSource.swift
//  Conv2d
//
//  Created by Tao Xu on 3/9/20.
//  Copyright © 2020 Tao Xu. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

class DataSource: MPSCNNConvolutionDataSource {
    
    var description: String
    
    let name: String
    let kernelWidth: Int
    let kernelHeight: Int
    let inputFeatureChannels: Int
    let outputFeatureChannels: Int
    
    init() {
        
    }
    
    func dataType() -> MPSDataType {
        <#code#>
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        <#code#>
    }
    
    func weights() -> UnsafeMutableRawPointer {
        <#code#>
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        <#code#>
    }
    
    func purge() {
        <#code#>
    }
    
    func label() -> String? {
        <#code#>
    }
    
    func copy(with zone: NSZone? = nil) -> Any {
        <#code#>
    }
    
    func isEqual(_ object: Any?) -> Bool {
        <#code#>
    }
    
    var hash: Int
    
    var superclass: AnyClass?
    
    func `self`() -> Self {
        <#code#>
    }
    
    func perform(_ aSelector: Selector!) -> Unmanaged<AnyObject>! {
        <#code#>
    }
    
    func perform(_ aSelector: Selector!, with object: Any!) -> Unmanaged<AnyObject>! {
        <#code#>
    }
    
    func perform(_ aSelector: Selector!, with object1: Any!, with object2: Any!) -> Unmanaged<AnyObject>! {
        <#code#>
    }
    
    func isProxy() -> Bool {
        <#code#>
    }
    
    func isKind(of aClass: AnyClass) -> Bool {
        <#code#>
    }
    
    func isMember(of aClass: AnyClass) -> Bool {
        <#code#>
    }
    
    func conforms(to aProtocol: Protocol) -> Bool {
        <#code#>
    }
    
    func responds(to aSelector: Selector!) -> Bool {
        <#code#>
    }
    

    
   
    
}
