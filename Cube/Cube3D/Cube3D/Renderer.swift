 //
 //  Renderer.swift
 //  Cube3D
 //
 //  Created by Tao Xu on 1/6/20.
 //  Copyright © 2020 Tao Xu. All rights reserved.
 //
 
 import Foundation
 
 struct Vertex {
    var x: Float = 0   // coordinate in 3D space
    var y: Float = 0
    var z: Float = 0
    
    var r: Float = 0   // color
    var g: Float = 0
    var b: Float = 0
    var a: Float = 1
    
    var nx: Float = 0  // normal vector (using for lighting)
    var ny: Float = 0
    var nz: Float = 0
 }
 
 struct Triangle {
    var vertices = [Vertex](repeating: Vertex(), count: 3)
 }
 
 let model: [Triangle] = { //6 facets -> 12 triangles
    var triangles = [Triangle]()
    var triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y: -10, z:  10, r: 0, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: 1)
    triangle.vertices[1] = Vertex(x: -10, y:  10, z:  10, r: 0, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: 1)
    triangle.vertices[2] = Vertex(x:  10, y: -10, z:  10, r: 0, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: 1)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y:  10, z:  10, r: 0, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: 1)
    triangle.vertices[1] = Vertex(x:  10, y: -10, z:  10, r: 0, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: 1)
    triangle.vertices[2] = Vertex(x:  10, y:  10, z:  10, r: 0, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: 1)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y: -10, z: -10, r: 1, g: 0, b: 0, a: 1, nx: 0, ny: 0, nz: -1)
    triangle.vertices[1] = Vertex(x:  10, y: -10, z: -10, r: 0, g: 1, b: 0, a: 1, nx: 0, ny: 0, nz: -1)
    triangle.vertices[2] = Vertex(x:  10, y:  10, z: -10, r: 0, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: -1)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y: -10, z: -10, r: 1, g: 1, b: 0, a: 1, nx: 0, ny: 0, nz: -1)
    triangle.vertices[1] = Vertex(x:  10, y:  10, z: -10, r: 0, g: 1, b: 1, a: 1, nx: 0, ny: 0, nz: -1)
    triangle.vertices[2] = Vertex(x: -10, y:  10, z: -10, r: 1, g: 0, b: 1, a: 1, nx: 0, ny: 0, nz: -1)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y:  10, z: -10, r: 1, g: 0, b: 0, a: 1, nx: 0, ny: 1, nz: 0)
    triangle.vertices[1] = Vertex(x: -10, y:  10, z:  10, r: 1, g: 0, b: 0, a: 1, nx: 0, ny: 1, nz: 0)
    triangle.vertices[2] = Vertex(x:  10, y:  10, z: -10, r: 1, g: 0, b: 0, a: 1, nx: 0, ny: 1, nz: 0)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y:  10, z:  10, r: 1, g: 0, b: 0, a: 1, nx: 0, ny: 1, nz: 0)
    triangle.vertices[1] = Vertex(x:  10, y:  10, z: -10, r: 1, g: 0, b: 0, a: 1, nx: 0, ny: 1, nz: 0)
    triangle.vertices[2] = Vertex(x:  10, y:  10, z:  10, r: 1, g: 0, b: 0, a: 1, nx: 0, ny: 1, nz: 0)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y: -10, z: -10, r: 1, g: 1, b: 1, a: 1, nx: 0, ny: -1, nz: 0)
    triangle.vertices[1] = Vertex(x:  10, y: -10, z: -10, r: 1, g: 1, b: 1, a: 1, nx: 0, ny: -1, nz: 0)
    triangle.vertices[2] = Vertex(x: -10, y: -10, z:  10, r: 1, g: 1, b: 1, a: 1, nx: 0, ny: -1, nz: 0)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x: -10, y: -10, z:  10, r: 1, g: 1, b: 1, a: 1, nx: 0, ny: -1, nz: 0)
    triangle.vertices[1] = Vertex(x:  10, y: -10, z:  10, r: 1, g: 1, b: 1, a: 1, nx: 0, ny: -1, nz: 0)
    triangle.vertices[2] = Vertex(x:  10, y: -10, z: -10, r: 1, g: 1, b: 1, a: 1, nx: 0, ny: -1, nz: 0)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x:  10, y: -10, z: -10, r: 0, g: 1, b: 0, a: 1, nx: 1, ny: 0, nz: 0)
    triangle.vertices[1] = Vertex(x:  10, y: -10, z:  10, r: 0, g: 1, b: 0, a: 1, nx: 1, ny: 0, nz: 0)
    triangle.vertices[2] = Vertex(x:  10, y:  10, z: -10, r: 0, g: 1, b: 0, a: 1, nx: 1, ny: 0, nz: 0)
    triangles.append(triangle)

    triangle = Triangle()
    triangle.vertices[0] = Vertex(x:  10, y: -10, z:  10, r: 0, g: 1, b: 0, a: 1, nx: 1, ny: 0, nz: 0)
    triangle.vertices[1] = Vertex(x:  10, y:  10, z: -10, r: 0, g: 1, b: 0, a: 1, nx: 1, ny: 0, nz: 0)
    triangle.vertices[2] = Vertex(x:  10, y:  10, z:  10, r: 0, g: 1, b: 0, a: 1, nx: 1, ny: 0, nz: 0)
    triangles.append(triangle)
    
    

    
    return triangles
 }()
