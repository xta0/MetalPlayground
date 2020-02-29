/*
  Copyright (c) 2016-2017 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import UIKit

extension UIImage {
  /** 
    Converts the image into an array of RGBA bytes.
   */
  @nonobjc public func toByteArray() -> [UInt8] {
    let width = Int(size.width)
    let height = Int(size.height)
    var bytes = [UInt8](repeating: 0, count: width * height * 4)

    bytes.withUnsafeMutableBytes { ptr in
      if let context = CGContext(
                    data: ptr.baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: width * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {

        if let image = self.cgImage {
          let rect = CGRect(x: 0, y: 0, width: size.width, height: size.height)
          context.draw(image, in: rect)
        }
      }
    }
    return bytes
  }

  /**
    Creates a new UIImage from an array of RGBA bytes.
   */
  @nonobjc public class func fromByteArray(_ bytes: UnsafeMutableRawPointer,
                                           width: Int,
                                           height: Int) -> UIImage {

    if let context = CGContext(data: bytes, width: width, height: height,
                               bitsPerComponent: 8, bytesPerRow: width * 4,
                               space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
       let cgImage = context.makeImage() {
      return UIImage(cgImage: cgImage, scale: 0, orientation: .up)
    } else {
      return UIImage()
    }
  }
}
