// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Copy copies src tensors into dst
func (context *Context[T]) Copy(k Continuation[T], node int, dst, src *V[T], options ...map[string]interface{}) bool {
	c := dst.CopyV(src)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		src.D[i] += j
		dst.D[i] += j
	}
	return false
}
