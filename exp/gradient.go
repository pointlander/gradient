// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
)

// Copy copies src tensors into dst
func (context *Context[T]) Copy(k Continuation[T], node int, dst, src *V[T], options ...map[string]interface{}) bool {
	c := dst.CopyV(src)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		src.D[i] += j
		dst.D[i] += j
	}
	return false
}

// Add adds two tensors
func (context *Context[T]) Add(k Continuation[T], node int, a, b *V[T], options ...map[string]interface{}) bool {
	length := len(b.X)
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Add(b)
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}

	if a.Seed != 0 {
		dropout, index := uint32((1-a.Drop)*math.MaxUint32), 0
		for i := 0; i < a.S[1]; i++ {
			rng := a.Seed
			for j := 0; j < a.S[0]; j++ {
				if rng.Next() > dropout {
					index++
					continue
				}
				d := c.D[index]
				a.D[index] += d
				b.D[index%length] += d
				index++
			}
		}
	} else {
		for i, j := range c.D {
			a.D[i] += j
			b.D[i%length] += j
		}
	}
	return false
}
