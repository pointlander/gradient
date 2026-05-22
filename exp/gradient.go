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

// Sub subtracts two tensors
func (context *Context[T]) Sub(k Continuation[T], node int, a, b *V[T], options ...map[string]interface{}) bool {
	length := len(b.X)
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Sub(b)
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j
		b.D[i%length] -= j
	}
	return false
}

// Mul multiplies two tensors
func (context *Context[T]) Mul(k Continuation[T], node int, a, b *V[T], options ...map[string]interface{}) bool {
	width := a.S[0]
	sizeA, sizeB, c :=
		len(a.X), len(b.X), NewV[T](a.S[1], b.S[1])
	c.X = c.X[:cap(c.X)]
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if a.Seed != 0 {
		if cached == nil {
			c = a.Mul(b)
		}
		context.Set(node, c.X)
		if k(c) {
			return true
		}
	} else {
		if cached == nil {
			c = a.Mul(b)
		}
		context.Set(node, c.X)
		if k(c) {
			return true
		}
	}

	if a.Seed != 0 {
		dropout := uint32((1 - a.Drop) * math.MaxUint32)

		done := make(chan bool, 8)

		// a derivatives
		go func() {
			derivativeDone := make(chan bool, 8)
			derivatives := func(index int, ad []T) {
				rows, bi := a.S[1], 0
				for i := 0; i < sizeB; i += width {
					bv, cd := b.X[i:i+width], c.D[index+bi*rows]

					axpy(cd, bv, ad)

					bi++
				}
				derivativeDone <- true
			}
			index, rng := 0, a.Seed
			for j := 0; j < sizeA; j += width {
				if rng.Next() > dropout {
					index++
					continue
				}
				ad := a.D[j : j+width]
				go derivatives(index, ad)
				index++
			}
			rng = a.Seed
			for j := 0; j < sizeA; j += width {
				if rng.Next() > dropout {
					continue
				}
				<-derivativeDone
			}
			done <- true
		}()

		// b derivatives
		derivativeDone := make(chan bool, 8)
		derivatives := func(index int, bd []T) {
			rng := a.Seed
			for j := 0; j < sizeA; j += width {
				if rng.Next() > dropout {
					index++
					continue
				}
				av, cd := a.X[j:j+width], c.D[index]

				axpy(cd, av, bd)

				index++
			}
			derivativeDone <- true
		}
		index, rows := 0, a.S[1]
		for i := 0; i < sizeB; i += width {
			bd := b.D[i : i+width]
			go derivatives(index, bd)
			index += rows
		}
		for i := 0; i < sizeB; i += width {
			<-derivativeDone
		}
		<-done

		return false
	}

	done := make(chan bool, 8)

	// a derivatives
	go func() {
		derivativeDone := make(chan bool, 8)
		derivatives := func(index int, ad []T) {
			rows, bi := a.S[1], 0
			for i := 0; i < sizeB; i += width {
				bv, cd := b.X[i:i+width], c.D[index+bi*rows]

				axpy(cd, bv, ad)

				bi++
			}
			derivativeDone <- true
		}
		index := 0
		for j := 0; j < sizeA; j += width {
			ad := a.D[j : j+width]
			go derivatives(index, ad)
			index++
		}
		for j := 0; j < sizeA; j += width {
			<-derivativeDone
		}
		done <- true
	}()

	// b derivatives
	derivativeDone := make(chan bool, 8)
	derivatives := func(index int, bd []T) {
		for j := 0; j < sizeA; j += width {
			av, cd := a.X[j:j+width], c.D[index]

			axpy(cd, av, bd)

			index++
		}
		derivativeDone <- true
	}
	index, rows := 0, a.S[1]
	for i := 0; i < sizeB; i += width {
		bd := b.D[i : i+width]
		go derivatives(index, bd)
		index += rows
	}
	for i := 0; i < sizeB; i += width {
		<-derivativeDone
	}
	<-done

	return false
}

// Sigmoid computes the sigmoid of a vector
func (context *Context[T]) Sigmoid(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Sigmoid()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		cx := c.X[i]
		a.D[i] += j * cx * (1 - cx)
	}
	return false
}

// Quadratic computes the quadratic cost of two tensors
func (context *Context[T]) Quadratic(k Continuation[T], node int, a, b *V[T], options ...map[string]interface{}) bool {
	width := a.S[0]
	c, size := NewV[T](a.S[1]), len(a.X)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Quadratic(b)
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	index := 0
	for i := 0; i < size; i += width {
		av, bv, ad, bd, d := a.X[i:i+width], b.X[i:i+width], a.D[i:i+width], b.D[i:i+width], c.D[index]
		for j, ax := range av {
			ad[j] += (ax - bv[j]) * d
			bd[j] += (bv[j] - ax) * d
		}
		index++
	}
	return false
}
