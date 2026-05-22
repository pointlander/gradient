// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
)

// NewV create a new tensor value
func NewV[T Number](s ...int) *V[T] {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	size := s[0] * s[1]
	return &V[T]{
		X: make([]T, 0, size),
		D: make([]T, size),
		S: s,
	}
}

// NewV create a new identity tensor value
func Identity[T Number](s ...int) *V[T] {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	if s[0] != s[1] {
		panic("identity matrix must be square")
	}
	size := s[0] * s[1]
	identity := V[T]{
		X: make([]T, size),
		D: make([]T, size),
		S: s,
	}
	j := 0
	for i := 0; i < size; i += s[0] {
		identity.X[i+j] = 1.0
		j++
	}
	return &identity
}

// Copy copies the weights of the value
func (a *V[T]) Copy() *V[T] {
	return &V[T]{
		N: a.N,
		X: a.X,
		D: make([]T, len(a.D)),
		S: a.S,
	}
}

// Meta returns a meta for the value
func (a *V[T]) Meta() Meta[T] {
	return func(k Continuation[T]) Continuation[T] {
		k(a)
		return Panic[T]
	}
}

// Zero zeros the partial derivatives
func (a *V[T]) Zero() {
	for i := range a.D {
		a.D[i] = 0
	}
}

// Set sets the values and zeros the partial derivatives
func (a *V[T]) Set(values []T) {
	for i, value := range values {
		if i >= len(a.X) {
			a.X = append(a.X, value)
			continue
		}
		a.X[i] = value
	}
	a.Zero()
}

// Copy copies src to dst
func (dst *V[T]) CopyV(src *V[T]) *V[T] {
	if len(src.S) != 2 || len(dst.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if (src.S[0] != dst.S[0]) || (src.S[1] != dst.S[1]) {
		panic("dimensions are not the same")
	}
	c := NewV[T](src.S...)
	c.X = append(c.X, src.X...)
	copy(dst.X, src.X)
	return c
}

// Add adds two tensors
func (a *V[T]) Add(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}

	c := NewV[T](a.S...)
	if a.Seed != 0 {
		dropout, index := uint32((1-a.Drop)*math.MaxUint32), 0
		c.Seed, c.Drop = a.Seed, a.Drop
		for i := 0; i < a.S[1]; i++ {
			rng := a.Seed
			for j := 0; j < a.S[0]; j++ {
				if rng.Next() > dropout {
					c.X = append(c.X, 0)
					index++
					continue
				}
				c.X = append(c.X, a.X[index]+b.X[index%length])
				index++
			}
		}
	} else {
		for i, j := range a.X {
			c.X = append(c.X, j+b.X[i%length])
		}
	}
	return c
}

// Sub subtracts two tensors
func (a *V[T]) Sub(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV[T](a.S...)
	for i, j := range a.X {
		c.X = append(c.X, j-b.X[i%length])
	}
	return c
}

// Mul multiplies two tensors
func (a *V[T]) Mul(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] {
		panic("first dimension is not the same")
	}
	sizeA, sizeB, c, done :=
		len(a.X), len(b.X), NewV[T](a.S[1], b.S[1]), make(chan bool, 8)
	c.X = c.X[:cap(c.X)]
	if a.Seed != 0 {
		c.Seed, c.Drop = a.Seed, a.Drop
		dropout := uint32((1 - a.Drop) * math.MaxUint32)
		mul := func(bv []T, i int) {
			rng := a.Seed
			for j := 0; j < sizeA; j += width {
				if rng.Next() > dropout {
					i++
					continue
				}

				av := a.X[j : j+width]
				sum := dot(av, bv)

				c.X[i] = sum
				i++
			}
			done <- true
		}
		index, step := 0, sizeA/width
		for i := 0; i < sizeB; i += width {
			go mul(b.X[i:i+width], index)
			index += step
		}
		for i := 0; i < sizeB; i += width {
			<-done
		}
	} else {
		mul := func(bv []T, i int) {
			for j := 0; j < sizeA; j += width {
				var sum T
				av := a.X[j : j+width]
				for k, bx := range bv {
					sum += av[k] * bx
				}
				c.X[i] = sum
				i++
			}
			done <- true
		}
		index, step := 0, sizeA/width
		for i := 0; i < sizeB; i += width {
			go mul(b.X[i:i+width], index)
			index += step
		}
		for i := 0; i < sizeB; i += width {
			<-done
		}
	}
	return c
}

// Sigmoid computes the sigmoid of a vector
func (a *V[T]) Sigmoid() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		e := exp(j)
		if isinf(e) {
			if sign(e) == 1 {
				c.X = append(c.X, 1.0)
			} else {
				c.X = append(c.X, 0)
			}
		} else {
			c.X = append(c.X, e/(e+1))
		}
	}
	return c
}

// Quadratic computes the quadratic cost of two tensors
func (a *V[T]) Quadratic(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := NewV[T](a.S[1]), len(a.X)
	for i := 0; i < size; i += width {
		var sum T
		av, bv := a.X[i:i+width], b.X[i:i+width]
		for j, ax := range av {
			p := ax - bv[j]
			sum += p * p
		}
		c.X = append(c.X, sum*.5)
	}
	return c
}
