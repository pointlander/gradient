// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

// Type is a type of matrix
type Type uint8

const (
	Parameters Type = iota
	Data
)

type (
	//F32  float32
	//F64  float64
	//C64  complex64
	//C128 complex128
	// Number is a number
	Number interface {
		~float32 | ~float64 | ~complex64 | ~complex128
	}
	// RNG is a random number generator
	RNG uint32
	// V is a tensor value
	V[T Number] struct {
		T      Type
		N      string // the name
		Seed   RNG
		Drop   float64
		X      []T   // the tensor
		D      []T   // the derivative
		S      []int // the shape
		States [][]T
	}
	// Set is a set of V
	Set[T Number] struct {
		Weights []*V[T]
		ByName  map[string]*V[T]
	}
	// Continuation is a continuation
	Continuation[T Number] func(a *V[T]) bool
	// Meta is a function that takes a continuation and return a continuation
	Meta[T Number] func(k Continuation[T]) Continuation[T]
	// Unary is a unary function
	Unary[T Number] func(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool
	// Binary is a binary function
	Binary[T Number] func(k Continuation[T], node int, a, b *V[T], options ...map[string]interface{}) bool
	// Operation is an operation that takes multiple parameters
	Operation[T Number] func(k Continuation[T], node int, a ...*V[T]) bool
	// Context is a function context
	Context[T Number] struct {
		Quantize uint
		Node     int
		Cache    map[int][]T
	}
)

// Next returns the next random number
func (r *RNG) Next() uint32 {
	lfsr := *r
	lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & LFSRMask)
	*r = lfsr
	return uint32(lfsr)
}

// Clear clears the cache
func (c *Context[T]) Clear() {
	c.Cache = make(map[int][]T)
}

// Get gets a value from the cache
func (c *Context[T]) Get(node int) []T {
	if c.Cache != nil {
		return c.Cache[node]
	}
	return nil
}

// Set sets a value in the cache
func (c *Context[T]) Set(node int, value []T) {
	if c.Cache != nil {
		c.Cache[node] = value
	}
}

// Op is a operation
func (context *Context[T]) Op(op Operation[T]) func(a ...Meta[T]) Meta[T] {
	return func(a ...Meta[T]) Meta[T] {
		node := context.Node
		context.Node++
		return func(k Continuation[T]) Continuation[T] {
			var call func(a []Meta[T], b []*V[T]) (bool, Continuation[T])
			call = func(a []Meta[T], b []*V[T]) (bool, Continuation[T]) {
				if len(a) == 0 {
					return op(k, node, b...), nil
				}
				derivatives := false
				continuation := a[0](func(c *V[T]) bool {
					derivatives, _ = call(a[1:], append(b, c))
					return derivatives
				})
				return derivatives, continuation
			}
			_, continuation := call(a, make([]*V[T], 0, len(a)))
			return continuation
		}
	}
}

// B converts a binary function into an operator
func (context *Context[T]) B(op Binary[T]) func(a, b Meta[T], options ...map[string]interface{}) Meta[T] {
	return func(a, b Meta[T], options ...map[string]interface{}) Meta[T] {
		node := context.Node
		context.Node++
		return func(k Continuation[T]) Continuation[T] {
			return a(func(a *V[T]) bool {
				derivatives := false
				b(func(b *V[T]) bool {
					derivatives = op(k, node, a, b, options...)
					return derivatives
				})
				return derivatives
			})
		}
	}
}

// U converts a unary function into an operator
func (context *Context[T]) U(op Unary[T]) func(a Meta[T], options ...map[string]interface{}) Meta[T] {
	return func(a Meta[T], options ...map[string]interface{}) Meta[T] {
		node := context.Node
		context.Node++
		return func(k Continuation[T]) Continuation[T] {
			return a(func(b *V[T]) bool {
				return op(k, node, b, options...)
			})
		}
	}
}

// Panic marks a place we should never get to
func Panic[T Number](a *V[T]) bool {
	panic("should not be here")
}

// NewSet creates a new weight set
func (context *Context[T]) NewSet() Set[T] {
	return Set[T]{
		ByName: make(map[string]*V[T]),
	}
}

// Add adds weights to a set
func (s *Set[T]) Add(name string, d ...int) {
	v := NewV[T](d...)
	v.N = name
	s.Weights = append(s.Weights, v)
	s.ByName[name] = v
}

// Get gets weights from the set by name
func (s *Set[T]) Get(name string) Meta[T] {
	return s.ByName[name].Meta()
}

// Copy generates a copy of a set
func (s *Set[T]) Copy(context *Context[T]) Set[T] {
	n := context.NewSet()
	for i := range s.Weights {
		cp := s.Weights[i].Copy()
		n.Weights = append(n.Weights, cp)
		n.ByName[cp.N] = cp
	}
	return n
}

// Zero zeros the partial derivatives
func (s *Set[T]) Zero() {
	for i := range s.Weights {
		s.Weights[i].Zero()
	}
}

// Gradient computes the gradient
func Gradient[T Number](a Meta[T]) (cost V[T]) {
	a(func(a *V[T]) bool {
		cost = *a
		a.D[0] = 1
		return false
	})
	return
}
