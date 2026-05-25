// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/gob"
	"math/cmplx"
	"math/rand"
	"os"
)

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

// Type is a type of matrix
type Type uint8

const (
	Parameters Type = iota
	Bias
	Data
)

type (
	// Number is a number
	Number interface {
		~float32 | ~float64 | ~complex64 | ~complex128
	}
	// RNG is a random number generator
	RNG uint32
	// V is a tensor value
	V[T Number] struct {
		Type
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
		Iteration uint64
		Cost      float64
		Epoch     uint64
		Weights   []*V[T]
		ByName    map[string]*V[T]
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

// AddBias adds bias weights to a set
func (s *Set[T]) AddBias(name string, d ...int) {
	v := NewV[T](d...)
	v.Type = Bias
	v.N = name
	s.Weights = append(s.Weights, v)
	s.ByName[name] = v
}

// AddData adds data to a set
func (s *Set[T]) AddData(name string, d ...int) {
	v := NewV[T](d...)
	v.Type = Data
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

// InitAdam initializes a set for adam optimization
func (s *Set[T]) InitAdam(rng *rand.Rand) {
	for ii := range s.Weights {
		w := s.Weights[ii]
		if w.Type == Data {
			w.X = w.X[:cap(w.X)]
			continue
		}
		if w.Type == Bias {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]T, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]T, len(w.X))
			}
			continue
		}
		factor := sqrt(2.0 / convert[T](float64(w.S[0])))
		for range cap(w.X) {
			switch any(factor).(type) {
			case float32:
				w.X = append(w.X, any(float32(rng.NormFloat64())).(T)*factor)
			case float64:
				w.X = append(w.X, any(rng.NormFloat64()).(T)*factor)
			case complex64:
				w.X = append(w.X, any(complex64(complex(float32(rng.NormFloat64()), float32(rng.NormFloat64())))).(T)*factor)
			case complex128:
				w.X = append(w.X, any(complex128(complex(rng.NormFloat64(), rng.NormFloat64()))).(T)*factor)
			}
		}
		w.States = make([][]T, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]T, len(w.X))
		}
	}
}

// Zero zeros the partial derivatives
func (s *Set[T]) Zero() {
	for i := range s.Weights {
		s.Weights[i].Zero()
	}
}

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-1
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

func (s *Set[T]) pow(x T) T {
	y := pow(x, convert[T](float64(s.Iteration+1)))
	if isnan(y) || isinf(y) {
		return 0
	}
	return y
}

func (s *Set[T]) Adam(B1, B2, Eta T) {
	norm := T(0.0)
	for _, p := range s.Weights {
		for _, d := range p.D {
			norm += d * d
		}
	}
	norm = sqrt(norm)
	b1, b2 := s.pow(B1), s.pow(B2)
	scaling := T(1.0)
	switch n := any(norm).(type) {
	case float32:
		if n > 1 {
			scaling = 1 / norm
		}
	case float64:
		if n > 1 {
			scaling = 1 / norm
		}
	case complex64:
		if cmplx.Abs(complex128(n)) > 1 {
			scaling = 1 / norm
		}
	case complex128:
		if cmplx.Abs(n) > 1 {
			scaling = 1 / norm
		}
	}
	for _, w := range s.Weights {
		for ii, d := range w.D {
			if w.Type == Data {
				continue
			}
			g := d * scaling
			m := B1*w.States[StateM][ii] + (1-B1)*g
			v := B2*w.States[StateV][ii] + (1-B2)*g*g
			w.States[StateM][ii] = m
			w.States[StateV][ii] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			switch v := any(vhat).(type) {
			case float32:
				if v < 0 {
					vhat = 0
				}
			case float64:
				if v < 0 {
					vhat = 0
				}
			case complex64:
				if cmplx.Abs(complex128(v)) < 0 {
					vhat = 0
				}
			case complex128:
				if cmplx.Abs(v) < 0 {
					vhat = 0
				}
			}
			w.X[ii] -= Eta * mhat / (sqrt(vhat) + 1e-8)
		}
	}
	s.Iteration++

}

// Save saves a set of weights
func (s *Set[T]) Save(file string, cost float64, epoch uint64) error {
	s.Cost = cost
	s.Epoch = epoch
	output, err := os.Create("file")
	if err != nil {
		return err
	}
	encoder := gob.NewEncoder(output)
	return encoder.Encode(s)
}

// Open opens a set of weights
func (s *Set[T]) Open(name string) (float64, uint64, error) {
	input, err := os.Open(name)
	if err != nil {
		return -1, 0, err
	}
	decoder := gob.NewDecoder(input)
	err = decoder.Decode(s)
	if err != nil {
		return -1, 0, err
	}
	return s.Cost, s.Epoch, nil
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
