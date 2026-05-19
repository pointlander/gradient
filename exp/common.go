// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Type is a type of matrix
type Type uint8

const (
	Parameters Type = iota
	Data
)

type (
	F32  float32
	F64  float64
	C64  complex64
	C128 complex128
	// Number is a number
	Number interface {
		F32 | F64 | C64 | C128
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
)
