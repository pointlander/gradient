// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"
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

// Square squares a tensor
func (context *Context[T]) Square(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	b := a
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
			c = a.Square()
		}
		context.Set(node, c.X)
		if k(c) {
			return true
		}
	} else {
		if cached == nil {
			c = a.Square()
		}
		context.Set(node, c.X)
		if k(c) {
			return true
		}
	}

	if a.Seed != 0 {
		c.Seed, c.Drop = a.Seed, a.Drop
		dropout := uint32((1 - a.Drop) * math.MaxUint32)
		// a derivatives
		{
			derivatives := func(index int, ad []T) {
				rows, bi := a.S[1], 0
				for i := 0; i < sizeB; i += width {
					bv, cd := b.X[i:i+width], c.D[index+bi*rows]

					axpy(cd, bv, ad)

					bi++
				}
			}
			index, rng := 0, a.Seed
			for j := 0; j < sizeA; j += width {
				if rng.Next() > dropout {
					index++
					continue
				}
				ad := a.D[j : j+width]
				derivatives(index, ad)
				index++
			}
		}

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
		}
		index, rows := 0, a.S[1]
		for i := 0; i < sizeB; i += width {
			bd := b.D[i : i+width]
			derivatives(index, bd)
			index += rows
		}

		return false
	}

	// a derivatives
	{
		derivatives := func(index int, ad []T) {
			rows, bi := a.S[1], 0
			for i := 0; i < sizeB; i += width {
				bv, cd := b.X[i:i+width], c.D[index+bi*rows]

				axpy(cd, bv, ad)

				bi++
			}
		}
		index := 0
		for j := 0; j < sizeA; j += width {
			ad := a.D[j : j+width]
			derivatives(index, ad)
			index++
		}
	}

	// b derivatives
	derivatives := func(index int, bd []T) {
		for j := 0; j < sizeA; j += width {
			av, cd := a.X[j:j+width], c.D[index]

			axpy(cd, av, bd)

			index++
		}
	}
	index, rows := 0, a.S[1]
	for i := 0; i < sizeB; i += width {
		bd := b.D[i : i+width]
		derivatives(index, bd)
		index += rows
	}

	return false
}

// Hadamard computes the hadamard product of two tensors
func (context *Context[T]) Hadamard(k Continuation[T], node int, a, b *V[T], options ...map[string]interface{}) bool {
	length := len(b.X)
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Hadamard(b)
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * b.X[i%length]
		b.D[i%length] += j * a.X[i]
	}
	return false
}

// T the transpose of the matrix
func (context *Context[T]) T(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S[1], a.S[0])
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.T()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	i := 0
	for p := 0; p < a.S[0]; p++ {
		for q := 0; q < a.S[1]; q++ {
			a.D[q*a.S[0]+p] += c.D[i]
			i++
		}
	}
	return false
}

// Slice a slice of the matrix
func (context *Context[T]) Slice(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	width := a.S[0]
	begin, end := *options[0]["begin"].(*int), *options[0]["end"].(*int)
	dd, ok := options[0]["d"].(*int)
	d := 0
	if ok {
		d = *dd
	}
	if d == 0 {
		c := NewV[T](end-begin, 1)
		cached := context.Get(node)
		if cached != nil {
			c.X = cached
		}
		if cached == nil {
			c = a.Slice(begin, end, d)
		}
		context.Set(node, c.X)
		if k(c) {
			return true
		}
		index := 0
		ad := a.D[begin:end]
		for j := range ad {
			ad[j] += c.D[index]
			index++
		}
	} else if d == 2 {
		c, size := NewV[T](end-begin, a.S[1]), len(a.X)
		cached := context.Get(node)
		if cached != nil {
			c.X = cached
		}
		if cached == nil {
			c = a.Slice(begin, end, d)
		}
		context.Set(node, c.X)
		if k(c) {
			return true
		}
		index := 0
		for i := 0; i < size; i += width {
			ad := a.D[i+begin : i+end]
			for j := range ad {
				ad[j] += c.D[index]
				index++
			}
		}
	}

	return false
}

// Concat concats two tensors
func (context *Context[T]) Concat(k Continuation[T], node int, a, b *V[T], options ...map[string]interface{}) bool {
	widthA, widthB := a.S[0], b.S[0]
	c, i, j := NewV[T](widthA+widthB, a.S[1]), 0, 0
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Concat(b)
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	index, i, j := 0, 0, 0
	for r := 0; r < a.S[1]; r++ {
		ad, bd := a.D[i:i+widthA], b.D[j:j+widthB]
		for s := range ad {
			ad[s] = c.D[index]
			index++
		}
		for s := range bd {
			bd[s] = c.D[index]
			index++
		}
		i += widthA
		j += widthB
	}
	return false
}

// Dropout is a dropout regularization function
func (context *Context[T]) Dropout(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	size, width := len(a.X), a.S[0]
	rng := options[0]["rng"].(*rand.Rand)
	drop := .1
	if options[0]["drop"] != nil {
		drop = *options[0]["drop"].(*float64)
	}
	c, drops := NewV[T](a.S...), make([]int, width)
	for i := range drops {
		if rng.Float64() > drop {
			drops[i] = 1
		}
	}
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Dropout(drop, drops)
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			if drops[j] == 1 {
				a.D[i+j] += c.D[i+j]
			}
		}
	}
	return false
}

// Sin the sine of a number
func (context *Context[T]) Sin(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Sin()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * cos(a.X[i])
	}
	return false
}

// Cos the cosine of a tensor
func (context *Context[T]) Cos(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Cos()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] -= j * sin(a.X[i])
	}
	return false
}

// Exp the base e exponential of a tensor
func (context *Context[T]) Exp(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Exp()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * c.X[i]
	}
	return false
}

// Log the natural logarithm of a tensor
func (context *Context[T]) Log(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Log()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j / a.X[i]
	}
	return false
}

// Sqrt is the sqrt of a number
func (context *Context[T]) Sqrt(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Sqrt()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j / (2 * c.X[i])
	}
	return false
}

// Inv is the inverse of a number
func (context *Context[T]) Inv(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Inv()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += -j / (a.X[i] * a.X[i])
	}
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

// TanH the hyperbolic tangent of a tensor
func (context *Context[T]) TanH(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.TanH()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		cx := c.X[i]
		a.D[i] += j * (1 - cx*cx)
	}
	return false
}

// Softplus the softplus activation function
func (context *Context[T]) Softplus(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		a = a.Softplus()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j / (1 + exp(-a.X[i]))
	}
	return false
}

// Everett computes the split reality activation function
func (context *Context[T]) Everett(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](2*a.S[0], a.S[1])
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Everett()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	if a.Seed != 0 {
		dropout := uint32((1 - a.Drop) * math.MaxUint32)
		index := 0
		for i := 0; i < a.S[1]; i++ {
			rng := a.Seed
			for j := 0; j < a.S[0]; j++ {
				if rng.Next() > dropout {
					index += 2
					continue
				}
				if c.X[index] != 0 || (c.X[index] == 0 && c.X[index+1] == 0) {
					a.D[index>>1] += c.D[index]
				}
				if c.X[index+1] != 0 || (c.X[index] == 0 && c.X[index+1] == 0) {
					a.D[index>>1] += c.D[index+1]
				}
				index += 2
			}
		}
		return false
	}

	for i, j := range c.D {
		if c.X[i] != 0 || (c.X[i&^1] == 0 && c.X[i|1] == 0) {
			a.D[i>>1] += j
		}
	}
	return false
}

// EverettReLu computes an adapter relu
func (context *Context[T]) EverettReLu(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](2*a.S[0], a.S[1])
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.EverettReLu()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		if c.X[i] != 0 {
			a.D[i>>1] += j
		}
	}
	return false
}

// ReLu computes the rectified linear activation function
func (context *Context[T]) ReLu(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.ReLu()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, j := range c.D {
		if c.X[i] != 0 {
			a.D[i] += j
		}
	}
	return false
}

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

// Softmax is the softmax function for big numbers
func (context *Context[T]) Softmax(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		S := S
		if len(options) > 0 {
			s, ok := options[0]["S"]
			if ok {
				S = s.(float64)
			}
		}
		c = a.Softmax(S)
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		for j := range c.X {
			if j == i {
				a.D[j] += d * cx * (1 - cx)
			} else {
				a.D[j] -= d * cx * c.X[j]
			}
		}
	}
	return false
}

// Sum sums a vector
func (context *Context[T]) Sum(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	c := NewV[T](1)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = a.Sum()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	d := c.D[0]
	for i := range a.D {
		a.D[i] += d
	}
	return false
}

// SumRows sums the rows of the matrix
func (context *Context[T]) SumRows(k Continuation[T], node int, a *V[T], options ...map[string]interface{}) bool {
	size, width := len(a.X), a.S[0]
	c := NewV[T](width)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c = c.SumRows()
	}
	context.Set(node, c.X)
	if k(c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			a.D[i+j] += c.D[j]
		}
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
