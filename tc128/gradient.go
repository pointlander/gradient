// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tc128

import (
	"io/ioutil"
	"math"
	"math/cmplx"
	"math/rand"
	"os"

	"github.com/golang/protobuf/proto"

	pro "github.com/pointlander/gradient/tc128/proto_tc128"
)

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

type (
	// RNG is a random number generator
	RNG uint32
	// V is a tensor value
	V struct {
		N      string // the name
		Seed   RNG
		Drop   float64
		X      []complex128 // the tensor
		D      []complex128 // the derivative
		S      []int        // the shape
		States [][]complex128
	}
	// Set is a set of V
	Set struct {
		Weights []*V
		ByName  map[string]*V
	}
	// Continuation is a continuation
	Continuation func(a *V) bool
	// Meta is a function that takes a continuation and return a continuation
	Meta func(k Continuation) Continuation
	// Unary is a unary function
	Unary func(k Continuation, node int, a *V, options ...map[string]interface{}) bool
	// Binary is a binary function
	Binary func(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool
	// Operation is an operation that takes multiple parameters
	Operation func(k Continuation, node int, a ...*V) bool
)

func abs(a complex128) complex128 {
	return complex(cmplx.Abs(a), 0)
}

var (
	sin   = cmplx.Sin
	cos   = cmplx.Cos
	exp   = cmplx.Exp
	log   = cmplx.Log
	sqrt  = cmplx.Sqrt
	isinf = cmplx.IsInf
)

func sign(a complex128) int {
	return 0
}
func convert(a float64) complex128 {
	return complex(a, 0)
}

// Next returns the next random number
func (r *RNG) Next() uint32 {
	lfsr := *r
	lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & LFSRMask)
	*r = lfsr
	return uint32(lfsr)
}

// NewV create a new tensor value
func NewV(s ...int) V {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	size := s[0] * s[1]
	return V{
		X: make([]complex128, 0, size),
		D: make([]complex128, size),
		S: s,
	}
}

// NewV create a new identity tensor value
func Identity(s ...int) V {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	if s[0] != s[1] {
		panic("identity matrix must be square")
	}
	size := s[0] * s[1]
	identity := V{
		X: make([]complex128, size),
		D: make([]complex128, size),
		S: s,
	}
	j := 0
	for i := 0; i < size; i += s[0] {
		identity.X[i+j] = 1
		j++
	}
	return identity
}

// Panic marks a place we should never get to
func Panic(a *V) bool {
	panic("should not be here")
	return false
}

// Copy copies the weights of the value
func (a *V) Copy() V {
	return V{
		N: a.N,
		X: a.X,
		D: make([]complex128, len(a.D)),
		S: a.S,
	}
}

// Meta returns a meta for the value
func (a *V) Meta() Meta {
	return func(k Continuation) Continuation {
		k(a)
		return Panic
	}
}

// Zero zeros the partial derivatives
func (a *V) Zero() {
	for i := range a.D {
		a.D[i] = 0
	}
}

// Set sets the values and zeros the partial derivatives
func (a *V) Set(values []complex128) {
	for i, value := range values {
		if i >= len(a.X) {
			a.X = append(a.X, value)
			continue
		}
		a.X[i] = value
	}
	a.Zero()
}

// NewSet creates a new weight set
func NewSet() Set {
	return Set{
		ByName: make(map[string]*V),
	}
}

// Add adds weights to a set
func (s *Set) Add(name string, d ...int) {
	v := NewV(d...)
	v.N = name
	s.Weights = append(s.Weights, &v)
	s.ByName[name] = &v
}

// Get gets weights from the set by name
func (s *Set) Get(name string) Meta {
	return s.ByName[name].Meta()
}

// Copy generates a copy of a set
func (s *Set) Copy() Set {
	n := NewSet()
	for i := range s.Weights {
		cp := s.Weights[i].Copy()
		n.Weights = append(n.Weights, &cp)
		n.ByName[cp.N] = &cp
	}
	return n
}

// Zero zeros the partial derivatives
func (s *Set) Zero() {
	for i := range s.Weights {
		s.Weights[i].Zero()
	}
}

func (s *Set) Save(file string, cost complex128, epoch int) error {
	set := pro.Set{
		CostReal:      real(cost),
		CostImaginary: imag(cost),
		Epoch:         uint64(epoch),
	}
	for _, w := range s.Weights {
		shape := make([]int64, len(w.S))
		for i := range shape {
			shape[i] = int64(w.S[i])
		}
		values, c := make([]float64, 2*len(w.X)), 0
		for _, x := range w.X {
			values[c] = real(x)
			c++
			values[c] = imag(x)
			c++
		}
		weights := pro.Weights{
			Name:   w.N,
			Shape:  shape,
			Values: values,
		}
		for _, state := range w.States {
			for _, x := range state {
				weights.States = append(weights.States, real(x))
				weights.States = append(weights.States, imag(x))
			}
		}
		set.Weights = append(set.Weights, &weights)
	}
	out, err := proto.Marshal(&set)
	if err != nil {
		return err
	}
	output, err := os.Create(file)
	if err != nil {
		return err
	}
	defer output.Close()
	_, err = output.Write(out)
	if err != nil {
		return err
	}
	return nil
}

// Open opens a set of weights
func (s *Set) Open(name string) (complex128, int, error) {
	in, err := ioutil.ReadFile(name)
	if err != nil {
		return 0, 0, err
	}
	set := pro.Set{}
	err = proto.Unmarshal(in, &set)
	if err != nil {
		return 0, 0, err
	}

	for _, w := range set.Weights {
		shape := make([]int, len(w.Shape))
		for i, s := range w.Shape {
			shape[i] = int(s)
		}
		x := make([]complex128, len(w.Values)/2)
		for i := 0; i < len(w.Values); i += 2 {
			x[i>>1] = complex(w.Values[i], w.Values[i+1])
		}
		v := V{
			N: w.Name,
			X: x,
			D: make([]complex128, len(w.Values)),
			S: shape,
		}
		width := len(v.X)
		for j := 0; j < len(w.States); j += width {
			x := make([]complex128, width)
			for i := 0; i < width; i += 2 {
				x[i>>1] = complex(w.States[j+i], w.States[j+i+1])
			}
			v.States = append(v.States, x)
		}
		s.Weights = append(s.Weights, &v)
		s.ByName[v.N] = &v
	}
	return complex(set.CostReal, set.CostImaginary), int(set.Epoch), nil
}

// Context is a function context
type Context struct {
	Quantize uint
	Node     int
	Cache    map[int][]complex128
}

// Clear clears the cache
func (c *Context) Clear() {
	c.Cache = make(map[int][]complex128)
}

// Get gets a value from the cache
func (c *Context) Get(node int) []complex128 {
	if c.Cache != nil {
		return c.Cache[node]
	}
	return nil
}

// Set sets a value in the cache
func (c *Context) Set(node int, value []complex128) {
	if c.Cache != nil {
		c.Cache[node] = value
	}
}

// Copy copies src tensors into dst
func (context *Context) Copy(k Continuation, node int, dst, src *V, options ...map[string]interface{}) bool {
	if len(src.S) != 2 || len(dst.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if (src.S[0] != dst.S[0]) || (src.S[1] != dst.S[1]) {
		panic("dimensions are not the same")
	}
	c := NewV(src.S...)
	c.X = append(c.X, src.X...)
	copy(dst.X, src.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		src.D[i] += j
		dst.D[i] += j
	}
	return false
}

// Add adds two tensors
func (context *Context) Add(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if a.Seed != 0 {
		dropout, index := uint32((1-a.Drop)*math.MaxUint32), 0
		c.Seed, c.Drop = a.Seed, a.Drop
		if cached == nil {
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
		}
		context.Set(node, c.X)
		if k(&c) {
			return true
		}
		index = 0
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
		return false
	}

	if cached == nil {
		for i, j := range a.X {
			c.X = append(c.X, j+b.X[i%length])
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j
		b.D[i%length] += j
	}
	return false
}

// Sub subtracts two tensors
func (context *Context) Sub(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i, j := range a.X {
			c.X = append(c.X, j-b.X[i%length])
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j
		b.D[i%length] -= j
	}
	return false
}

// Mul multiplies two tensors
func (context *Context) Mul(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] {
		panic("first dimension is not the same")
	}
	sizeA, sizeB, c, done :=
		len(a.X), len(b.X), NewV(a.S[1], b.S[1]), make(chan bool, 8)
	c.X = c.X[:cap(c.X)]
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if a.Seed != 0 {
		c.Seed, c.Drop = a.Seed, a.Drop
		dropout := uint32((1 - a.Drop) * math.MaxUint32)
		if cached == nil {
			mul := func(bv []complex128, i int) {
				rng := a.Seed
				for j := 0; j < sizeA; j += width {
					if rng.Next() > dropout {
						i++
						continue
					}

					av, sum := a.X[j:j+width], complex128(0.0)
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
		context.Set(node, c.X)
		if k(&c) {
			return true
		}

		done = make(chan bool, 8)

		// a derivatives
		go func() {
			derivativeDone := make(chan bool, 8)
			derivatives := func(index int, ad []complex128) {
				rows, bi := a.S[1], 0
				for i := 0; i < sizeB; i += width {
					bv, cd := b.X[i:i+width], c.D[index+bi*rows]

					for k, bx := range bv {
						ad[k] += bx * cd
					}

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
		derivatives := func(index int, bd []complex128) {
			rng := a.Seed
			for j := 0; j < sizeA; j += width {
				if rng.Next() > dropout {
					index++
					continue
				}
				av, cd := a.X[j:j+width], c.D[index]

				for k, ax := range av {
					bd[k] += ax * cd
				}

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

	if cached == nil {
		mul := func(bv []complex128, i int) {
			for j := 0; j < sizeA; j += width {
				av, sum := a.X[j:j+width], complex128(0.0)
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
	context.Set(node, c.X)
	if k(&c) {
		return true
	}

	done = make(chan bool, 8)

	// a derivatives
	go func() {
		derivativeDone := make(chan bool, 8)
		derivatives := func(index int, ad []complex128) {
			rows, bi := a.S[1], 0
			for i := 0; i < sizeB; i += width {
				bv, cd := b.X[i:i+width], c.D[index+bi*rows]

				for k, bx := range bv {
					ad[k] += bx * cd
				}

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
	derivatives := func(index int, bd []complex128) {
		for j := 0; j < sizeA; j += width {
			av, cd := a.X[j:j+width], c.D[index]

			for k, ax := range av {
				bd[k] += ax * cd
			}

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

// Hadamard computes the hadamard product of two tensors
func (context *Context) Hadamard(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := len(b.X)
	if a.S[0] != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i, j := range a.X {
			c.X = append(c.X, j*b.X[i%length])
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * b.X[i%length]
		b.D[i%length] += j * a.X[i]
	}
	return false
}

// T the transpose of the matrix
func (context *Context) T(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S[1], a.S[0])
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for p := 0; p < a.S[0]; p++ {
			for q := 0; q < a.S[1]; q++ {
				c.X = append(c.X, a.X[q*a.S[0]+p])
			}
		}
	}
	context.Set(node, c.X)
	if k(&c) {
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

// H the conjugate transpose of the matrix
func (context *Context) H(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S[1], a.S[0])
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for p := 0; p < a.S[0]; p++ {
			for q := 0; q < a.S[1]; q++ {
				c.X = append(c.X, cmplx.Conj(a.X[q*a.S[0]+p]))
			}
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	i := 0
	for p := 0; p < a.S[0]; p++ {
		for q := 0; q < a.S[1]; q++ {
			ax := cmplx.Conj(a.X[q*a.S[0]+p])
			x := real(ax)
			y := imag(ax)
			if x != 0 {
				x /= x
			}
			if y != 0 {
				y /= y
			}
			a.D[q*a.S[0]+p] += c.D[i] * complex(x, y)
			i++
		}
	}
	return false
}

// Slice a slice of the matrix
func (context *Context) Slice(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	width := a.S[0]
	begin, end := *options[0]["begin"].(*int), *options[0]["end"].(*int)
	dd, ok := options[0]["d"].(*int)
	d := 0
	if ok {
		d = *dd
	}
	if d == 0 {
		c := NewV(end-begin, 1)
		cached := context.Get(node)
		if cached != nil {
			c.X = cached
		}
		if cached == nil {
			av := a.X[begin:end]
			for _, ax := range av {
				c.X = append(c.X, ax)
			}
		}
		context.Set(node, c.X)
		if k(&c) {
			return true
		}
		index := 0
		ad := a.D[begin:end]
		for j := range ad {
			ad[j] += c.D[index]
			index++
		}
	} else if d == 2 {
		c, size := NewV(end-begin, a.S[1]), len(a.X)
		cached := context.Get(node)
		if cached != nil {
			c.X = cached
		}
		if cached == nil {
			for i := 0; i < size; i += width {
				av := a.X[i+begin : i+end]
				for _, ax := range av {
					c.X = append(c.X, ax)
				}
			}
		}
		context.Set(node, c.X)
		if k(&c) {
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
func (context *Context) Concat(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	widthA, widthB := a.S[0], b.S[0]
	c, i, j := NewV(widthA+widthB, a.S[1]), 0, 0
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for r := 0; r < a.S[1]; r++ {
			av, bv := a.X[i:i+widthA], b.X[j:j+widthB]
			c.X = append(c.X, av...)
			c.X = append(c.X, bv...)
			i += widthA
			j += widthB
		}
	}
	context.Set(node, c.X)
	if k(&c) {
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
func (context *Context) Dropout(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	size, width := len(a.X), a.S[0]
	rng := options[0]["rng"].(*rand.Rand)
	drop := .1
	if options[0]["drop"] != nil {
		drop = *options[0]["drop"].(*float64)
	}
	c, drops, factor := NewV(a.S...), make([]int, width), convert(1.0/(1.0-drop))
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
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				if drops[j] == 1 {
					c.X[i+j] = ax * factor
				}
			}
		}
	}
	context.Set(node, c.X)
	if k(&c) {
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
func (context *Context) Sin(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			c.X = append(c.X, sin(j))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * cos(a.X[i])
	}
	return false
}

// Cos the cosine of a tensor
func (context *Context) Cos(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			c.X = append(c.X, cos(j))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] -= j * sin(a.X[i])
	}
	return false
}

// Exp the base e exponential of a tensor
func (context *Context) Exp(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			c.X = append(c.X, exp(j))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * c.X[i]
	}
	return false
}

// Log the natural logarithm of a tensor
func (context *Context) Log(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			c.X = append(c.X, log(j))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j / a.X[i]
	}
	return false
}

// Sigmoid computes the sigmoid of a vector
func (context *Context) Sigmoid(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			e := exp(j)
			if isinf(e) {
				if sign(e) == 1 {
					c.X = append(c.X, 1)
				} else {
					c.X = append(c.X, 0)
				}
			} else {
				c.X = append(c.X, e/(e+1))
			}
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		cx := c.X[i]
		a.D[i] += j * cx * (1 - cx)
	}
	return false
}

// TanH the hyperbolic tangent of a tensor
func (context *Context) TanH(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			e1, e2 := exp(j), exp(-j)
			c.X = append(c.X, (e1-e2)/(e1+e2))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		cx := c.X[i]
		a.D[i] += j * (1 - cx*cx)
	}
	return false
}

// Softplus the softplus activation function
func (context *Context) Softplus(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			c.X = append(c.X, log(1+exp(j)))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j / (1 + exp(-a.X[i]))
	}
	return false
}

// Everett computes the split reality activation function
func (context *Context) Everett(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(2*a.S[0], a.S[1])
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if a.Seed != 0 {
		c.Seed, c.Drop = a.Seed, a.Drop
		index, dropout, factor := 0, uint32((1-a.Drop)*math.MaxUint32), complex(1/(1-a.Drop), 0)
		if cached == nil {
			for i := 0; i < a.S[1]; i++ {
				rng := a.Seed
				for j := 0; j < a.S[0]; j++ {
					if rng.Next() > dropout {
						c.X = append(c.X, 0, 0)
						index++
						continue
					}
					ax := a.X[index]
					rmin, rmax := real(ax), real(ax)
					if rmin > 0 {
						rmin = 0
					}
					if rmax < 0 {
						rmax = 0
					}
					imin, imax := imag(ax), imag(ax)
					if imin > 0 {
						imin = 0
					}
					if imax < 0 {
						imax = 0
					}
					c.X = append(c.X, complex(rmin, imin)*factor, complex(rmax, imax)*factor)
					index++
				}
			}
		}
		context.Set(node, c.X)
		if k(&c) {
			return true
		}
		index = 0
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

	if cached == nil {
		for _, j := range a.X {
			rmin, rmax := real(j), real(j)
			if rmin > 0 {
				rmin = 0
			}
			if rmax < 0 {
				rmax = 0
			}
			imin, imax := imag(j), imag(j)
			if imin > 0 {
				imin = 0
			}
			if imax < 0 {
				imax = 0
			}
			c.X = append(c.X, complex(rmin, imin), complex(rmax, imax))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		if c.X[i] != 0 || (c.X[i&^1] == 0 && c.X[i|1] == 0) {
			a.D[i>>1] += j
		}
	}
	return false
}

// Softmax is the softmax function
func (context *Context) Softmax(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c, size, width := NewV(a.S...), len(a.X), a.S[0]
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i := 0; i < size; i += width {
			sum := complex128(0.0)
			for _, ax := range a.X[i : i+width] {
				e := exp(ax)
				sum += e
				c.X = append(c.X, e)
			}
			for j, cx := range c.X[i : i+width] {
				c.X[i+j] = cx / sum
			}
		}
	}
	context.Set(node, c.X)
	if k(&c) {
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
func (context *Context) Sum(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c, sum := NewV(1), complex128(0.0)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			sum += j
		}
		c.X = append(c.X, sum)
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	d := c.D[0]
	for i := range a.D {
		a.D[i] += d
	}
	return false
}

// SumRows sums the rows of the matrix
func (context *Context) SumRows(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	size, width := len(a.X), a.S[0]
	c := NewV(width)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		c.X = c.X[:cap(c.X)]
		for i := 0; i < size; i += width {
			for j, ax := range a.X[i : i+width] {
				c.X[j] += ax
			}
		}
	}
	context.Set(node, c.X)
	if k(&c) {
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
func (context *Context) Quadratic(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := NewV(a.S[1]), len(a.X)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i := 0; i < size; i += width {
			av, bv, sum := a.X[i:i+width], b.X[i:i+width], complex128(0.0)
			for j, ax := range av {
				p := (ax - bv[j])
				sum += p * p
			}
			c.X = append(c.X, .5*sum)
		}
	}
	context.Set(node, c.X)
	if k(&c) {
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

// CrossEntropy computes the cross entropy cost of two tensors
func (context *Context) CrossEntropy(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := NewV(a.S[1]), len(a.X)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i := 0; i < size; i += width {
			av, bv, sum := a.X[i:i+width], b.X[i:i+width], complex128(0.0)
			for j, ax := range av {
				bx := bv[j]
				if bx == 1 {
					sum += log(ax + .001)
				} else {
					sum += log(1 - ax + .001)
				}
			}
			c.X = append(c.X, -sum)
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < size; i += width {
		av, bv, ad, bd, d := a.X[i:i+width], b.X[i:i+width], a.D[i:i+width], b.D[i:i+width], c.D[index]
		for j, ax := range av {
			bx := bv[j]
			if bx == 1 {
				ad[j] -= d / (ax + .001)
				bd[j] -= log(ax+.001) * d
			} else {
				ad[j] += d / (1 - ax + .001)
				bd[j] -= log(1-ax+.001) * d
			}
		}
		index++
	}
	return false
}

// Similarity computes the cosine similarity cost of two tensors
func (context *Context) Similarity(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	length := a.S[1]
	c, size := NewV(length), len(a.X)
	ab, aa, bb := make([]complex128, 0, length), make([]complex128, 0, length), make([]complex128, 0, length)
	for i := 0; i < size; i += width {
		av, bv := a.X[i:i+width], b.X[i:i+width]
		sumAB, sumAA, sumBB := complex128(0.0), complex128(0.0), complex128(0.0)
		for j, ax := range av {
			bx := bv[j]
			sumAB += ax * bx
			sumAA += ax * ax
			sumBB += bx * bx
		}
		c.X, ab, aa, bb =
			append(c.X, sumAB/(sqrt(sumAA)*sqrt(sumBB))), append(ab, sumAB), append(aa, sumAA), append(bb, sumBB)
	}
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < size; i += width {
		av, bv, ad, bd, cd := a.X[i:i+width], b.X[i:i+width], a.D[i:i+width], b.D[i:i+width], c.D[index]
		sumAB, sumAA, sumBB := ab[index], aa[index], bb[index]
		denominator := sqrt(sumAA) * sqrt(sumBB)
		for j, ax := range av {
			bx := bv[j]
			ad[j] += cd * (bx/denominator - ax*sumAB/(sumAA*denominator))
			bd[j] += cd * (ax/denominator - bx*sumAB/(sumBB*denominator))
		}
		index++
	}
	return false
}

// Orthogonality computes the cosine similarity between all vectros
func (context *Context) Orthogonality(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := ((a.S[1] - 1) * a.S[1]) / 2
	c, size, width := NewV(length), len(a.X), a.S[0]
	ab, aa, bb := make([]complex128, 0, length), make([]complex128, 0, length), make([]complex128, 0, length)
	for i := 0; i < size; i += width {
		for j := i + width; j < size; j += width {
			sumAB, sumAA, sumBB := complex128(0.0), complex128(0.0), complex128(0.0)
			for k := 0; k < width; k++ {
				a, b := a.X[i+k], a.X[j+k]
				sumAB += a * b
				sumAA += a * a
				sumBB += b * b
			}
			c.X, ab, aa, bb =
				append(c.X, sumAB/(sqrt(sumAA)*sqrt(sumBB))), append(ab, sumAB), append(aa, sumAA), append(bb, sumBB)
		}
	}
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < size; i += width {
		for j := i + width; j < size; j += width {
			cd, sumAB, sumAA, sumBB := c.D[index], ab[index], aa[index], bb[index]
			denominator := sqrt(sumAA) * sqrt(sumBB)
			for k := 0; k < width; k++ {
				ax, bx := a.X[i+k], a.X[j+k]
				a.D[i+k] += cd * (bx/denominator - ax*sumAB/(sumAA*denominator))
				a.D[j+k] += cd * (ax/denominator - bx*sumAB/(sumBB*denominator))
			}
			index++
		}
	}
	return false
}

// Entropy computes the entropy of the vectors
func (context *Context) Entropy(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	c, size, width := NewV(a.S[1]), len(a.X), a.S[0]
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i := 0; i < size; i += width {
			sum := complex128(0.0)
			for k := 0; k < width; k++ {
				ax := a.X[i+k]
				sum += ax * log(ax)
			}
			c.X = append(c.X, -sum)
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < size; i += width {
		cd := c.D[index]
		for k := 0; k < width; k++ {
			ax := a.X[i+k]
			a.D[i+k] -= cd * (log(ax) + 1)
		}
		index++
	}
	return false
}

// Variance computes the variance of the vectors
func (context *Context) Variance(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := a.S[1]
	c, size, width, means := NewV(length), len(a.X), a.S[0], make([]complex128, 0, length)

	n := complex128(complex(float64(width), 0))

	for i := 0; i < size; i += width {
		sum := complex128(0.0)
		for k := 0; k < width; k++ {
			sum += a.X[i+k]
		}
		mean := sum / n
		sum = complex128(0.0)
		for k := 0; k < width; k++ {
			d := a.X[i+k] - mean
			sum += d * d
		}
		c.X, means = append(c.X, sum/n), append(means, mean)
	}
	if k(&c) {
		return true
	}
	index, nn := 0, n*n
	for i := 0; i < size; i += width {
		cd, mean := c.D[index], means[index]
		for j := 0; j < width; j++ {
			sum := complex128(0.0)
			for k := 0; k < width; k++ {
				d := a.X[i+k] - mean
				if j == k {
					d *= (n - 1)
				} else {
					d *= -1
				}
				sum += d
			}
			a.D[i+j] += cd * 2 * sum / nn
		}
		index++
	}
	return false
}

// Abs computes the absolute value of the tensor
func (context *Context) Abs(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, ax := range a.X {
			c.X = append(c.X, abs(ax))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, cd := range c.D {

		a.D[i] += cd

	}
	return false
}

// Avg computes the average of the tensor
func (context *Context) Avg(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c, sum := NewV(1), complex128(0.0)

	total := complex128(complex(float64(len(a.X)), 0))

	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, j := range a.X {
			sum += j
		}
		c.X = append(c.X, sum/total)
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	d := c.D[0] / total
	for i := range a.D {
		a.D[i] += d
	}
	return false
}

// Combines two complex tensors to a complex tensor
func (context *Context) Complex(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for i, aX := range a.X {
			c.X = append(c.X, cmplx.Rect(cmplx.Abs(aX), cmplx.Phase(b.X[i%length])))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, cD := range c.D {
		aX, bX := cmplx.Abs(a.X[i]), cmplx.Phase(b.X[i%length])
		cos, sin := math.Cos(bX), math.Sin(bX)
		a.D[i] += cD * complex(cos+sin, 0)
		b.D[i%length] += cD * complex(aX*(cos-sin), 0)
	}
	return false
}

// Phase computes the phase of a complex tensor
func (context *Context) Phase(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(a.S...)
	cached := context.Get(node)
	if cached != nil {
		c.X = cached
	}
	if cached == nil {
		for _, ax := range a.X {
			c.X = append(c.X, complex(cmplx.Phase(ax), 0))
		}
	}
	context.Set(node, c.X)
	if k(&c) {
		return true
	}
	for i, cD := range c.D {
		ax := a.X[i]
		aa := real(ax)
		bb := imag(ax)
		cc := (aa*aa + bb*bb)
		a.D[i] += cD * complex((bb*bb-aa*aa)/(cc*cc), 0)
	}
	return false
}

// Op is a operation
func (context *Context) Op(op Operation) func(a ...Meta) Meta {
	return func(a ...Meta) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			var call func(a []Meta, b []*V) (bool, Continuation)
			call = func(a []Meta, b []*V) (bool, Continuation) {
				if len(a) == 0 {
					return op(k, node, b...), nil
				}
				derivatives := false
				continuation := a[0](func(c *V) bool {
					derivatives, _ = call(a[1:], append(b, c))
					return derivatives
				})
				return derivatives, continuation
			}
			_, continuation := call(a, make([]*V, 0, len(a)))
			return continuation
		}
	}
}

// B converts a binary function into an operator
func (context *Context) B(op Binary) func(a, b Meta, options ...map[string]interface{}) Meta {
	return func(a, b Meta, options ...map[string]interface{}) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(a *V) bool {
				derivatives := false
				b(func(b *V) bool {
					derivatives = op(k, node, a, b, options...)
					return derivatives
				})
				return derivatives
			})
		}
	}
}

// U converts a unary function into an operator
func (context *Context) U(op Unary) func(a Meta, options ...map[string]interface{}) Meta {
	return func(a Meta, options ...map[string]interface{}) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(b *V) bool {
				return op(k, node, b, options...)
			})
		}
	}
}

var (
	// Static is the static context
	Static Context
	// Op is a operation
	Op = Static.Op
	// B converts a binary function into an operator
	B = Static.B
	// U converts a unary function into an operator
	U = Static.U
	// Copy copies src tensors into dst
	Copy = B(Static.Copy)
	// Add adds two tensors
	Add = B(Static.Add)
	// Sub subtracts two tensors
	Sub = B(Static.Sub)
	// Mul multiplies two tensors
	Mul = B(Static.Mul)
	// Hadamard computes the hadamard product of two tensors
	Hadamard = B(Static.Hadamard)
	// T the transpose of the matrix
	T = U(Static.T)

	// H the conjugate transpose of the matrix
	H = U(Static.H)

	// Slice slices the matrix
	Slice = U(Static.Slice)
	// Concat concats two tensors
	Concat = B(Static.Concat)
	// Dropout is a dropout regularization function
	Dropout = U(Static.Dropout)
	// Sin the sin of a tensors
	Sin = U(Static.Sin)
	// Cos the cosine of a tensor
	Cos = U(Static.Cos)
	// Exp the base e exponential of a tensor
	Exp = U(Static.Exp)
	// Log the natural logarithm of a tensor
	Log = U(Static.Log)
	// Sigmoid the sigmoid of a tensors
	Sigmoid = U(Static.Sigmoid)
	// TanH the hyperbolic tangent of a tensor
	TanH = U(Static.TanH)
	// Softplus the softplus activation function
	Softplus = U(Static.Softplus)
	// Everett computes the split reality activation function
	Everett = U(Static.Everett)

	// Softmax is the softmax function
	Softmax = U(Static.Softmax)
	// Sum sums a vector
	Sum = U(Static.Sum)
	// SumRows sums the rows of the matrix
	SumRows = U(Static.SumRows)
	// Quadratic computes the quadratic cost of two tensors
	Quadratic = B(Static.Quadratic)
	// CrossEntropy computes the cross entropy cost of two tensors
	CrossEntropy = B(Static.CrossEntropy)
	// Similarity computes the cosine similarity cost of two tensors
	Similarity = B(Static.Similarity)
	// Orthogonality computes the cosine similarity between all vectros
	Orthogonality = U(Static.Orthogonality)
	// Entropy computes the entropy of the vectors
	Entropy = U(Static.Entropy)
	// Variance computes the variance of the vectors
	Variance = U(Static.Variance)
	// Abs computes the absolute value of the tensor
	Abs = U(Static.Abs)

	// Avg computes the average of the tensor
	Avg = U(Static.Avg)

	// Combines two complex tensors to a complex tensor
	Complex = B(Static.Complex)
	// Phase computes the phase of a complex tensor
	Phase = U(Static.Phase)
)

// Gradient computes the gradient
func Gradient(a Meta) (cost V) {
	a(func(a *V) bool {
		cost = *a
		a.D[0] = 1
		return false
	})
	return
}
