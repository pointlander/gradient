// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tf64

import (
	"io/ioutil"
	"math"
	"os"

	"github.com/golang/protobuf/proto"

	pro "github.com/pointlander/gradient/tf64/proto_tf64"
)

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

type (
	// RNG is a random number generator
	RNG uint32
	// V is a tensor value
	V struct {
		N    string // the name
		Seed RNG
		Drop float64
		X    []float64 // the tensor
		D    []float64 // the derivative
		S    []int     // the shape
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
	Unary func(k Continuation, node int, a *V) bool
	// Binary is a binary function
	Binary func(k Continuation, node int, a, b *V) bool
	// Operation is an operation that takes multiple parameters
	Operation func(k Continuation, node int, a ...*V) bool
)

var (
	abs           = math.Abs
	sin           = math.Sin
	cos           = math.Cos
	exp           = math.Exp
	log           = math.Log
	sqrt          = math.Sqrt
	floatBits     = math.Float64bits
	floatFrombits = math.Float64frombits
)

const (
	QuantizeMask = (1 << 64) - 1
	FractionBits = 52
)

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
		X: make([]float64, 0, size),
		D: make([]float64, size),
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
		X: make([]float64, size),
		D: make([]float64, size),
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
		D: make([]float64, len(a.D)),
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
func (a *V) Set(values []float64) {
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

// Save saves a set of weights
func (s *Set) Save(file string, cost float64, epoch int) error {
	set := pro.Set{
		Cost:  float64(cost),
		Epoch: uint64(epoch),
	}
	for _, w := range s.Weights {
		shape := make([]int64, len(w.S))
		for i := range shape {
			shape[i] = int64(w.S[i])
		}
		weights := pro.Weights{
			Name:   w.N,
			Shape:  shape,
			Values: w.X,
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
func (s *Set) Open(name string) (float64, int, error) {
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
		v := V{
			N: w.Name,
			X: w.Values,
			D: make([]float64, len(w.Values)),
			S: shape,
		}
		s.Weights = append(s.Weights, &v)
		s.ByName[v.N] = &v
	}
	return float64(set.Cost), int(set.Epoch), nil
}

// Context is a function context
type Context struct {
	Quantize uint
	Node     int
}

// Add adds two tensors
func (context *Context) Add(k Continuation, node int, a, b *V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV(a.S...)
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

	for i, j := range a.X {
		c.X = append(c.X, j+b.X[i%length])
	}
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
func (context *Context) Sub(k Continuation, node int, a, b *V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV(a.S...)
	for i, j := range a.X {
		c.X = append(c.X, j-b.X[i%length])
	}
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
func (context *Context) Mul(k Continuation, node int, a, b *V) bool {
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
	if a.Seed != 0 {
		c.Seed, c.Drop = a.Seed, a.Drop
		dropout := uint32((1 - a.Drop) * math.MaxUint32)
		mul := func(bv []float64, i int) {
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
		if k(&c) {
			return true
		}

		done = make(chan bool, 8)

		// a derivatives
		go func() {
			derivativeDone := make(chan bool, 8)
			derivatives := func(index int, ad []float64) {
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
		derivatives := func(index int, bd []float64) {
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

	mul := func(bv []float64, i int) {
		for j := 0; j < sizeA; j += width {
			av, sum := a.X[j:j+width], float64(0.0)
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
	if k(&c) {
		return true
	}

	done = make(chan bool, 8)

	// a derivatives
	go func() {
		derivativeDone := make(chan bool, 8)
		derivatives := func(index int, ad []float64) {
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
	derivatives := func(index int, bd []float64) {
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

// Hadamard computes the hadamard product of two tensors
func (context *Context) Hadamard(k Continuation, node int, a, b *V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := len(b.X)
	if a.S[0] != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV(a.S...)
	for i, j := range a.X {
		c.X = append(c.X, j*b.X[i%length])
	}
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
func (context *Context) T(k Continuation, node int, a *V) bool {
	c := NewV(a.S[1], a.S[0])
	for p := 0; p < a.S[0]; p++ {
		for q := 0; q < a.S[1]; q++ {
			c.X = append(c.X, a.X[q*a.S[0]+p])
		}
	}

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

// T the transpose of the matrix
func (context *Context) Slice(k Continuation, node int, a *V, b *V) bool {
	if b.S[0] != 2 && b.S[1] != 1 {
		panic("invalid size for slice")
	}
	width := a.S[0]

	begin, end := int(b.X[0]), int(b.X[1])

	c, size := NewV(end-begin, a.S[1]), len(a.X)
	for i := 0; i < size; i += width {
		av := a.X[i+begin : i+end]
		for _, ax := range av {
			c.X = append(c.X, ax)
		}
	}
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
	return false
}

// Concat concats two tensors
func (context *Context) Concat(k Continuation, node int, a, b *V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	widthA, widthB := a.S[0], b.S[0]
	c, i, j := NewV(widthA+widthB, a.S[1]), 0, 0
	for r := 0; r < a.S[1]; r++ {
		av, bv := a.X[i:i+widthA], b.X[j:j+widthB]
		c.X = append(c.X, av...)
		c.X = append(c.X, bv...)
		i += widthA
		j += widthB
	}
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

// Sin the sine of a number
func (context *Context) Sin(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, j := range a.X {
		c.X = append(c.X, sin(j))
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * cos(a.X[i])
	}
	return false
}

// Cos the cosine of a tensor
func (context *Context) Cos(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, j := range a.X {
		c.X = append(c.X, cos(j))
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] -= j * sin(a.X[i])
	}
	return false
}

// Exp the base e exponential of a tensor
func (context *Context) Exp(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, j := range a.X {
		c.X = append(c.X, exp(j))
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j * c.X[i]
	}
	return false
}

// Log the natural logarithm of a tensor
func (context *Context) Log(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, j := range a.X {
		c.X = append(c.X, log(j))
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j / a.X[i]
	}
	return false
}

// Sigmoid computes the sigmoid of a vector
func (context *Context) Sigmoid(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, j := range a.X {
		e := exp(j)
		c.X = append(c.X, e/(e+1))
	}
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
func (context *Context) TanH(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, j := range a.X {
		e1, e2 := exp(j), exp(-j)
		c.X = append(c.X, (e1-e2)/(e1+e2))
	}
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
func (context *Context) Softplus(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, j := range a.X {
		c.X = append(c.X, log(1+exp(j)))
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		a.D[i] += j / (1 + exp(-a.X[i]))
	}
	return false
}

// Everett computes the split reality activation function
func (context *Context) Everett(k Continuation, node int, a *V) bool {
	c := NewV(2*a.S[0], a.S[1])
	if a.Seed != 0 {
		c.Seed, c.Drop = a.Seed, a.Drop
		index, dropout, factor := 0, uint32((1-a.Drop)*math.MaxUint32), float64(1/(1-a.Drop))
		for i := 0; i < a.S[1]; i++ {
			rng := a.Seed
			for j := 0; j < a.S[0]; j++ {
				if rng.Next() > dropout {
					c.X = append(c.X, 0, 0)
					index++
					continue
				}
				ax := a.X[index]
				min, max := ax, ax
				if min > 0 {
					min = 0
				}
				if max < 0 {
					max = 0
				}
				c.X = append(c.X, min*factor, max*factor)
				index++
			}
		}
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

	for _, j := range a.X {
		min, max := j, j
		if min > 0 {
			min = 0
		}
		if max < 0 {
			max = 0
		}
		c.X = append(c.X, min, max)
	}
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

func (context *Context) EverettReLu(k Continuation, node int, a *V) bool {
	c := NewV(2*a.S[0], a.S[1])
	for _, j := range a.X {
		max := j
		if max < 0 {
			max = 0
		}
		c.X = append(c.X, 0, max)
	}
	if k(&c) {
		return true
	}
	for i, j := range c.D {
		if c.X[i] != 0 {
			a.D[i>>1] += j
		}
	}
	return false
}

// Softmax is the softmax function
func (context *Context) Softmax(k Continuation, node int, a *V) bool {
	c, size, width := NewV(a.S...), len(a.X), a.S[0]
	for i := 0; i < size; i += width {
		sum := float64(0.0)
		for _, ax := range a.X[i : i+width] {
			e := exp(ax)
			sum += e
			c.X = append(c.X, e)
		}
		for j, cx := range c.X[i : i+width] {
			c.X[i+j] = cx / sum
		}
	}
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		a.D[i] += d * (cx - cx*cx)
	}
	return false
}

// Sum sums a vector
func (context *Context) Sum(k Continuation, node int, a *V) bool {
	c, sum := NewV(1), float64(0.0)
	for _, j := range a.X {
		sum += j
	}
	c.X = append(c.X, sum)
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
func (context *Context) SumRows(k Continuation, node int, a *V) bool {
	size, width := len(a.X), a.S[0]
	c := NewV(width)
	c.X = c.X[:cap(c.X)]
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			c.X[j] += ax
		}
	}
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
func (context *Context) Quadratic(k Continuation, node int, a, b *V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := NewV(a.S[1]), len(a.X)
	for i := 0; i < size; i += width {
		av, bv, sum := a.X[i:i+width], b.X[i:i+width], float64(0.0)
		for j, ax := range av {
			p := (ax - bv[j])
			sum += p * p
		}
		c.X = append(c.X, .5*sum)
	}
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
func (context *Context) CrossEntropy(k Continuation, node int, a, b *V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := NewV(a.S[1]), len(a.X)
	for i := 0; i < size; i += width {
		av, bv, sum := a.X[i:i+width], b.X[i:i+width], float64(0.0)
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
func (context *Context) Similarity(k Continuation, node int, a, b *V) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	length := a.S[1]
	c, size := NewV(length), len(a.X)
	ab, aa, bb := make([]float64, 0, length), make([]float64, 0, length), make([]float64, 0, length)
	for i := 0; i < size; i += width {
		av, bv := a.X[i:i+width], b.X[i:i+width]
		sumAB, sumAA, sumBB := float64(0.0), float64(0.0), float64(0.0)
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
func (context *Context) Orthogonality(k Continuation, node int, a *V) bool {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := ((a.S[1] - 1) * a.S[1]) / 2
	c, size, width := NewV(length), len(a.X), a.S[0]
	ab, aa, bb := make([]float64, 0, length), make([]float64, 0, length), make([]float64, 0, length)
	for i := 0; i < size; i += width {
		for j := i + width; j < size; j += width {
			sumAB, sumAA, sumBB := float64(0.0), float64(0.0), float64(0.0)
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
func (context *Context) Entropy(k Continuation, node int, a *V) bool {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	c, size, width := NewV(a.S[1]), len(a.X), a.S[0]
	for i := 0; i < size; i += width {
		sum := float64(0.0)
		for k := 0; k < width; k++ {
			ax := a.X[i+k]
			sum += ax * log(ax)
		}
		c.X = append(c.X, -sum)
	}
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
func (context *Context) Variance(k Continuation, node int, a *V) bool {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := a.S[1]
	c, size, width, means := NewV(length), len(a.X), a.S[0], make([]float64, 0, length)

	n := float64(width)

	for i := 0; i < size; i += width {
		sum := float64(0.0)
		for k := 0; k < width; k++ {
			sum += a.X[i+k]
		}
		mean := sum / n
		sum = float64(0.0)
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
			sum := float64(0.0)
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
func (context *Context) Abs(k Continuation, node int, a *V) bool {
	c := NewV(a.S...)
	for _, ax := range a.X {
		c.X = append(c.X, abs(ax))
	}
	if k(&c) {
		return true
	}
	for i, cd := range c.D {

		sign := float64(1)
		if a.X[i] < 0 {
			sign = -1
		}
		a.D[i] += cd * sign

	}
	return false
}

// Quantize quantizes the values
func (context *Context) Quant(k Continuation, node int, a *V) bool {
	if context.Quantize > FractionBits {
		panic("too much quantization")
	}
	c := NewV(a.S...)
	for _, ax := range a.X {
		bits := floatBits(ax)
		bits &= QuantizeMask << context.Quantize
		c.X = append(c.X, floatFrombits(bits))
	}
	if k(&c) {
		return true
	}
	for i, cd := range c.D {
		a.D[i] += cd
	}
	return false
}

// Avg computes the average of the tensor
func (context *Context) Avg(k Continuation, node int, a *V) bool {
	c, sum := NewV(1), float64(0.0)
	for _, j := range a.X {
		sum += j
	}

	total := float64(len(a.X))

	c.X = append(c.X, sum/total)
	if k(&c) {
		return true
	}
	d := c.D[0] / total
	for i := range a.D {
		a.D[i] += d
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
func (context *Context) B(op Binary) func(a, b Meta) Meta {
	return func(a, b Meta) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(a *V) bool {
				derivatives := false
				b(func(b *V) bool {
					derivatives = op(k, node, a, b)
					return derivatives
				})
				return derivatives
			})
		}
	}
}

// U converts a unary function into an operator
func (context *Context) U(op Unary) func(a Meta) Meta {
	return func(a Meta) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(b *V) bool {
				return op(k, node, b)
			})
		}
	}
}

var (
	// Static is the static context
	Static Context
	// Add adds two tensors
	Add = Static.B(Static.Add)
	// Sub subtracts two tensors
	Sub = Static.B(Static.Sub)
	// Mul multiplies two tensors
	Mul = Static.B(Static.Mul)
	// Hadamard computes the hadamard product of two tensors
	Hadamard = Static.B(Static.Hadamard)
	// T the transpose of the matrix
	T = Static.U(Static.T)
	// Slice slices the matrix
	Slice = Static.B(Static.Slice)
	// Concat concats two tensors
	Concat = Static.B(Static.Concat)
	// Sin the sin of a tensors
	Sin = Static.U(Static.Sin)
	// Cos the cosine of a tensor
	Cos = Static.U(Static.Cos)
	// Exp the base e exponential of a tensor
	Exp = Static.U(Static.Exp)
	// Log the natural logarithm of a tensor
	Log = Static.U(Static.Log)
	// Sigmoid the sigmoid of a tensors
	Sigmoid = Static.U(Static.Sigmoid)
	// TanH the hyperbolic tangent of a tensor
	TanH = Static.U(Static.TanH)
	// Softplus the softplus activation function
	Softplus = Static.U(Static.Softplus)

	// Everett computes the split reality activation function
	Everett     = Static.U(Static.Everett)
	EverettReLu = Static.U(Static.EverettReLu)

	// Softmax is the softmax function
	Softmax = Static.U(Static.Softmax)
	// Sum sums a vector
	Sum = Static.U(Static.Sum)
	// SumRows sums the rows of the matrix
	SumRows = Static.U(Static.SumRows)
	// Quadratic computes the quadratic cost of two tensors
	Quadratic = Static.B(Static.Quadratic)
	// CrossEntropy computes the cross entropy cost of two tensors
	CrossEntropy = Static.B(Static.CrossEntropy)
	// Similarity computes the cosine similarity cost of two tensors
	Similarity = Static.B(Static.Similarity)
	// Orthogonality computes the cosine similarity between all vectros
	Orthogonality = Static.U(Static.Orthogonality)
	// Entropy computes the entropy of the vectors
	Entropy = Static.U(Static.Entropy)
	// Variance computes the variance of the vectors
	Variance = Static.U(Static.Variance)
	// Abs computes the absolute value of the tensor
	Abs = Static.U(Static.Abs)

	// Quantize quantizes the values
	Quant = Static.U(Static.Quant)

	// Avg computes the average of the tensor
	Avg = Static.U(Static.Avg)
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
