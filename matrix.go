// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradient

import (
	"math"
	"math/cmplx"
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
	for i, j := range a.X {
		c.X = append(c.X, j+b.X[i%length])
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
	return c
}

// Square squares a tensor
func (a *V[T]) Square() *V[T] {
	b := a
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] {
		panic("first dimension is not the same")
	}
	sizeA, sizeB, c :=
		len(a.X), len(b.X), NewV[T](a.S[1], b.S[1])
	c.X = c.X[:cap(c.X)]
	mul := func(bv []T, i int) {
		for j := 0; j < sizeA; j += width {
			av, sum := a.X[j:j+width], T(0.0)
			for k, bx := range bv {
				sum += av[k] * bx
			}
			c.X[i] = sum
			i++
		}
	}
	index, step := 0, sizeA/width
	for i := 0; i < sizeB; i += width {
		mul(b.X[i:i+width], index)
		index += step
	}
	return c
}

// Hadamard computes the hadamard product of two tensors
func (a *V[T]) Hadamard(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := len(b.X)
	if a.S[0] != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV[T](a.S...)
	for i, j := range a.X {
		c.X = append(c.X, j*b.X[i%length])
	}
	return c
}

// T the transpose of the matrix
func (a *V[T]) T() *V[T] {
	c := NewV[T](a.S[1], a.S[0])
	for p := 0; p < a.S[0]; p++ {
		for q := 0; q < a.S[1]; q++ {
			c.X = append(c.X, a.X[q*a.S[0]+p])
		}
	}
	return c
}

// H the conjugate transpose of the matrix
func (a *V[T]) H() *V[T] {
	c := NewV[T](a.S[1], a.S[0])
	for p := 0; p < a.S[0]; p++ {
		for q := 0; q < a.S[1]; q++ {
			switch ax := any(a.X[q*a.S[0]+p]).(type) {
			case float32:
				c.X = append(c.X, any(ax).(T))
			case float64:
				c.X = append(c.X, any(ax).(T))
			case complex64:
				c.X = append(c.X, any(complex64(cmplx.Conj(complex128(ax)))).(T))
			case complex128:
				c.X = append(c.X, any(cmplx.Conj(ax)).(T))
			}
		}
	}
	return c
}

// Slice a slice of the matrix
func (a *V[T]) Slice(begin, end, d int) *V[T] {
	width := a.S[0]
	if d == 2 {
		c, size := NewV[T](end-begin, a.S[1]), len(a.X)
		for i := 0; i < size; i += width {
			av := a.X[i+begin : i+end]
			for _, ax := range av {
				c.X = append(c.X, ax)
			}
		}
		return c
	}

	c := NewV[T](end-begin, 1)
	av := a.X[begin:end]
	for _, ax := range av {
		c.X = append(c.X, ax)
	}
	return c
}

// Concat concats two tensors
func (a *V[T]) Concat(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	if a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	widthA, widthB := a.S[0], b.S[0]
	c, i, j := NewV[T](widthA+widthB, a.S[1]), 0, 0
	for r := 0; r < a.S[1]; r++ {
		av, bv := a.X[i:i+widthA], b.X[j:j+widthB]
		c.X = append(c.X, av...)
		c.X = append(c.X, bv...)
		i += widthA
		j += widthB
	}
	return c
}

// Dropout is a dropout regularization function
func (a *V[T]) Dropout(drop float64, drops []int) *V[T] {
	size, width := len(a.X), a.S[0]
	c, factor := NewV[T](a.S...), Convert[T](1.0/(1.0-drop))
	c.X = c.X[:cap(c.X)]
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			if drops[j] == 1 {
				c.X[i+j] = ax * factor
			}
		}
	}
	return c
}

// Sin the sine of a number
func (a *V[T]) Sin() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		c.X = append(c.X, Sin(j))
	}
	return c
}

// Cos the cosine of a tensor
func (a *V[T]) Cos() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		c.X = append(c.X, Cos(j))
	}
	return c
}

// Exp the base e exponential of a tensor
func (a *V[T]) Exp() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		c.X = append(c.X, Exp(j))
	}
	return c
}

// Log the natural logarithm of a tensor
func (a *V[T]) Log() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		c.X = append(c.X, Log(j))
	}
	return c
}

// Sqrt is the sqrt of a number
func (a *V[T]) Sqrt() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		c.X = append(c.X, Sqrt(j))
	}
	return c
}

// Inv is the inverse of a number
func (a *V[T]) Inv() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		if j == 0 {
			c.X = append(c.X, 0)
			continue
		}
		c.X = append(c.X, 1/j)
	}
	return c
}

// Sigmoid computes the sigmoid of a vector
func (a *V[T]) Sigmoid() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		e := Exp(j)
		if IsInf(e) {
			if Sign(e) == 1 {
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

// TanH the hyperbolic tangent of a tensor
func (a *V[T]) TanH() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		e1, e2 := Exp(j), Exp(-j)
		c.X = append(c.X, (e1-e2)/(e1+e2))
	}
	return c
}

// Softplus the softplus activation function
func (a *V[T]) Softplus() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		c.X = append(c.X, Log(1+Exp(j)))
	}
	return c
}

// Everett computes the split reality activation function
func (a *V[T]) Everett() *V[T] {
	c := NewV[T](2*a.S[0], a.S[1])
	for _, j := range a.X {
		switch tax := any(j).(type) {
		case float32:
			min, max := max(tax, 0), min(tax, 0)
			c.X = append(c.X, any(max).(T), any(min).(T))
		case float64:
			min, max := max(tax, 0), min(tax, 0)
			c.X = append(c.X, any(max).(T), any(min).(T))
		case complex64:
			rmin, rmax := max(real(tax), 0), min(real(tax), 0)
			imin, imax := max(imag(tax), 0), min(imag(tax), 0)
			c.X = append(c.X, any(complex(rmax, imax)).(T), any(complex(rmin, imin)).(T))
		case complex128:
			rmin, rmax := max(real(tax), 0), min(real(tax), 0)
			imin, imax := max(imag(tax), 0), min(imag(tax), 0)
			c.X = append(c.X, any(complex(rmax, imax)).(T), any(complex(rmin, imin)).(T))
		}
	}
	return c
}

// EverettReLu computes an adapter relu
func (a *V[T]) EverettReLu() *V[T] {
	c := NewV[T](2*a.S[0], a.S[1])
	for _, j := range a.X {
		switch tax := any(j).(type) {
		case float32:
			min := max(tax, 0)
			c.X = append(c.X, 0, any(min).(T))
		case float64:
			min := max(tax, 0)
			c.X = append(c.X, 0, any(min).(T))
		case complex64:
			rmin := max(real(tax), 0)
			imin := max(real(tax), 0)
			c.X = append(c.X, 0, any(complex(rmin, imin)).(T))
		case complex128:
			rmin := max(real(tax), 0)
			imin := max(real(tax), 0)
			c.X = append(c.X, 0, any(complex(rmin, imin)).(T))
		}
	}
	return c
}

// ReLu computes the rectified linear activation function
func (a *V[T]) ReLu() *V[T] {
	c := NewV[T](a.S...)
	for _, j := range a.X {
		switch tax := any(j).(type) {
		case float32:
			min := max(tax, 0)
			c.X = append(c.X, any(min).(T))
		case float64:
			min := max(tax, 0)
			c.X = append(c.X, any(min).(T))
		case complex64:
			rmin := max(real(tax), 0)
			imin := max(real(tax), 0)
			c.X = append(c.X, any(complex(rmin, imin)).(T))
		case complex128:
			rmin := max(real(tax), 0)
			imin := max(real(tax), 0)
			c.X = append(c.X, any(complex(rmin, imin)).(T))
		}
	}
	return c
}

// Softmax is the softmax function for big numbers
func (a *V[T]) Softmax(S float64) *V[T] {
	c, size, width := NewV[T](a.S...), len(a.X), a.S[0]
	s := Convert[T](S)
	switch any(s).(type) {
	case float32:
		var vv float32
		for _, v := range a.X {
			vv = max(vv, any(v).(float32))
		}
		s *= Convert[T](float64(vv))
	case float64:
		var vv float64
		for _, v := range a.X {
			vv = max(vv, any(v).(float64))
		}
		s *= Convert[T](float64(vv))
	}
	values := make([]T, width)
	for i := 0; i < size; i += width {
		sum := T(0.0)
		for j, ax := range a.X[i : i+width] {
			values[j] = Exp(ax - s)
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, cx/sum)
		}
	}
	return c
}

// Sum sums a vector
func (a *V[T]) Sum() *V[T] {
	c, sum := NewV[T](1), T(0.0)
	for _, j := range a.X {
		sum += j
	}
	c.X = append(c.X, sum)
	return c
}

// SumRows sums the rows of the matrix
func (a *V[T]) SumRows() *V[T] {
	size, width := len(a.X), a.S[0]
	c := NewV[T](width)
	c.X = c.X[:cap(c.X)]
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			c.X[j] += ax
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

// CrossEntropy computes the cross entropy cost of two tensors
func (a *V[T]) CrossEntropy(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, size := NewV[T](a.S[1]), len(a.X)
	for i := 0; i < size; i += width {
		av, bv, sum := a.X[i:i+width], b.X[i:i+width], T(0.0)
		for j, ax := range av {
			bx := bv[j]
			if bx == 1 {
				sum += Log(ax + .001)
			} else {
				sum += Log(1 - ax + .001)
			}
		}
		c.X = append(c.X, -sum)
	}
	return c
}

// Similarity computes the cosine similarity cost of two tensors
func (a *V[T]) Similarity(b *V[T]) (*V[T], []T, []T, []T) {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	length := a.S[1]
	c, size := NewV[T](length), len(a.X)
	ab, aa, bb := make([]T, 0, length), make([]T, 0, length), make([]T, 0, length)
	for i := 0; i < size; i += width {
		av, bv := a.X[i:i+width], b.X[i:i+width]
		sumAB, sumAA, sumBB := T(0.0), T(0.0), T(0.0)
		for j, ax := range av {
			bx := bv[j]
			sumAB += ax * bx
			sumAA += ax * ax
			sumBB += bx * bx
		}
		c.X, ab, aa, bb =
			append(c.X, sumAB/(Sqrt(sumAA)*Sqrt(sumBB))), append(ab, sumAB), append(aa, sumAA), append(bb, sumBB)
	}
	return c, ab, aa, bb
}

// Orthogonality computes the cosine similarity between all vectors
func (a *V[T]) Orthogonality() (*V[T], []T, []T, []T) {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := ((a.S[1] - 1) * a.S[1]) / 2
	c, size, width := NewV[T](length), len(a.X), a.S[0]
	ab, aa, bb := make([]T, 0, length), make([]T, 0, length), make([]T, 0, length)
	for i := 0; i < size; i += width {
		for j := i + width; j < size; j += width {
			sumAB, sumAA, sumBB := T(0.0), T(0.0), T(0.0)
			for k := 0; k < width; k++ {
				a, b := a.X[i+k], a.X[j+k]
				sumAB += a * b
				sumAA += a * a
				sumBB += b * b
			}
			c.X, ab, aa, bb =
				append(c.X, sumAB/(Sqrt(sumAA)*Sqrt(sumBB))), append(ab, sumAB), append(aa, sumAA), append(bb, sumBB)
		}
	}
	return c, ab, aa, bb
}

// Entropy computes the entropy of the vectors
func (a *V[T]) Entropy() *V[T] {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	c, size, width := NewV[T](a.S[1]), len(a.X), a.S[0]
	for i := 0; i < size; i += width {
		sum := T(0.0)
		for k := 0; k < width; k++ {
			ax := a.X[i+k]
			sum += ax * Log(ax)
		}
		c.X = append(c.X, -sum)
	}
	return c
}

// Variance computes the variance of the vectors
func (a *V[T]) Variance() (*V[T], []T) {
	if len(a.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	length := a.S[1]
	c, size, width, means := NewV[T](length), len(a.X), a.S[0], make([]T, 0, length)

	n := Convert[T](float64(width))

	for i := 0; i < size; i += width {
		sum := T(0.0)
		for k := 0; k < width; k++ {
			sum += a.X[i+k]
		}
		mean := sum / n
		sum = T(0.0)
		for k := 0; k < width; k++ {
			d := a.X[i+k] - mean
			sum += d * d
		}
		c.X, means = append(c.X, sum/n), append(means, mean)
	}
	return c, means
}

// Abs computes the absolute value of the tensor
func (a *V[T]) Abs() *V[T] {
	c := NewV[T](a.S...)
	for _, ax := range a.X {
		c.X = append(c.X, Abs(ax))
	}
	return c
}

// Quantize quantizes the values
func (a *V[T]) Quant(context *Context[T]) *V[T] {
	c := NewV[T](a.S...)
	for _, ax := range a.X {
		switch tax := any(ax).(type) {
		case float32:
			const (
				QuantizeMask = (1 << 32) - 1
				FractionBits = 23
			)
			if context.Quantize > FractionBits {
				panic("too much quantization")
			}
			bits := math.Float32bits(tax)
			bits &= QuantizeMask << context.Quantize
			c.X = append(c.X, any(math.Float32frombits(bits)).(T))
		case float64:
			const (
				QuantizeMask = (1 << 64) - 1
				FractionBits = 52
			)
			if context.Quantize > FractionBits {
				panic("too much quantization")
			}
			bits := math.Float64bits(tax)
			bits &= QuantizeMask << context.Quantize
			c.X = append(c.X, any(math.Float64frombits(bits)).(T))
		case complex64:
			rtax, itax := real(tax), imag(tax)
			const (
				QuantizeMask = (1 << 32) - 1
				FractionBits = 23
			)
			if context.Quantize > FractionBits {
				panic("too much quantization")
			}
			rbits := math.Float32bits(rtax)
			rbits &= QuantizeMask << context.Quantize
			ibits := math.Float32bits(itax)
			ibits &= QuantizeMask << context.Quantize
			c.X = append(c.X, any(complex(math.Float32frombits(rbits), math.Float32frombits(ibits))).(T))
		case complex128:
			rtax, itax := real(tax), imag(tax)
			const (
				QuantizeMask = (1 << 64) - 1
				FractionBits = 52
			)
			if context.Quantize > FractionBits {
				panic("too much quantization")
			}
			rbits := math.Float64bits(rtax)
			rbits &= QuantizeMask << context.Quantize
			ibits := math.Float64bits(itax)
			ibits &= QuantizeMask << context.Quantize
			c.X = append(c.X, any(complex(math.Float64frombits(rbits), math.Float64frombits(ibits))).(T))
		}
	}
	return c
}

// Avg computes the average of the tensor
func (a *V[T]) Avg() *V[T] {
	c, sum := NewV[T](1), T(0.0)
	total := Convert[T](float64(len(a.X)))
	for _, j := range a.X {
		sum += j
	}
	c.X = append(c.X, sum/total)
	return c
}

// Combines two complex tensors to a complex tensor
func (a *V[T]) Complex(b *V[T]) *V[T] {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width, length := a.S[0], len(b.X)
	if width != b.S[0] || (a.S[1] != b.S[1] && b.S[1] != 1) {
		panic("dimensions are not the same")
	}
	c := NewV[T](a.S...)
	for i, aX := range a.X {
		switch ax := any(aX).(type) {
		case complex64:
			r := cmplx.Abs(complex128(ax))
			p := cmplx.Phase(complex128(any(b.X[i%length]).(complex64)))
			c.X = append(c.X, any(complex64(cmplx.Rect(r, p))).(T))
		case complex128:
			c.X = append(c.X, any(cmplx.Rect(cmplx.Abs(ax), cmplx.Phase(any(b.X[i%length]).(complex128)))).(T))
		}
	}
	return c
}

// Phase computes the phase of a complex tensor
func (a *V[T]) Phase() *V[T] {
	c := NewV[T](a.S...)
	for _, ax := range a.X {
		switch ax := any(ax).(type) {
		case complex64:
			c.X = append(c.X, Convert[T](cmplx.Phase(complex128(ax))))
		case complex128:
			c.X = append(c.X, Convert[T](cmplx.Phase(ax)))
		}
	}
	return c
}
