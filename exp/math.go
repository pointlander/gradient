// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/cmplx"
)

func dot[T Number](a, b []T) T {
	var sum T
	for i, value := range a {
		sum += value * b[i]
	}
	return sum
}

func axpy[T Number](alpha T, X []T, Y []T) {
	for i, y := range Y {
		Y[i] = alpha * (X[i] + y)
	}
}

func abs[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		switch v := any(float32(math.Abs(float64(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case float64:
		switch v := any(math.Abs(v)).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex64:
		switch v := any(complex64(complex(float32(cmplx.Abs(complex128(v))), 0))).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex128:
		switch v := any(complex128(complex(cmplx.Abs(v), 0))).(type) {
		case T:
			return v
		default:
			return x
		}
	default:
		return x
	}
}

func sin[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		switch v := any(float32(math.Sin(float64(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case float64:
		switch v := any(math.Sin(v)).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex64:
		switch v := any(complex64(cmplx.Sin(complex128(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex128:
		switch v := any(complex128(cmplx.Sin(v))).(type) {
		case T:
			return v
		default:
			return x
		}
	default:
		return x
	}
}

func cos[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		switch v := any(float32(math.Cos(float64(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case float64:
		switch v := any(math.Cos(v)).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex64:
		switch v := any(complex64(cmplx.Cos(complex128(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex128:
		switch v := any(complex128(cmplx.Cos(v))).(type) {
		case T:
			return v
		default:
			return x
		}
	default:
		return x
	}
}

func exp[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		switch v := any(float32(math.Exp(float64(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case float64:
		switch v := any(math.Exp(v)).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex64:
		switch v := any(complex64(cmplx.Exp(complex128(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex128:
		switch v := any(complex128(cmplx.Exp(v))).(type) {
		case T:
			return v
		default:
			return x
		}
	default:
		return x
	}
}

func log[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		switch v := any(float32(math.Log(float64(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case float64:
		switch v := any(math.Log(v)).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex64:
		switch v := any(complex64(cmplx.Log(complex128(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex128:
		switch v := any(complex128(cmplx.Log(v))).(type) {
		case T:
			return v
		default:
			return x
		}
	default:
		return x
	}
}

func sqrt[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		switch v := any(float32(math.Sqrt(float64(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case float64:
		switch v := any(math.Sqrt(v)).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex64:
		switch v := any(complex64(cmplx.Sqrt(complex128(v)))).(type) {
		case T:
			return v
		default:
			return x
		}
	case complex128:
		switch v := any(complex128(cmplx.Sqrt(v))).(type) {
		case T:
			return v
		default:
			return x
		}
	default:
		return x
	}
}

func isinf[T Number](x T) bool {
	switch v := any(x).(type) {
	case float32:
		return math.IsInf(float64(v), 0)
	case float64:
		return math.IsInf(v, 0)
	case complex64:
		return cmplx.IsInf(complex128(v))
	case complex128:
		return cmplx.IsInf(v)
	default:
		return false
	}
}

func sign[T Number](x T) int {
	switch v := any(x).(type) {
	case float32:
		switch true {
		case v > 0:
			return 1
		case v < 0:
			return -1
		default:
			return 0
		}
	case float64:
		switch true {
		case v > 0:
			return 1
		case v < 0:
			return -1
		default:
			return 0
		}
	case complex64:
		return 0
	case complex128:
		return 0
	default:
		return 0
	}
}

type Math[T Number] interface {
	Set(T) Math[T]
	Add(Math[T]) Math[T]
	Sub(Math[T]) Math[T]
	Mul(Math[T]) Math[T]
	Div(Math[T]) Math[T]
	Abs() Math[T]
	Sin() Math[T]
	Cos() Math[T]
	Exp() Math[T]
	Log() Math[T]
	Sqrt() Math[T]
	IsInf() bool
	Sign() int
}

// Add
func (f F32) Set(a F32) F32 {
	return a
}

func (f F64) Set(a F64) F64 {
	return a
}

func (c C64) Set(a C64) C64 {
	return a
}

func (c C128) Set(a C128) C128 {
	return a
}

// Add
func (f F32) Add(a F32) F32 {
	return f + a
}

func (f F64) Add(a F64) F64 {
	return f + a
}

func (c C64) Add(a C64) C64 {
	return c + a
}

func (c C128) Add(a C128) C128 {
	return c + a
}

// Sub
func (f F32) Sub(a F32) F32 {
	return f - a
}

func (f F64) Sub(a F64) F64 {
	return f - a
}

func (c C64) Sub(a C64) C64 {
	return c - a
}

func (c C128) Sub(a C128) C128 {
	return c - a
}

// Mul
func (f F32) Mul(a F32) F32 {
	return f * a
}

func (f F64) Mul(a F64) F64 {
	return f * a
}

func (c C64) Mul(a C64) C64 {
	return c * a
}

func (c C128) Mul(a C128) C128 {
	return c * a
}

// Div
func (f F32) Div(a F32) F32 {
	return f / a
}

func (f F64) Div(a F64) F64 {
	return f / a
}

func (c C64) Div(a C64) C64 {
	return c / a
}

func (c C128) Div(a C128) C128 {
	return c / a
}

// Abs
func (f F32) Abs() F32 {
	return F32(math.Abs(float64(f)))
}

func (f F64) Abs() F64 {
	return F64(math.Abs(float64(f)))
}

func (c C64) Abs() C64 {
	return C64(complex(float32(cmplx.Abs(complex128(c))), 0))
}

func (c C128) Abs() C128 {
	return C128(complex(cmplx.Abs(complex128(c)), 0))
}

// Sin
func (f F32) Sin() F32 {
	return F32(math.Sin(float64(f)))
}

func (f F64) Sin() F64 {
	return F64(math.Sin(float64(f)))
}

func (c C64) Sin() C64 {
	return C64(cmplx.Sin(complex128(c)))
}

func (c C128) Sin() C128 {
	return C128(cmplx.Sin(complex128(c)))
}

// Cos
func (f F32) Cos() F32 {
	return F32(math.Cos(float64(f)))
}

func (f F64) Cos() F64 {
	return F64(math.Cos(float64(f)))
}

func (c C64) Cos() C64 {
	return C64(cmplx.Cos(complex128(c)))
}

func (c C128) Cos() C128 {
	return C128(cmplx.Cos(complex128(c)))
}

// Exp
func (f F32) Exp() F32 {
	return F32(math.Exp(float64(f)))
}

func (f F64) Exp() F64 {
	return F64(math.Exp(float64(f)))
}

func (c C64) Exp() C64 {
	return C64(cmplx.Exp(complex128(c)))
}

func (c C128) Exp() C128 {
	return C128(cmplx.Exp(complex128(c)))
}

// Log
func (f F32) Log() F32 {
	return F32(math.Log(float64(f)))
}

func (f F64) Log() F64 {
	return F64(math.Log(float64(f)))
}

func (c C64) Log() C64 {
	return C64(cmplx.Log(complex128(c)))
}

func (c C128) Log() C128 {
	return C128(cmplx.Log(complex128(c)))
}

// Log
func (f F32) Sqrt() F32 {
	return F32(math.Sqrt(float64(f)))
}

func (f F64) Sqrt() F64 {
	return F64(math.Sqrt(float64(f)))
}

func (c C64) Sqrt() C64 {
	return C64(cmplx.Sqrt(complex128(c)))
}

func (c C128) Sqrt() C128 {
	return C128(cmplx.Sqrt(complex128(c)))
}

// IsInf
func (f F32) IsInf() bool {
	return math.IsInf(float64(f), 0)
}

func (f F64) IsInf() bool {
	return math.IsInf(float64(f), 0)
}

func (c C64) IsInf() bool {
	return cmplx.IsInf(complex128(c))
}

func (c C128) IsInf() bool {
	return cmplx.IsInf(complex128(c))
}

// Sign
func (f F32) Sign() int {
	switch true {
	case f > 0:
		return 1
	case f < 0:
		return -1
	default:
		return 0
	}
}

func (f F64) Sign() int {
	switch true {
	case f > 0:
		return 1
	case f < 0:
		return -1
	default:
		return 0
	}
}

func (c C64) Sign() int {
	return 0
}

func (c C128) Sign() int {
	return 0
}
