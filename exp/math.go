// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/cmplx"
)

type Math[T Number] interface {
	Abs() T
	Sin() T
	Cos() T
	Exp() T
	Log() T
	Sqrt() T
	IsInf() bool
	Sign() int
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

func (c C128) ISign() int {
	return 0
}
