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

func convert[T Number](x float64) T {
	switch any(x).(type) {
	case float32:
		switch v := any(float32(x)).(type) {
		case T:
			return v
		}
	case float64:
		switch v := any(x).(type) {
		case T:
			return v
		}
	case complex64:
		switch v := any(complex(float32(x), 0)).(type) {
		case T:
			return v
		}
	case complex128:
		switch v := any(complex(x, 0)).(type) {
		case T:
			return v
		}
	default:
		return 0
	}
	return 0
}
