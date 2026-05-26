// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exp

import (
	"math"
	"math/cmplx"
)

func Dot[T Number](a, b []T) T {
	var sum T
	for i, value := range a {
		sum += value * b[i]
	}
	return sum
}

func Axpy[T Number](alpha T, X []T, Y []T) {
	for i, y := range Y {
		Y[i] = alpha * (X[i] + y)
	}
}

func Abs[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		return any(float32(math.Abs(float64(v)))).(T)
	case float64:
		return any(math.Abs(v)).(T)
	case complex64:
		return any(complex64(complex(float32(cmplx.Abs(complex128(v))), 0))).(T)
	case complex128:
		return any(complex128(complex(cmplx.Abs(v), 0))).(T)
	default:
		panic("invalid type")
	}
}

func Sin[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		return any(float32(math.Sin(float64(v)))).(T)
	case float64:
		return any(math.Sin(v)).(T)
	case complex64:
		return any(complex64(cmplx.Sin(complex128(v)))).(T)
	case complex128:
		return any(complex128(cmplx.Sin(v))).(T)
	default:
		panic("invalid type")
	}
}

func Cos[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		return any(float32(math.Cos(float64(v)))).(T)
	case float64:
		return any(math.Cos(v)).(T)
	case complex64:
		return any(complex64(cmplx.Cos(complex128(v)))).(T)
	case complex128:
		return any(complex128(cmplx.Cos(v))).(T)
	default:
		panic("invalid type")
	}
}

func Exp[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		return any(float32(math.Exp(float64(v)))).(T)
	case float64:
		return any(math.Exp(v)).(T)
	case complex64:
		return any(complex64(cmplx.Exp(complex128(v)))).(T)
	case complex128:
		return any(complex128(cmplx.Exp(v))).(T)
	default:
		panic("invalid type")
	}
}

func Log[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		return any(float32(math.Log(float64(v)))).(T)
	case float64:
		return any(math.Log(v)).(T)
	case complex64:
		return any(complex64(cmplx.Log(complex128(v)))).(T)
	case complex128:
		return any(complex128(cmplx.Log(v))).(T)
	default:
		panic("invalid type")
	}
}

func Sqrt[T Number](x T) T {
	switch v := any(x).(type) {
	case float32:
		return any(float32(math.Sqrt(float64(v)))).(T)
	case float64:
		return any(math.Sqrt(v)).(T)
	case complex64:
		return any(complex64(cmplx.Sqrt(complex128(v)))).(T)
	case complex128:
		return any(complex128(cmplx.Sqrt(v))).(T)
	default:
		panic("invalid type")
	}
}

func Pow[T Number](x, y T) T {
	switch xx := any(x).(type) {
	case float32:
		return any(float32(math.Pow(float64(xx), float64(any(y).(float32))))).(T)
	case float64:
		return any(math.Pow(xx, any(y).(float64))).(T)
	case complex64:
		return any(complex64(cmplx.Pow(complex128(xx), complex128(any(y).(complex64))))).(T)
	case complex128:
		return any(complex128(cmplx.Pow(xx, any(y).(complex128)))).(T)
	default:
		panic("invalid type")
	}
}

func IsInf[T Number](x T) bool {
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
		panic("invalid type")
	}
}

func IsNaN[T Number](x T) bool {
	switch v := any(x).(type) {
	case float32:
		return math.IsNaN(float64(v))
	case float64:
		return math.IsNaN(v)
	case complex64:
		return cmplx.IsNaN(complex128(v))
	case complex128:
		return cmplx.IsNaN(v)
	default:
		panic("invalid type")
	}
}

func Sign[T Number](x T) int {
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
		panic("invalid type")
	}
}

func Convert[T Number](x float64) T {
	switch any(x).(type) {
	case float32:
		return any(float32(x)).(T)
	case float64:
		return any(x).(T)
	case complex64:
		return any(complex(float32(x), 0)).(T)
	case complex128:
		return any(complex(x, 0)).(T)
	default:
		panic("invalid type")
	}
}
