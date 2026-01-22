// Copyright 2026 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!noasm && arm) || (!noasm && arm64)
// +build !noasm,arm !noasm,arm64

package tf32

import (
	"unsafe"
)

func dot(x, y []float32) (z float32) {
	vdot(unsafe.Pointer(&x[0]), unsafe.Pointer(&y[0]), unsafe.Pointer(uintptr(len(x))), unsafe.Pointer(&z))
	return z
}

func axpy(alpha float32, X []float32, Y []float32) {
	for i, y := range Y {
		Y[i] = alpha*X[i] + y
	}
}
