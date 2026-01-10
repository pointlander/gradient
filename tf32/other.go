// Copyright 2020 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || arm || arm64 || wasm
// +build 386 arm arm64 wasm

package tf32

func dot(X, Y []float32) float32 {
	var sum float32
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

func axpy(alpha float32, X []float32, Y []float32) {
	for i, y := range Y {
		Y[i] = alpha*X[i] + y
	}
}
