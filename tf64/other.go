// Copyright 2020 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 arm

package tf64

func dot(X, Y []float64) float64 {
	var sum float64
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}
