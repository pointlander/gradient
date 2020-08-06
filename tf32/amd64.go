// Copyright 2020 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

package tf32

import (
	"github.com/ziutek/blas"
)

func dot(X, Y []float32) float32 {
	return blas.Sdot(len(X), X, 1, Y, 1)
}
