// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package float

import (
	"fmt"
	"testing"
)

func TestGradient(t *testing.T) {
	var context = Context{
		Precision: 64,
	}
	v1, v2 := V{}, V{}
	v1.X.SetFloat64(0.5).SetPrec(64)
	v1.D.SetPrec(64)
	v2.X.SetFloat64(0.4).SetPrec(64)
	v2.D.SetPrec(64)
	v6 := func(a *V) bool {
		a.D.SetFloat64(1)
		return false
	}
	v5 := func(a *V) bool {
		return context.TanH(v6, a)
	}
	v4 := func(a *V) bool {
		return context.Mul(v5, a, &v2)
	}
	context.Add(v4, &v1, &v2)

	w1, w2 := V{}, V{}
	w1.X.SetFloat64(0.5).SetPrec(64)
	w1.D.SetPrec(64)
	w2.X.SetFloat64(0.4).SetPrec(64)
	w2.D.SetPrec(64)
	Gradient(TanH(Mul(w2.Meta(), Add(w1.Meta(), w2.Meta()))))

	if fmt.Sprintf("%s", w1.D.String()) != "0.3523309083" {
		t.Fatalf("w1(%s) != 0.3523309083", w1.D.String())
	} else if fmt.Sprintf("%s", w2.D.String()) != "1.145075452" {
		t.Fatalf("w1(%s) != 1.145075452", w2.D.String())
	} else if v1.D.String() != w1.D.String() {
		t.Fatalf("v1(%s) != w1(%s)", v1.D.String(), w1.D.String())
	} else if v2.D.String() != w2.D.String() {
		t.Fatalf("v2(%s) != w2(%s)", v2.D.String(), w1.D.String())
	}
}
