package gradient

import (
	"fmt"
	"testing"
)

func TestGradient(t *testing.T) {
	v1, v2 := V{0.5, 0}, V{0.4, 0}
	v6 := func(a *V) {
		a.D = 1
	}
	v5 := func(a *V) {
		TanH(a)(v6)
	}
	v4 := func(a *V) {
		Mul(a, &v2)(v5)
	}
	v3 := Add(&v1, &v2)
	v3(v4)

	w1, w2 := V{0.5, 0}, V{0.4, 0}
	Gradient(TanHOp(MulOp(w2.Value(), AddOp(w1.Value(), w2.Value()))))

	if fmt.Sprintf("%f", w1.D) != "0.352331" {
		t.Fatalf("w1(%f) != 0.352331", w1.D)
	} else if fmt.Sprintf("%f", w2.D) != "1.145075" {
		t.Fatalf("w1(%f) != 1.145075", w2.D)
	} else if v1.D != w1.D {
		t.Fatalf("v1(%f) != w1(%f)", v1.D, w1.D)
	} else if v2.D != w2.D {
		t.Fatalf("v2(%f) != w2(%f)", v2.D, w1.D)
	}
}
