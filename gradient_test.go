package gradient

import (
	"fmt"
	"math/rand"
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

func TestXORNetwork(t *testing.T) {
	rand.Seed(1)
	random64 := func(a, b float64) float64 {
		return (b-a)*rand.Float64() + a
	}
	type Weight struct {
		V
		Delta float64
	}

	i1, i2, o, weights := V{}, V{}, V{}, [6]Weight{}
	for i := range weights {
		weights[i].X = random64(-1, 1)
	}
	n1 := SigmoidOp(AddOp(MulOp(i1.Value(), weights[0].Value()), MulOp(i2.Value(), weights[1].Value())))
	n2 := SigmoidOp(AddOp(MulOp(i1.Value(), weights[2].Value()), MulOp(i2.Value(), weights[3].Value())))
	n3 := SigmoidOp(AddOp(MulOp(n1, weights[4].Value()), MulOp(n2, weights[5].Value())))
	d := SubOp(n3, o.Value())
	cost := MulOp(d, d)

	data := [...][3]float64{
		{0, 0, 0},
		{1, 0, 1},
		{0, 1, 1},
		{1, 1, 0},
	}
	alpha, eta := .4, .6
	for i := 0; i < 1000; i++ {
		for i, sample := range data {
			j := i + rand.Intn(len(data)-i)
			data[i], data[j] = data[j], sample
		}
		total := 0.0
		for j := range data {
			i1.X, i2.X, o.X, i1.D, i2.D, o.D = 0, 0, 0, data[j][0], data[j][1], data[j][2]
			total += Gradient(cost).X
			for k := range weights {
				weights[k].Delta, weights[k].D = alpha*weights[k].Delta-eta*weights[k].D, 0
				weights[k].X += weights[k].Delta
			}
		}
		t.Log(i, total)
		if total < .001 {
			break
		}
	}
	for i := range data {
		i1.D, i2.D = data[i][0], data[i][1]
		var output V
		n3(func(a *V) {
			output = *a
		})
		if output.X > .5 && data[i][2] != 1 {
			t.Fatal("output should be 1")
		} else if output.X <= .5 && data[i][2] != 0 {
			t.Fatal("output should be 0")
		}
	}
}
