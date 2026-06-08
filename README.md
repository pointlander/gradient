[![Go Reference](https://pkg.go.dev/badge/github.com/pointlander/gradient.svg)](https://pkg.go.dev/github.com/pointlander/gradient)

# Reverse Mode Automatic Differentiation with Continuation Passing Style
This project implements [reverse mode automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) with [continuation passing style](https://en.wikipedia.org/wiki/Continuation-passing_style) (CPS). Tensors are supported with number type of float32, float64, complex64, or complex128. Gradient was inspired by [Lantern](https://feiwang3311.github.io/Lantern/) as described by this [paper](http://papers.nips.cc/paper/8221-backpropagation-with-callbacks-foundations-for-efficient-and-expressive-differentiable-programming.pdf).

# Feedfoward example
```go
data := [...][3]float64{
	{0, 0, 0},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 0},
}
context := Context[float64]{}
set := context.NewSet()
set.Add("w0", 2, 8)
set.AddBias("b0", 8)
set.Add("w1", 16, 1)
set.AddBias("b1", 1)
set.AddData("input", 2, 4)
set.AddData("output", 1, 4)

rng := rand.New(rand.NewSource(2))
set.InitAdam(rng)
input := set.ByName["input"]
output := set.ByName["output"]
index0, index1 := 0, 0
for _, d := range data {
	input.X[index0] = d[0]
	index0++
	input.X[index0] = d[1]
	index0++
	output.X[index1] = d[2]
	index1++
}

Add := context.B(context.Add)
Mul := context.B(context.Mul)
Everett := context.U(context.Everett)
Sigmoid := context.U(context.Sigmoid)
Quadratic := context.B(context.Quadratic)
Avg := context.U(context.Avg)
l0 := Everett(Add(Mul(set.Get("w0"), set.Get("input")), set.Get("b0")))
l1 := Sigmoid(Add(Mul(set.Get("w1"), l0), set.Get("b1")))
loss := Avg(Quadratic(l1, set.Get("output")))
for range 33 {
	set.Zero()
	l := Gradient(loss)
	set.Adam(B1, B2, .1)
	t.Log(l)
}

l1(func(a *V[float64]) bool {
	x := a.X
	t.Log(x)
	for i, d := range data {
		if d[2] == 0 && x[i] > .5 {
			t.Fatal("incorrect", 0, x[i])
		} else if d[2] == 1 && x[i] <= .5 {
			t.Fatal("incorrect", 1, x[i])
		}
	}
	return true
})
```
