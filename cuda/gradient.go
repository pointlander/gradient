// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cuda

import (
	"fmt"
	"os"
)

type (
	// V is a tensor value
	V struct {
		N string // the name
		S []int  // the shape
	}
	// Set is a set of V
	Set struct {
		Weights []*V
		ByName  map[string]*V
	}
	// Continuation is a continuation
	Continuation func(a *V) bool
	// Meta is a function that takes a continuation and return a continuation
	Meta func(k Continuation) Continuation
	// Unary is a unary function
	Unary func(k Continuation, node int, a *V, options ...map[string]interface{}) bool
	// Binary is a binary function
	Binary func(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool
	// Operation is an operation that takes multiple parameters
	Operation func(k Continuation, node int, a ...*V) bool
)

// Panic marks a place we should never get to
func Panic(a *V) bool {
	panic("should not be here")
}

// Meta returns a meta for the value
func (a *V) Meta() Meta {
	return func(k Continuation) Continuation {
		k(a)
		return Panic
	}
}

// NewSet creates a new weight set
func NewSet() Set {
	return Set{
		ByName: make(map[string]*V),
	}
}

// Add adds weights to a set
func (s *Set) Add(context *Context, name string, d ...int) {
	v := NewV(context.Node, d...)
	v.N = name
	s.Weights = append(s.Weights, &v)
	s.ByName[name] = &v
}

// Get gets weights from the set by name
func (s *Set) Get(name string) Meta {
	return s.ByName[name].Meta()
}

// Context is a function context
type Context struct {
	Output *os.File
	Node   int
}

// NewV create a new tensor value
func NewV(node int, s ...int) V {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	return V{
		N: fmt.Sprintf("node%d", node),
		S: s,
	}
}

// Allocate generates the code which allocates the variable
func (v *V) Allocate(output *os.File) {
	fmt.Fprintf(output, "\tint *device_%s = 0;\n", v.N)
	fmt.Fprintf(output, "\tCHECK(cudaMalloc((void**)&device_%s, %d * sizeof(float)));\n", v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tCHECK(cudaMemset(device_%s, 0, %d * sizeof(float)));\n", v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tint *device_%s_d = 0;\n", v.N)
	fmt.Fprintf(output, "\tCHECK(cudaMalloc((void**)&device_%s_d, %d * sizeof(float)));\n", v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tCHECK(cudaMemset(device_%s_d, 0, %d * sizeof(float)));\n", v.N, v.S[0]*v.S[1])
}

// Free generates the code to free the variable
func (v *V) Free(output *os.File) {
	fmt.Fprintf(output, "\tCHECK(cudaFree(device_%s_d));\n", v.N)
	fmt.Fprintf(output, "\tCHECK(cudaFree(device_%s));\n", v.N)
}

// Mul multiplies two tensors
func (context *Context) Mul(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] {
		panic("first dimension is not the same")
	}
	c := NewV(node, a.S[1], b.S[1])
	M := a.S[1]
	N := b.S[1]

	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlock(16, 16); 
	dim3 blocksPerGrid((%d + threadsPerBlock.x - 1) / threadsPerBlock.x, (%d + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mul<<<blocksPerGrid, threadsPerBlock>>>((float *)device_%s, (float *)device_%s, (float *)device_%s, %d, %d, %d);
`, N, M, a.N, b.N, c.N, N, width, M)

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlocka(16, 16); 
	dim3 blocksPerGrida((%d + threadsPerBlock.x - 1) / threadsPerBlock.x, (%d + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mul_ad<<<blocksPerGrida, threadsPerBlocka>>>((float *)device_%s_d, (float *)device_%s, (float *)device_%s_d, %d, %d, %d, %d, %d);
`, a.S[1], a.S[0], c.N, b.N, a.N, N, width, M, a.S[1], a.S[0])

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockb(16, 16); 
	dim3 blocksPerGridb((%d + threadsPerBlock.x - 1) / threadsPerBlock.x, (%d + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mul_bd<<<blocksPerGridb, threadsPerBlockb>>>((float *)device_%s_d, (float *)device_%s, (float *)device_%s_d, %d, %d, %d, %d, %d);
`, b.S[1], b.S[0], c.N, a.N, b.N, N, width, M, b.S[1], b.S[0])

	return false
}

// Add adds two tensors
func (context *Context) Add(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] {
		panic("first dimension is not the same")
	}
	c := NewV(node, a.S[0], a.S[1])

	c.Allocate(context.Output)
	defer c.Free(context.Output)

	if k(&c) {
		return true
	}

	return false
}

// Everett computes the split reality activation function
func (context *Context) Everett(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, 2*a.S[0], a.S[1])

	c.Allocate(context.Output)
	defer c.Free(context.Output)

	if k(&c) {
		return true
	}

	return false
}

// Quadratic computes the quadratic cost of two tensors
func (context *Context) Quadratic(k Continuation, node int, a, b *V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c := NewV(node, a.S[1])

	c.Allocate(context.Output)
	defer c.Free(context.Output)

	if k(&c) {
		return true
	}

	return false
}

// Avg computes the average of the tensor
func (context *Context) Avg(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, 1)

	c.Allocate(context.Output)
	defer c.Free(context.Output)

	if k(&c) {
		return true
	}

	return false
}

// T the transpose of the matrix
func (context *Context) T(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, a.S[1], a.S[0])

	c.Allocate(context.Output)
	defer c.Free(context.Output)

	if k(&c) {
		return true
	}

	return false
}

// Dropout is a dropout regularization function
func (context *Context) Dropout(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	drop := .1
	_ = drop
	if len(options) > 0 && options[0]["drop"] != nil {
		drop = *options[0]["drop"].(*float64)
	}
	c := NewV(node, a.S[0], a.S[1])

	c.Allocate(context.Output)
	defer c.Free(context.Output)

	if k(&c) {
		return true
	}

	return false
}

// Op is a operation
func (context *Context) Op(op Operation) func(a ...Meta) Meta {
	return func(a ...Meta) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			var call func(a []Meta, b []*V) (bool, Continuation)
			call = func(a []Meta, b []*V) (bool, Continuation) {
				if len(a) == 0 {
					fmt.Fprintf(context.Output, "\t{\n")
					c := op(k, node, b...)
					fmt.Fprintf(context.Output, "\t}\n")
					return c, nil
				}
				derivatives := false
				continuation := a[0](func(c *V) bool {
					derivatives, _ = call(a[1:], append(b, c))
					return derivatives
				})
				return derivatives, continuation
			}
			_, continuation := call(a, make([]*V, 0, len(a)))
			return continuation
		}
	}
}

// B converts a binary function into an operator
func (context *Context) B(op Binary) func(a, b Meta, options ...map[string]interface{}) Meta {
	return func(a, b Meta, options ...map[string]interface{}) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(a *V) bool {
				derivatives := false
				b(func(b *V) bool {
					fmt.Fprintf(context.Output, "\t{\n")
					derivatives = op(k, node, a, b, options...)
					fmt.Fprintf(context.Output, "\t}\n")
					return derivatives
				})
				return derivatives
			})
		}
	}
}

// U converts a unary function into an operator
func (context *Context) U(op Unary) func(a Meta, options ...map[string]interface{}) Meta {
	return func(a Meta, options ...map[string]interface{}) Meta {
		node := context.Node
		context.Node++
		return func(k Continuation) Continuation {
			return a(func(b *V) bool {
				fmt.Fprintf(context.Output, "\t{\n")
				derivatives := op(k, node, b, options...)
				fmt.Fprintf(context.Output, "\t}\n")
				return derivatives
			})
		}
	}
}

// Gradient computes the gradient
func (context *Context) Gradient(set Set, a Meta) (cost V) {
	fmt.Fprintf(context.Output, `#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void mul(float* a, float* b, float* c, int n, int width, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < m) && (col < n)) {
		float sum = 0.0f;
		for (int k = 0; k < width; k++) {
			sum += a[row * width + k] * b[col * width + k];
		}
		c[col * width + row] = sum;
	}
}
__global__ void mul_ad(float* cd, float* b, float* ad, int n, int width, int m, int r, int c) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < r) && (col < c)) {
		float sum = 0;
		for (int i = 0; i < n; i++) {
			sum += cd[row+i*m]*b[i*width+col];
		}
		ad[row*width+col] += sum;
	}
}
__global__ void mul_bd(float* cd, float* a, float* bd, int n, int width, int m, int r, int c) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < r) && (col < c)) {
		float sum = 0;
		for (int i = 0; i < m; i++) {
			sum += cd[i + row*m]*a[i*width+col];
		}
		bd[row*width+col] += sum;
	}
}


#define CHECK(err) check(__FILE__, __LINE__, __func__, (err))

void check(const char* file, int line, const char* func, cudaError_t err) {
	if (err == cudaSuccess) {
		return;
	}
	printf("[ERROR] %%s:%%d: in function %%s - %%s\n", file, line, func, cudaGetErrorString(err));
	exit(1);
}

void callback(float* output, int w, int h);

struct V {
	int W;
	int H;
	float *X;
	float *D;	
};
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "struct V %s;\n", value.N)
	}

	fmt.Fprintf(context.Output, `void init(void) {
`)
	fmt.Fprintf(context.Output, `
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\t%s.W = %d;\n", value.N, value.S[0])
		fmt.Fprintf(context.Output, "\t%s.H = %d;\n", value.N, value.S[1])
		fmt.Fprintf(context.Output, "\t%s.X = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\t%s.D = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
	}
	fmt.Fprintf(context.Output, `}
void uninit(void) {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tfree(%s.X);\n", value.N)
		fmt.Fprintf(context.Output, "\tfree(%s.D);\n", value.N)
	}
	fmt.Fprintf(context.Output, `
}
int gradient(void) {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tint *device_%s = 0;\n", value.N)
		fmt.Fprintf(context.Output, "\tCHECK(cudaMalloc((void**)&device_%s, %d * sizeof(float)));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(device_%s, %s.X, %d * sizeof(float), cudaMemcpyHostToDevice));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tint *device_%s_d = 0;\n", value.N)
		fmt.Fprintf(context.Output, "\tCHECK(cudaMalloc((void**)&device_%s_d, %d * sizeof(float)));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(device_%s_d, %s.D, %d * sizeof(float), cudaMemcpyHostToDevice));\n",
			value.N, value.N, value.S[0]*value.S[1])
	}
	a(func(a *V) bool {
		fmt.Fprintf(context.Output, "\t{\n")
		fmt.Fprintf(context.Output, "\tfloat* host_%s = (float*)calloc(%d, sizeof(float));\n", a.N, a.S[0]*a.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(host_%s, device_%s, %d * sizeof(float), cudaMemcpyDeviceToHost));\n",
			a.N, a.N, a.S[0]*a.S[1])
		fmt.Fprintf(context.Output, "\tcallback(host_%s, %d, %d);\n", a.N, a.S[0], a.S[1])
		fmt.Fprintf(context.Output, `	for (int i = 0; i < %d; i++) {
			host_%s[i] = 1.0;
	}
`, a.S[0]*a.S[1], a.N)
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(device_%s_d, host_%s, %d * sizeof(float), cudaMemcpyHostToDevice));\n",
			a.N, a.N, a.S[0]*a.S[1])
		fmt.Fprintf(context.Output, "\tfree(host_%s);\n", a.N)
		fmt.Fprintf(context.Output, "\t}\n")
		return false
	})
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tCHECK(cudaFree(device_%s));\n", value.N)
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(%s.D, device_%s_d, %d * sizeof(float), cudaMemcpyDeviceToHost));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaFree(device_%s_d));\n", value.N)
	}
	fmt.Fprintf(context.Output, `
	return 0;
}
`)
	return
}
