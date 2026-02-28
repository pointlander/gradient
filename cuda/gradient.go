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
		Skip bool
		Set  int
		N    string // the name
		S    []int  // the shape
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
	fmt.Fprintf(output, "\tfloat *device_%s = 0;\n", v.N)
	fmt.Fprintf(output, "\tCHECK(cudaMalloc((void**)&device_%s, %d * sizeof(float)));\n", v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tCHECK(cudaMemset(device_%s, 0, %d * sizeof(float)));\n", v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tfloat *device_%s_d = 0;\n", v.N)
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
	N := a.S[1]
	M := b.S[1]

	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((%d + threadsPerBlock.x - 1) / threadsPerBlock.x, (%d + threadsPerBlock.y - 1) / threadsPerBlock.y);
    mul<<<blocksPerGrid, threadsPerBlock>>>((float *)device_%s, (float *)device_%s, (float *)device_%s, %d, %d, %d, %d);
`, N, M, a.N, b.N, c.N, N, M, width, N)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlocka(16, 16);
	dim3 blocksPerGrida((%d + threadsPerBlocka.x - 1) / threadsPerBlocka.x, (%d + threadsPerBlocka.y - 1) / threadsPerBlocka.y);
    mul_ad<<<blocksPerGrida, threadsPerBlocka>>>((float *)device_%s_d, (float *)device_%s, (float *)device_%s_d, %d, %d, %d, %d, %d);
`, a.S[1], a.S[0], c.N, b.N, a.N, width, a.S[1], a.S[0], a.S[1], b.S[1])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockb(16, 16);
	dim3 blocksPerGridb((%d + threadsPerBlockb.x - 1) / threadsPerBlockb.x, (%d + threadsPerBlockb.y - 1) / threadsPerBlockb.y);
    mul_bd<<<blocksPerGridb, threadsPerBlockb>>>((float *)device_%s_d, (float *)device_%s, (float *)device_%s_d, %d, %d, %d, %d, %d);
`, a.S[1], b.S[0], c.N, a.N, b.N, width, b.S[1], b.S[0], b.S[0], a.S[1])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

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
	M := a.S[1]
	N := a.S[0]

	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((%d + threadsPerBlock.x - 1) / threadsPerBlock.x, (%d + threadsPerBlock.y - 1) / threadsPerBlock.y);
    add<<<blocksPerGrid, threadsPerBlock>>>((float *)device_%s, (float *)device_%s, (float *)device_%s, %d, %d);
`, N, M, a.N, b.N, c.N, N, M)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlocka(16, 16);
	dim3 blocksPerGrida((%d + threadsPerBlocka.x - 1) / threadsPerBlocka.x, (%d + threadsPerBlocka.y - 1) / threadsPerBlocka.y);
    add_ad<<<blocksPerGrida, threadsPerBlocka>>>((float *)device_%s_d, (float *)device_%s_d, %d, %d);
`, a.S[0], a.S[1], c.N, a.N, N, M)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockb(16);
	dim3 blocksPerGridb((%d + threadsPerBlockb.x - 1) / threadsPerBlockb.x);
    add_bd<<<blocksPerGridb, threadsPerBlockb>>>((float *)device_%s_d, (float *)device_%s_d, %d, %d);
`, b.S[0], c.N, b.N, N, M)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	return false
}

// Everett computes the split reality activation function
func (context *Context) Everett(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, 2*a.S[0], a.S[1])

	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlock(16);
	dim3 blocksPerGrid((%d + threadsPerBlock.x - 1) / threadsPerBlock.x);
	everett<<<blocksPerGrid, threadsPerBlock>>>((float *)device_%s, (float *)device_%s, %d);
`, a.S[0]*a.S[1], a.N, c.N, a.S[0]*a.S[1])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockd(16);
	dim3 blocksPerGridd((%d + threadsPerBlockd.x - 1) / threadsPerBlockd.x);
	everett_d<<<blocksPerGridd, threadsPerBlockd>>>((float *)device_%s, (float *)device_%s_d, (float *)device_%s_d, %d);
`, c.S[0]*c.S[1], c.N, c.N, a.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

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
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlock(16);
	dim3 blocksPerGrid((%d + threadsPerBlock.x - 1) / threadsPerBlock.x);
	quadratic<<<blocksPerGrid, threadsPerBlock>>>((float *)device_%s, (float *)device_%s, (float *)device_%s, %d, %d);
`, c.S[0], c.N, a.N, b.N, c.S[0], width)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockd(16, 16);
	dim3 blocksPerGridd((%d + threadsPerBlockd.x - 1) / threadsPerBlockd.x, (%d + threadsPerBlockd.y - 1) / threadsPerBlockd.y);
	quadratic_d<<<blocksPerGridd, threadsPerBlockd>>>((float *)device_%s, (float *)device_%s, (float *)device_%s_d, (float *)device_%s_d, (float *)device_%s_d, %d, %d, %d);
`, a.S[0], a.S[1], a.N, b.N, c.N, a.N, b.N, a.S[0], a.S[1], width)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	return false
}

// Avg computes the average of the tensor
func (context *Context) Avg(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, 1)
	n := a.S[0] * a.S[1]

	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	int threadsPerBlock = 256;
	int blocksPerGrid = (%d + (threadsPerBlock * 2 - 1)) / (threadsPerBlock * 2);
	size_t sharedMemSize = threadsPerBlock * sizeof(float);
	float *d_odata;
	CHECK(cudaMalloc(&d_odata, blocksPerGrid * sizeof(float)));
	reduce<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>((float *)device_%s, (float *)d_odata, %d);
`, n, a.N, n)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")
	fmt.Fprintf(context.Output, `	float *h_odata = (float*)malloc(blocksPerGrid * sizeof(float));
	CHECK(cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
	float sum = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		sum += h_odata[i];
	}
	free(h_odata);
	CHECK(cudaFree(d_odata));
	sum /= ((float)%d);
	CHECK(cudaMemcpy((float *)device_%s, &sum, sizeof(float), cudaMemcpyHostToDevice));
`, n, c.N)

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockd(16);
	dim3 blocksPerGridd((%d + threadsPerBlockd.x - 1) / threadsPerBlockd.x);
	avg_d<<<blocksPerGridd, threadsPerBlockd>>>((float *)device_%s_d, (float *)device_%s_d, %d);
`, n, a.N, c.N, n)
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	return false
}

// T the transpose of the matrix
func (context *Context) T(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, a.S[1], a.S[0])

	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((%d + threadsPerBlock.x - 1) / threadsPerBlock.x, (%d + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose<<<blocksPerGrid, threadsPerBlock>>>((float *)device_%s, (float *)device_%s, %d, %d);
`, a.S[1], a.S[0], a.N, c.N, a.S[1], a.S[0])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockd(16, 16);
	dim3 blocksPerGridd((%d + threadsPerBlockd.x - 1) / threadsPerBlockd.x, (%d + threadsPerBlockd.y - 1) / threadsPerBlockd.y);
    transpose_d<<<blocksPerGridd, threadsPerBlockd>>>((float *)device_%s_d, (float *)device_%s_d, %d, %d);
`, a.S[1], a.S[0], c.N, a.N, a.S[1], a.S[0])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

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
	fmt.Fprintf(context.Output, "\tstatic uint lfsr = 1;\n")
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlock(16);
	dim3 blocksPerGrid((%d + threadsPerBlock.x - 1) / threadsPerBlock.x);
	dropout<<<blocksPerGrid, threadsPerBlock>>>((float *)device_%s, (float *)device_%s, %f, lfsr, %d);
`, c.S[0]*c.S[1], a.N, c.N, drop, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")

	if k(&c) {
		return true
	}
	fmt.Fprintf(context.Output, `	dim3 threadsPerBlockd(16);
	dim3 blocksPerGridd((%d + threadsPerBlockd.x - 1) / threadsPerBlockd.x);
	dropout<<<blocksPerGridd, threadsPerBlockd>>>((float *)device_%s_d, (float *)device_%s_d, %f, lfsr, %d);
`, c.S[0]*c.S[1], c.N, a.N, drop, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")
	fmt.Fprintf(context.Output, "\tconst uint LFSRMask = 0x80000057;\n")
	fmt.Fprintf(context.Output, "\tlfsr = (lfsr >> 1) ^ (-(lfsr & 1) & LFSRMask);\n")

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

__global__ void mul(float* a, float* b, float* c, int n, int m, int width, int aw) {
	int ai = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	if ((bi < m) && (ai < n)) {
		float sum = 0.0f;
		for (int k = 0; k < width; k++) {
			sum += a[ai * width + k] * b[bi * width + k];
		}
		c[bi * aw +  ai] = sum;
	}
}
__global__ void mul_ad(float* cd, float* b, float* ad, int width, int r, int c, int cols, int rows) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if ((row < r) && (col < c)) {
		float sum = 0;
		for (int i = 0; i < rows; i++) {
			sum += cd[row+i*cols]*b[i*width+col];
		}
		ad[row*width+col] += sum;
	}
}
__global__ void mul_bd(float* cd, float* a, float* bd, int width, int r, int c, int cols, int rows) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	if ((row < r) && (col < c)) {
		float sum = 0;
		for (int i = 0; i < rows; i++) {
			sum += cd[i + row*cols]*a[i*width+col];
		}
		bd[row*width+col] += sum;
	}
}
__global__ void add(float* a, float* b, float* c, int n, int m) {
	int ai = blockIdx.x * blockDim.x + threadIdx.x;
	int bi = blockIdx.y * blockDim.y + threadIdx.y;
	if ((ai < n) && (bi < m)) {
		c[bi * n + ai] = a[bi * n + ai] + b[ai];
	}
}
__global__ void add_ad(float* cd, float* ad, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < m) && (col < n)) {
		ad[row * n + col] += cd[row * n + col];
	}
}
__global__ void add_bd(float* cd, float* bd, int n, int m) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n) {
		float sum = 0;
		for (int i = 0; i < m; i++) {
			sum += cd[i * n + col];
		}
		bd[col] += sum;
	}
}
__global__ void everett(float* a, float* c, int n) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n) {
		const int idx2 = 2*col;
		const float in = a[col];
		if (in > 0.0f) {
			c[idx2] = 0.0f;
		} else {
			c[idx2] = in;
		}
		if (in < 0.0f) {
			c[idx2+1] = 0.0f;
		} else {
			c[idx2+1] = in;
		}
	}
}
__global__ void everett_d(float *c, float* cd, float* ad, int n) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n) {
		const int idxA = col&~1U;
		const int idxB = col|1U;
		if ((c[col] != 0) || ((c[idxA] == 0) && (c[idxB] == 0))) {
			ad[col>>1] += cd[col];
		}
	}
}
__global__ void quadratic(float* c, float* a, float* b, int n, int width) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n) {
		float sum = 0;
		for (int i = 0; i < width; i++) {
			float diff = a[col*width + i] - b[col*width + i];
			sum += diff*diff;
		}
		c[col] = sum;
	}
}
__global__ void quadratic_d(float* a, float* b, float* cd, float* ad, float* bd, int n, int m, int width) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((row < m) && (col < n)) {
		const float d = cd[row];
		ad[row*width + col] += (a[row*width + col] - b[row*width + col]) * d;
		bd[row*width + col] += (b[row*width + col] - a[row*width + col]) * d;
	}
}
__global__ void reduce(float* g_idata, float* g_odata, unsigned int n) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	float mySum = (i < n) ? g_idata[i] : 0;
	if (i + blockDim.x < n) {
		mySum += g_idata[i + blockDim.x];
	}
	sdata[tid] = mySum;
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = mySum = mySum + sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}
__global__ void avg_d(float* ad, float* cd, int size) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < size) {
		const float d = cd[0] / (float)size;
		ad[col] += d;
	}
}
__global__ void transpose(float* input, float* output, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((col < n) && (row < m)) {
		output[col * m + row] = input[row * n + col];
	}
}
__global__ void transpose_d(float* input, float* output, int n, int m) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if ((col < n) && (row < m)) {
		output[col * m + row] += input[row * n + col];
	}
}
__global__ void dropout(float* input, float* output, float drop, ulong seed, int n) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n) {
		ulong x = (ulong)col*seed;
		ulong y = x;
		ulong z = y + seed;
		x = x*x + y;
		x = (x >> 32) | (x << 32);
		x = x*x + z;
		x = (x >> 32) | (x << 32);
		x = x*x + y;
		x = (x >> 32) | (x << 32);
		x = (x*x + z) >> 32;
		ulong rate = (ulong)(drop * (float)0xFFFFFFFF);
		if (x > rate) {
			output[col] = input[col] / (1.0f - drop);
		} else {
			output[col] = 0.0f;
		}
	}
}
__global__ void dropout_d(float* input, float* output, float drop, ulong seed, int n) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < n) {
		ulong x = (ulong)col*seed;
		ulong y = x;
		ulong z = y + seed;
		x = x*x + y;
		x = (x >> 32) | (x << 32);
		x = x*x + z;
		x = (x >> 32) | (x << 32);
		x = x*x + y;
		x = (x >> 32) | (x << 32);
		x = (x*x + z) >> 32;
		ulong rate = (ulong)(drop * (float)0xFFFFFFFF);
		if (x > rate) {
			output[col] += input[col];
		}
	}
}
__global__ void scalar(float* w, const float* d, const float alpha, const int n) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < n) {
		w[x] -= alpha*d[x];
	}
}
const float B1 = 0.8;
const float B2 = 0.89;
__global__ void norm(float* g_idata, float* g_odata, unsigned int n) {
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	float mySum = (i < n) ? g_idata[i]*g_idata[i] : 0;
	if (i + blockDim.x < n) {
		const float input = g_idata[i + blockDim.x];
		mySum += input * input;
	}
	sdata[tid] = mySum;
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] = mySum = mySum + sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}
}
__global__ void adam(float* w, const float* d, float* m, float* v, const float scaling, const float b1, const float b2, const float alpha, const int n, const int set) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if ((x > set) && (x < n)) {
		const float g = d[x] * scaling;
		const float mm = B1*m[x] + (1-B1)*g;
		const float vv = B2*v[x] + (1-B2)*g*g;
		m[x] = mm;
		v[x] = vv;
		const float mhat = mm / (1 - b1);
		float vhat = vv / (1 - b2);
		if (vhat < 0) {
			vhat = 0;
		}
		w[x] -= alpha * mhat / (sqrt(vhat) + 1e-8);
	}
}
`)
	fmt.Fprintf(context.Output, `
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
	float *M;
	float *V;
};
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "struct V %s;\n", value.N)
	}

	fmt.Fprintf(context.Output, `void init(void) {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\t%s.W = %d;\n", value.N, value.S[0])
		fmt.Fprintf(context.Output, "\t%s.H = %d;\n", value.N, value.S[1])
		fmt.Fprintf(context.Output, "\t%s.X = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\t%s.D = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\t%s.M = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\t%s.V = (float*)calloc(%d, sizeof(float));\n", value.N, value.S[0]*value.S[1])
	}
	fmt.Fprintf(context.Output, `}
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "float *device_%s = 0;\n", value.N)
		fmt.Fprintf(context.Output, "float *device_%s_d = 0;\n", value.N)
		fmt.Fprintf(context.Output, "float *device_%s_m = 0;\n", value.N)
		fmt.Fprintf(context.Output, "float *device_%s_v = 0;\n", value.N)
	}
	fmt.Fprintf(context.Output, `
void zero() {`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, `
	CHECK(cudaMemset(device_%s_d, 0, %d * sizeof(float)));
`, value.N, value.S[0]*value.S[1])
	}
	fmt.Fprintf(context.Output, `
}
`)

	fmt.Fprintf(context.Output, `
void scalar(const float alpha) {`)
	for _, value := range set.Weights {
		if value.Skip {
			continue
		}
		fmt.Fprintf(context.Output, `
	dim3 threadsPerBlock_%s(16);
	dim3 blocksPerGrid_%s((%d + threadsPerBlock_%s.x - 1) / threadsPerBlock_%s.x);
	scalar<<<blocksPerGrid_%s, threadsPerBlock_%s>>>((float *)device_%s, (float *)device_%s_d, alpha, %d);
`, value.N, value.N, value.S[0]*value.S[1], value.N, value.N, value.N, value.N, value.N, value.N, value.S[0]*value.S[1])
	}
	fmt.Fprintf(context.Output, `
}
`)
	fmt.Fprintf(context.Output, `
void adam(const int iteration, const float alpha) {
	float sum = 0;
`)
	for _, value := range set.Weights {
		if value.Skip {
			continue
		}
		n := value.S[0] * value.S[1]
		fmt.Fprintf(context.Output, `	int threadsPerBlock_%s = 256;
	int blocksPerGrid_%s = (%d + (threadsPerBlock_%s * 2 - 1)) / (threadsPerBlock_%s * 2);
	size_t sharedMemSize_%s = threadsPerBlock_%s * sizeof(float);
	float *d_odata_%s;
	CHECK(cudaMalloc(&d_odata_%s, blocksPerGrid_%s * sizeof(float)));
	norm<<<blocksPerGrid_%s, threadsPerBlock_%s, sharedMemSize_%s>>>((float *)device_%s, (float *)d_odata_%s, %d);
`, value.N, value.N, n, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N, n)
		fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")
		fmt.Fprintf(context.Output, `	float *h_odata_%s = (float*)malloc(blocksPerGrid_%s * sizeof(float));
	CHECK(cudaMemcpy(h_odata_%s, d_odata_%s, blocksPerGrid_%s * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < blocksPerGrid_%s; i++) {
		sum += h_odata_%s[i];
	}
	free(h_odata_%s);
	CHECK(cudaFree(d_odata_%s));
`, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N)
	}
	fmt.Fprintf(context.Output, `	const float norm = sqrt((float)sum);
	const float b1 = pow(B1, (float)(iteration + 1));
	const float b2 = pow(B2, (float)(iteration + 1));
	float scaling = 1.0;
	if (norm > 1.0) {
		scaling = 1.0 / norm;
	}
`)
	for _, value := range set.Weights {
		if value.Skip {
			continue
		}
		fmt.Fprintf(context.Output, `
	dim3 threadsPerBlock_%s_a(16);
	dim3 blocksPerGrid_%s_a((%d + threadsPerBlock_%s_a.x - 1) / threadsPerBlock_%s_a.x);
	adam<<<blocksPerGrid_%s_a, threadsPerBlock_%s_a>>>((float *)device_%s, (float *)device_%s_d, (float *)device_%s_m, (float *)device_%s_v, scaling, b1, b2, alpha, %d, %d);
`, value.N, value.N, value.S[0]*value.S[1], value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.N, value.S[0]*value.S[1], value.Set)
		fmt.Fprintf(context.Output, "\tCHECK(cudaGetLastError());\n")
	}
	fmt.Fprintf(context.Output, `
}
`)

	fmt.Fprintf(context.Output, `
void uninit(void) {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tfree(%s.X);\n", value.N)
		fmt.Fprintf(context.Output, "\tfree(%s.D);\n", value.N)
		fmt.Fprintf(context.Output, "\tfree(%s.M);\n", value.N)
		fmt.Fprintf(context.Output, "\tfree(%s.V);\n", value.N)
	}
	fmt.Fprintf(context.Output, `
}
void load() {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tCHECK(cudaMalloc((void**)&device_%s, %d * sizeof(float)));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(device_%s, %s.X, %d * sizeof(float), cudaMemcpyHostToDevice));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMalloc((void**)&device_%s_d, %d * sizeof(float)));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(device_%s_d, %s.D, %d * sizeof(float), cudaMemcpyHostToDevice));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMalloc((void**)&device_%s_m, %d * sizeof(float)));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(device_%s_m, %s.M, %d * sizeof(float), cudaMemcpyHostToDevice));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMalloc((void**)&device_%s_v, %d * sizeof(float)));\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(device_%s_v, %s.V, %d * sizeof(float), cudaMemcpyHostToDevice));\n",
			value.N, value.N, value.S[0]*value.S[1])
	}
	fmt.Fprintf(context.Output, `}
int gradient(void) {
`)
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
	fmt.Fprintf(context.Output, `
	return 0;
}
void store() {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(%s.X, device_%s, %d * sizeof(float), cudaMemcpyDeviceToHost));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaFree(device_%s));\n", value.N)
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(%s.D, device_%s_d, %d * sizeof(float), cudaMemcpyDeviceToHost));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaFree(device_%s_d));\n", value.N)
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(%s.M, device_%s_m, %d * sizeof(float), cudaMemcpyDeviceToHost));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaFree(device_%s_m));\n", value.N)
		fmt.Fprintf(context.Output, "\tCHECK(cudaMemcpy(%s.V, device_%s_v, %d * sizeof(float), cudaMemcpyDeviceToHost));\n",
			value.N, value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(cudaFree(device_%s_v));\n", value.N)
	}
	fmt.Fprintf(context.Output, `
}
`)
	return
}
