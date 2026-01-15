// Copyright 2019 The Gradient Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clblast

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
	fmt.Fprintf(output, "\tcl_int err = 0;\n")
	fmt.Fprintf(output, "\tcl_mem device_%s = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, &err);\n",
		v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tCHECK(err);\n")
	fmt.Fprintf(output, "\terr = 0;\n")
	fmt.Fprintf(output, "\tcl_mem device_%s_d = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, &err);\n",
		v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tCHECK(err);\n")
	fmt.Fprintf(output, "\tcl_event event_a = NULL;\n")
	fmt.Fprintf(output, "\tCHECK(clEnqueueFillBuffer(queue, device_%s, &pattern_value, pattern_size, 0, %d * sizeof(float), 0, NULL, &event_a));\n",
		v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tcl_event event_b = NULL;\n")
	fmt.Fprintf(output, "\tCHECK(clEnqueueFillBuffer(queue, device_%s_d, &pattern_value, pattern_size, 0, %d * sizeof(float), 0, NULL, &event_b));\n",
		v.N, v.S[0]*v.S[1])
	fmt.Fprintf(output, "\tcl_event events[2] = {event_a, event_b};\n")
	fmt.Fprintf(output, `	clWaitForEvents(2, events);
	clReleaseEvent(event_a);
	clReleaseEvent(event_b);
`)
}

// Free generates the code to free the variable
func (v *V) Free(output *os.File) {
	fmt.Fprintf(output, "\tclReleaseMemObject(device_%s_d);\n", v.N)
	fmt.Fprintf(output, "\tclReleaseMemObject(device_%s);\n", v.N)
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

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	cl_kernel kernel = clCreateKernel(program, "mul", NULL);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_%s);
	cl_int width = %d;
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&width);
	size_t global_work_size[] = {%d, %d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.N, b.N, c.N, width, a.S[1], b.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel);\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	cl_kernel kernel_ad = clCreateKernel(program, "mul_ad", NULL);
	clSetKernelArg(kernel_ad, 0, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_ad, 1, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel_ad, 2, sizeof(cl_mem), (void*)&device_%s_d);
	cl_int width_ad = %d;
	cl_int r_ad = %d;
	cl_int c_ad = %d;
	clSetKernelArg(kernel_ad, 3, sizeof(cl_int), (void*)&width_ad);
	clSetKernelArg(kernel_ad, 4, sizeof(cl_int), (void*)&r_ad);
	clSetKernelArg(kernel_ad, 5, sizeof(cl_int), (void*)&c_ad);
	size_t global_work_size_ad[] = {%d, %d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel_ad, 2, NULL, global_work_size_ad, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, c.N, b.N, a.N, width, b.S[1], b.S[0], a.S[1], a.S[0])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_ad);\n")

	fmt.Fprintf(context.Output, `	cl_kernel kernel_bd = clCreateKernel(program, "mul_bd", NULL);
	clSetKernelArg(kernel_bd, 0, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_bd, 1, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel_bd, 2, sizeof(cl_mem), (void*)&device_%s_d);
	cl_int width_bd = %d;
	cl_int r_bd = %d;
	cl_int c_bd = %d;
	clSetKernelArg(kernel_bd, 3, sizeof(cl_int), (void*)&width_bd);
	clSetKernelArg(kernel_bd, 4, sizeof(cl_int), (void*)&r_bd);
	clSetKernelArg(kernel_bd, 5, sizeof(cl_int), (void*)&c_bd);
	size_t global_work_size_bd[] = {%d, %d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel_bd, 2, NULL, global_work_size_bd, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, c.N, a.N, b.N, width, a.S[1], a.S[0], b.S[1], b.S[0])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_bd);\n")

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

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	cl_kernel kernel = clCreateKernel(program, "add", NULL);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_%s);
	size_t global_work_size[] = {%d, %d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.N, b.N, c.N, a.S[0], a.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel);\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	cl_kernel kernel_ad = clCreateKernel(program, "add_ad", NULL);
	clSetKernelArg(kernel_ad, 0, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_ad, 1, sizeof(cl_mem), (void*)&device_%s_d);
	size_t global_work_size_ad[] = {%d, %d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel_ad, 2, NULL, global_work_size_ad, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, c.N, a.N, c.S[0], c.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_ad);\n")

	fmt.Fprintf(context.Output, `	cl_kernel kernel_bd = clCreateKernel(program, "add_bd", NULL);
	clSetKernelArg(kernel_bd, 0, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_bd, 1, sizeof(cl_mem), (void*)&device_%s_d);
	cl_int height = %d;
	clSetKernelArg(kernel_bd, 2, sizeof(cl_int), (void*)&height);
	size_t global_work_size_bd[] = {%d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel_bd, 1, NULL, global_work_size_bd, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, c.N, b.N, a.S[1], a.S[0])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_bd);\n")

	return false
}

// Everett computes the split reality activation function
func (context *Context) Everett(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, 2*a.S[0], a.S[1])

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	cl_kernel kernel = clCreateKernel(program, "everett", NULL);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_%s);
	size_t global_work_size[] = {%d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.N, c.N, a.S[0]*a.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel);\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	cl_kernel kernel_d = clCreateKernel(program, "everett_d", NULL);
	clSetKernelArg(kernel_d, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel_d, 1, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_d, 2, sizeof(cl_mem), (void*)&device_%s_d);
	size_t global_work_size_a[] = {%d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel_d, 1, NULL, global_work_size_a, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, c.N, c.N, a.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_d);\n")

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

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	cl_kernel kernel = clCreateKernel(program, "quadratic", NULL);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_%s);
	cl_int width = %d;
	clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&width);
	size_t global_work_size[] = {%d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.N, b.N, c.N, a.S[0], a.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel);\n")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	cl_kernel kernel_d = clCreateKernel(program, "quadratic_d", NULL);
	clSetKernelArg(kernel_d, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel_d, 1, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel_d, 2, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_d, 3, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_d, 4, sizeof(cl_mem), (void*)&device_%s_d);
	size_t global_work_size_d[] = {%d, %d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel_d, 2, NULL, global_work_size_d, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.N, b.N, c.N, a.N, b.N, a.S[0], a.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_d);\n")

	return false
}

// Avg computes the average of the tensor
func (context *Context) Avg(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := NewV(node, 1)

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	c.Allocate(context.Output)
	defer c.Free(context.Output)
	fmt.Fprintf(context.Output, `	int n = %d;
	size_t local_size = 256;
	size_t num_groups = (n + local_size - 1) / local_size;
	size_t global_size = num_groups * local_size;
	float *h_output = (float*)calloc(num_groups, sizeof(float));
	cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_groups, NULL, &err);
	CHECK(err);
	err = 0;
	CHECK(clEnqueueWriteBuffer(queue, d_output, CL_TRUE, 0, num_groups * sizeof(float), h_output, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.S[0]*a.S[1])

	fmt.Fprintf(context.Output, `	cl_kernel kernel = clCreateKernel(program, "reduce_sum", NULL);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_%s);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_output);
	clSetKernelArg(kernel, 2, local_size * sizeof(float), NULL);
	clSetKernelArg(kernel, 3, sizeof(unsigned int), (void*)&n);
	size_t global_work_size[] = {global_size};
	size_t local_work_size[] = {local_size};
	CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
	CHECK(clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, num_groups * sizeof(float), h_output, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
	float sum = 0;
	for(int i = 0; i < (int)num_groups; i++) {
		sum += h_output[i];
	}
	sum = sum / ((float)n);
	CHECK(clEnqueueWriteBuffer(queue, device_%s, CL_TRUE, 0, 1 * sizeof(float), &sum, 0, NULL, &event));
	
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.N, c.N)
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel);\n")
	fmt.Fprintf(context.Output, "\tclReleaseMemObject(d_output);\n")
	fmt.Fprintf(context.Output, "\tfree(h_output);")

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, `	cl_kernel kernel_d = clCreateKernel(program, "avg_d", NULL);
	clSetKernelArg(kernel_d, 0, sizeof(cl_mem), (void*)&device_%s_d);
	clSetKernelArg(kernel_d, 1, sizeof(cl_mem), (void*)&device_%s_d);
	size_t global_work_size_d[] = {%d};
	CHECK(clEnqueueNDRangeKernel(queue, kernel_d, 1, NULL, global_work_size_d, NULL, 0, NULL, &event));
	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`, a.N, c.N, a.S[0]*a.S[1])
	fmt.Fprintf(context.Output, "\tclReleaseKernel(kernel_d);\n")

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
	fmt.Fprintf(context.Output, `//go:build ignore
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
	
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <clblast_c.h>

const char* kernel_source = "__kernel void everett(__global float* input, __global float* output) {\n\
	const int idx = get_global_id(0);\n\
	const int idx2 = 2*idx;\n\
	const float in = input[idx];\n\
	if (in > 0.0f) {\n\
		output[idx2] = 0.0f;\n\
	} else {\n\
		output[idx2] = in;\n\
	}\n\
	if (in < 0.0f) {\n\
		output[idx2+1] = 0.0f;\n\
	} else {\n\
		output[idx2+1] = in;\n\
	}\n\
}\n\
__kernel void everett_d(__global float* cx, __global float* cd, __global float* ad) {\n\
	const int idx = get_global_id(0);\n\
	const int idxA = idx&~1U;\n\
	const int idxB = idx|1U;\n\
	if ((cx[idx] != 0) || ((cx[idxA] == 0) && (cx[idxB] == 0))) {\n\
		ad[idx>>1] += cd[idx];\n\
	}\n\
}\n\
__kernel void mul(__global float* a, __global float* b, __global float* c, const int width) {\n\
	int aw = get_global_size(0);\n\
	int ai = get_global_id(0);\n\
	int bi = get_global_id(1);\n\
	float sum = 0;\n\
	for (int i = 0; i < width; i++) {\n\
		sum += a[ai*width+i] * b[bi*width+i];\n\
	}\n\
	c[bi*aw + ai] = sum;\n\
}\n\
__kernel void mul_ad(__global float* cd, __global float* b, __global float* ad, const int width, const int r, const int c) {\n\
	int cols = get_global_size(0);\n\
	int row = get_global_id(0);\n\
	int col = get_global_id(1);\n\
	float sum = 0;\n\
	for (int i = 0; i < r; i++) {\n\
		sum += cd[row+i*cols]*b[i*width+col];\n\
	}\n\
	ad[row*width+col] += sum;\n\
}\n\
__kernel void mul_bd(__global float* cd, __global float* a, __global float* bd, const int width, const int r, const int c) {\n\
	int cols = get_global_size(1);\n\
	int row = get_global_id(0);\n\
	int col = get_global_id(1);\n\
	float sum = 0;\n\
	for (int i = 0; i < r; i++) {\n\
		sum += cd[i+row*cols]*a[i*width+col];\n\
	}\n\
	bd[row*width+col] += sum;\n\
}\n\
__kernel void add(__global float* a, __global float* b, __global float* c) {\n\
	int width = get_global_size(0);\n\
	int ai = get_global_id(0);\n\
	int bi = get_global_id(1);\n\
	c[bi*width + ai] = a[bi*width+ai] + b[ai];\n\
}\n\
__kernel void add_ad(__global float* cd, __global float* ad) {\n\
	int width = get_global_size(0);\n\
	int ai = get_global_id(0);\n\
	int bi = get_global_id(1);\n\
	ad[bi*width+ai] += cd[bi*width + ai];\n\
}\n\
__kernel void add_bd(__global float* cd, __global float* bd, const int height) {\n\
	int width = get_global_size(0);\n\
	int ai = get_global_id(0);\n\
	float sum = 0;\n\
	for (int i = 0; i < height; i++) {\n\
		sum += cd[i*width + ai];\n\
	}\n\
	bd[ai] += sum;\n\
}\n\
__kernel void quadratic(__global float* a, __global float* b, __global float* c, const int width) {\n\
	int ai = get_global_id(0);\n\
	float sum = 0;\n\
	for (int i = 0; i < width; i++) {\n\
		float diff = a[ai*width + i] - b[ai*width + i];\n\
		sum += diff*diff;\n\
	}\n\
	c[ai] = sum;\n\
}\n\
__kernel void quadratic_d(__global float* a, __global float* b, __global float* cd, __global float* ad, __global float* bd) {\n\
	int width = get_global_size(0);\n\
	int ai = get_global_id(0);\n\
	int bi = get_global_id(1);\n\
	float d = cd[bi];\n\
	ad[bi*width+ai] += (a[bi*width+ai] - b[bi*width+ai]) * d;\n\
	bd[bi*width+ai] += (b[bi*width+ai] - a[bi*width+ai]) * d;\n\
}\n\
__kernel void reduce_sum(__global const float* input, __global float* output, __local float* local_cache, unsigned int n) {\n\
	uint global_id = get_global_id(0);\n\
	uint local_id = get_local_id(0);\n\
	uint group_size = get_local_size(0);\n\
	local_cache[local_id] = (global_id < n) ? input[global_id] : 0.0f;\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
	for (uint stride = group_size / 2; stride > 0; stride /= 2) {\n\
		if (local_id < stride) {\n\
			local_cache[local_id] += local_cache[local_id + stride];\n\
		}\n\
		barrier(CLK_LOCAL_MEM_FENCE);\n\
	}\n\
	if (local_id == 0) {\n\
		output[get_group_id(0)] = local_cache[0];\n\
	}\n\
}\n\
__kernel void avg_d(__global float* ad, __global float* cd) {\n\
	int size = get_global_size(0);\n\
	int ai = get_global_id(0);\n\
	const float d = cd[0] / (float)size;\n\
	ad[ai] += d;\n\
}\n";

const char* getErrorString(cl_int error) {
    switch (error) {
		case CL_SUCCESS: return "CL_SUCCESS";
		case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
		case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
		case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
		case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
		case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT: return "CL_INVALID_EVENT"; 
		case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
		case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
		case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
		case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_LINKER_OPTIONS";

		case CLBlastNotImplemented: return "CLBlastNotImplemented";
		case CLBlastInvalidMatrixA: return "CLBlastInvalidMatrixA";
		case CLBlastInvalidMatrixB: return "CLBlastInvalidMatrixB";
		case CLBlastInvalidMatrixC: return "CLBlastInvalidMatrixC";
		case CLBlastInvalidVectorX: return "CLBlastInvalidVectorX";
		case CLBlastInvalidVectorY: return "CLBlastInvalidVectorY";
		case CLBlastInvalidDimension: return "CLBlastInvalidDimension";
		case CLBlastInvalidLeadDimA: return "CLBlastInvalidLeadDimA";
		case CLBlastInvalidLeadDimB: return "CLBlastInvalidLeadDimB";
		case CLBlastInvalidLeadDimC: return "CLBlastInvalidLeadDimC";
		case CLBlastInvalidIncrementX: return "CLBlastInvalidIncrementX";
		case CLBlastInvalidIncrementY: return "CLBlastInvalidIncrementY";
		case CLBlastInsufficientMemoryA: return "CLBlastInsufficientMemoryA";
		case CLBlastInsufficientMemoryB: return "CLBlastInsufficientMemoryB";
		case CLBlastInsufficientMemoryC: return "CLBlastInsufficientMemoryC";
		case CLBlastInsufficientMemoryX: return "CLBlastInsufficientMemoryX";
		case CLBlastInsufficientMemoryY: return "CLBlastInsufficientMemoryY";

		case CLBlastInsufficientMemoryTemp: return "CLBlastInsufficientMemoryTemp";
		case CLBlastInvalidBatchCount: return "CLBlastInvalidBatchCount";
		case CLBlastInvalidOverrideKernel: return "CLBlastInvalidOverrideKernel";
		case CLBlastMissingOverrideParameter: return "CLBlastMissingOverrideParameter";
		case CLBlastInvalidLocalMemUsage: return "CLBlastInvalidLocalMemUsage";
		case CLBlastNoHalfPrecision: return "CLBlastNoHalfPrecision";
		case CLBlastNoDoublePrecision: return "CLBlastNoDoublePrecision";
		case CLBlastInvalidVectorScalar: return "CLBlastInvalidVectorScalar";
		case CLBlastInsufficientMemoryScalar: return "CLBlastInsufficientMemoryScalar";
		case CLBlastDatabaseError: return "CLBlastDatabaseError";
		case CLBlastUnknownError: return "CLBlastUnknownError";
		case CLBlastUnexpectedError: return "CLBlastUnexpectedError";
		default: return "Unknown OpenCL Error";
	}
}

void _check(const char* file, int line, const char* func, cl_int err) {
	if (err == CL_SUCCESS) {
		return;
	}
	printf("[ERROR] %%s:%%d: in function %%s - %%s\n", file, line, func, getErrorString(err));
	exit(1);
}

#define CHECK(err) _check(__FILE__, __LINE__, __func__, err)

void callback(float* output, int w, int h);

float pattern_value = 0.0f;
size_t pattern_size = sizeof(float);

cl_context context;
cl_command_queue queue;
cl_program program;

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
	fmt.Fprintf(context.Output, `	const size_t platform_id = 0;
	const size_t device_id = 0;
	cl_int err = 0;
	
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	printf("%%d\n", num_platforms);
	cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
	CHECK(clGetPlatformIDs(num_platforms, platforms, NULL));
	cl_platform_id platform = platforms[platform_id];

	cl_uint num_devices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	printf("%%d\n", num_devices);
	cl_device_id* devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
	CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));
	cl_device_id device = devices[device_id];

	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK(err);
	err = 0;
	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK(err);
	err = 0;
	size_t kernel_source_length = strlen(kernel_source);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_length, &err);
	CHECK(err);
	err = 0;
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (err != 0) {
		size_t log_size = 0;
		cl_int err = 0;
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK(err);
		char* log = (char*)calloc(log_size, 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK(err);
		printf("build log: %%s\n", log);
		free(log);
	}
	CHECK(err);

	free(platforms);
	free(devices);	
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
	fmt.Fprintf(context.Output, `	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}
int gradient(void) {
`)
	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	fmt.Fprintf(context.Output, "\tcl_int err = 0;\n")
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tcl_mem device_%s = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, &err);\n",
			value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(err);\n")
		fmt.Fprintf(context.Output, "\terr = 0;\n")
		fmt.Fprintf(context.Output, "\tCHECK(clEnqueueWriteBuffer(queue, device_%s, CL_TRUE, 0, %d * sizeof(float), %s.X, 0, NULL, &event));\n",
			value.N, value.S[0]*value.S[1], value.N)
		fmt.Fprintf(context.Output, `	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`)
		fmt.Fprintf(context.Output, "\tcl_mem device_%s_d = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, &err);\n",
			value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(err);\n")
		fmt.Fprintf(context.Output, "\terr = 0;\n")
		fmt.Fprintf(context.Output, "\tCHECK(clEnqueueWriteBuffer(queue, device_%s_d, CL_TRUE, 0, %d * sizeof(float), %s.D, 0, NULL, &event));\n",
			value.N, value.S[0]*value.S[1], value.N)
		fmt.Fprintf(context.Output, `	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`)
	}
	a(func(a *V) bool {
		fmt.Fprintf(context.Output, "\t{\n")
		fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
		fmt.Fprintf(context.Output, "\tcl_int status = 0;\n")
		fmt.Fprintf(context.Output, "\tfloat* host_%s = (float*)calloc(%d, sizeof(float));\n", a.N, a.S[0]*a.S[1])
		fmt.Fprintf(context.Output, "\tCHECK(clEnqueueReadBuffer(queue, device_%s, CL_TRUE, 0, %d * sizeof(float), host_%s, 0, NULL, &event));\n",
			a.N, a.S[0]*a.S[1], a.N)
		fmt.Fprintf(context.Output, `	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`)
		fmt.Fprintf(context.Output, "\tcallback(host_%s, %d, %d);\n", a.N, a.S[0], a.S[1])
		fmt.Fprintf(context.Output, `	for (int i = 0; i < %d; i++) {
			host_%s[i] = 1.0;
	}
`, a.S[0]*a.S[1], a.N)
		fmt.Fprintf(context.Output, "\tCHECK(clEnqueueWriteBuffer(queue, device_%s_d, CL_TRUE, 0, %d * sizeof(float), host_%s, 0, NULL, &event));\n",
			a.N, a.S[0]*a.S[1], a.N)
		fmt.Fprintf(context.Output, `	clWaitForEvents(1, &event);
	clReleaseEvent(event);
	event = NULL;
`)
		fmt.Fprintf(context.Output, "\tfree(host_%s);\n", a.N)
		fmt.Fprintf(context.Output, "\t}\n")
		return false
	})
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s);\n", value.N)
		fmt.Fprintf(context.Output, "\tCHECK(clEnqueueReadBuffer(queue, device_%s_d, CL_TRUE, 0, %d * sizeof(float), %s.D, 0, NULL, NULL));\n",
			value.N, value.S[0]*value.S[1], value.N)
		fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s_d);\n", value.N)
	}
	fmt.Fprintf(context.Output, `
}
`)
	return
}
