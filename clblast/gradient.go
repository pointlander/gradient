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
	v := context.NewV(d...)
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
func (context *Context) NewV(s ...int) V {
	if len(s) == 1 {
		s = []int{s[0], 1}
	}
	return V{
		N: fmt.Sprintf("node%d", context.Node),
		S: s,
	}
}

// Avg computes the average of the tensor
func (context *Context) Avg(k Continuation, node int, a *V, options ...map[string]interface{}) bool {
	c := context.NewV(1)

	fmt.Fprintf(context.Output, "\tcl_event event = NULL;\n")
	fmt.Fprintf(context.Output, "\tcl_mem device_%s = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
		c.N, c.S[0]*c.S[1])
	fmt.Fprintf(context.Output, "\tCLBlastStatusCode status = CLBlastSsum(%d, device_%s, 0, device_%s, 0, 1, &queue, &event);\n",
		c.S[0]*c.S[1], c.N, a.N)
	fmt.Fprintf(context.Output, `	if (status == CLBlastSuccess) {
		clWaitForEvents(1, &event);
		clReleaseEvent(event);
	}
`)

	if k(&c) {
		return true
	}

	fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s);\n", c.N)

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
					return op(k, node, b...), nil
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
					derivatives = op(k, node, a, b, options...)
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
				return op(k, node, b, options...)
			})
		}
	}
}

// Gradient computes the gradient
func (context *Context) Gradient(set Set, a Meta) (cost V) {
	mk, err := os.Create("Makefile")
	if err != nil {
		panic(err)
	}
	defer mk.Close()
	fmt.Fprintf(mk, `all: *.c
	gcc -o model *.c -lclblast -lOpenCL
`)

	fmt.Fprintf(context.Output, `#include <stdio.h>
#include <stdlib.h>
#include <string.h>
	
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <clblast_c.h>

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
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\t%s.W = %d;\n", value.N, value.S[0])
		fmt.Fprintf(context.Output, "\t%s.H = %d;\n", value.N, value.S[1])
		fmt.Fprintf(context.Output, "\t%s.X = (float*)malloc(sizeof(float) * %d);\n", value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\t%s.D = (float*)malloc(sizeof(float) * %d);\n", value.N, value.S[0]*value.S[1])
	}
	fmt.Fprintf(context.Output, `}
void uninit(void) {
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tfree(%s.X);\n", value.N)
		fmt.Fprintf(context.Output, "\tfree(%s.D);\n", value.N)
	}
	fmt.Fprintf(context.Output, `}
int main(void) {
	init();
	const size_t platform_id = 0;
	const size_t device_id = 0;

	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	printf("%%d\n", num_platforms);
	cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	clGetPlatformIDs(num_platforms, platforms, NULL);
	cl_platform_id platform = platforms[platform_id];

	cl_uint num_devices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	printf("%%d\n", num_devices);
	cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	cl_device_id device = devices[device_id];

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
`)
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tcl_mem device_%s = clCreateBuffer(context, CL_MEM_READ_WRITE, %d * sizeof(float), NULL, NULL);\n",
			value.N, value.S[0]*value.S[1])
		fmt.Fprintf(context.Output, "\tclEnqueueWriteBuffer(queue, device_%s, CL_TRUE, 0, %d * sizeof(float), %s.X, 0, NULL, NULL);\n",
			value.N, value.S[0]*value.S[1], value.N)
	}
	a(func(a *V) bool {
		fmt.Fprintf(context.Output, "\tfloat* host_%s = (float*)malloc(sizeof(float) * %d);\n", a.N, a.S[0]*a.S[1])
		fmt.Fprintf(context.Output, "\tclEnqueueReadBuffer(queue, device_%s, CL_TRUE, 0, %d * sizeof(float), host_%s, 0, NULL, NULL);\n",
			a.N, a.S[0]*a.S[1], a.N)
		fmt.Fprintf(context.Output, "\tfree(host_%s);\n", a.N)
		return false
	})
	for _, value := range set.Weights {
		fmt.Fprintf(context.Output, "\tclReleaseMemObject(device_%s);\n", value.N)
	}
	fmt.Fprintf(context.Output, `
	uninit();
	free(platforms);
	free(devices);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);	
}
`)
	return
}
