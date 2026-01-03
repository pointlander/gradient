#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // to disable deprecation warnings

#include <CL/opencl.h> // Include the OpenCL header

const char* kernel_source = "__kernel void everett(__global float* input, __global float* output) {\n\
    const int idx = get_global_id(0);\n\
    const int idx2 = 2*idx;\n\
	const float in = input[idx];\n\
	if (in < 0.0f) {\n\
		output[idx2] = 0.0f;\n\
	} else {\n\
		output[idx2] = in;\n\
	}\n\
	if (in > 0.0f) {\n\
		output[idx2+1] = 0.0f;\n\
	} else {\n\
		output[idx2+1] = in;\n\
	}\n\
}\n\
__kernel void everett_d(__global float* cx, __global float* cd, __global float* ad) {\n\
	const int idx = get_global_id(0);\n\
	const int idxA = idx&~1;\n\
	const int idxB = idx|1;\n\
	if ((cx[idx] != 0) || ((cx[idxA] == 0) && (cx[idxB] == 0))) {\n\
		ad[idx>>1] += cd[idx];\n\
	}\n\
}\n";

int main() {
    // 1. Platform and device discovery
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL); // Use CL_DEVICE_TYPE_GPU or CL_DEVICE_TYPE_DEFAULT

    // 2. Context creation
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // 3. Command queue creation
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL); // For OpenCL 2.0+

    // 4. Create and build the program
    size_t kernel_source_length = strlen(kernel_source);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_length, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL); // Compile the program for the device

    // 5. Create the kernel
    cl_kernel kernel = clCreateKernel(program, "everett", NULL); // Name must match the kernel function in the .cl file

    // (Omitted for brevity: memory buffer creation and data transfer using clCreateBuffer and clEnqueueWriteBuffer)

    // 6. Set kernel arguments
    // Assuming 'memobj_A', 'memobj_B', 'memobj_C' are cl_mem objects created previously
    // clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_A);
    // clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_B);
    // clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_C);

	int n = 1024;
	float* host_input = (float*)malloc(sizeof(float) * n);
	for (int i = 0; i < n; ++i) {
		if (n&1) {
			host_input[i] = 1.0;
		} else {
			host_input[i] = -1.0;
		}
	}
	float* host_input_d = (float*)calloc(n, sizeof(float));
	float* host_output = (float*)calloc(2*n, sizeof(float));
	float* host_output_d = (float*)calloc(2*n, sizeof(float));
	cl_mem device_input = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_input, CL_TRUE, 0, n * sizeof(float), host_input, 0, NULL, NULL);
	cl_mem device_input_d = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_input_d, CL_TRUE, 0, n * sizeof(float), host_input_d, 0, NULL, NULL);
	cl_mem device_output = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * n * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_output, CL_TRUE, 0, 2 * n * sizeof(float), host_output, 0, NULL, NULL);
	cl_mem device_output_d = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * n * sizeof(float), NULL, NULL);
	clEnqueueWriteBuffer(queue, device_output_d, CL_TRUE, 0, 2 * n * sizeof(float), host_output_d, 0, NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_input);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_output);
	
    // 7. Configure NDRange and enqueue the kernel
    size_t global_work_size[] = {n}; // Example for 1D array of 1024 elements
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL); // Execute the kernel

    // 8. Synchronize and read results
    clFinish(queue); // Ensure execution is complete

	cl_kernel kernel_d = clCreateKernel(program, "everett_d", NULL);
	clSetKernelArg(kernel_d, 0, sizeof(cl_mem), (void*)&device_output);
	clSetKernelArg(kernel_d, 1, sizeof(cl_mem), (void*)&device_output_d);
	clSetKernelArg(kernel_d, 1, sizeof(cl_mem), (void*)&device_input_d);
	size_t global_work_size_t[] = {2*n}; // Example for 1D array of 1024 elements
    clEnqueueNDRangeKernel(queue, kernel_d, 1, NULL, global_work_size_t, NULL, 0, NULL, NULL); // Execute the kernel

	clFinish(queue);

    // (Omitted for brevity: read results back to host using clEnqueueReadBuffer)

    // 9. Cleanup
	clReleaseMemObject(device_input);
	clReleaseMemObject(device_input_d);
	clReleaseMemObject(device_output);
	clReleaseMemObject(device_output_d);
    free(host_input);
    free(host_input_d);
    free(host_output);
    free(host_output_d);
    clReleaseKernel(kernel);
    clReleaseKernel(kernel_d);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
