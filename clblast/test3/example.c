#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // to disable deprecation warnings

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

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
		default: return "Unknown OpenCL Error";
	}
}

void _check(const char* file, int line, const char* func, cl_int err) {
	if (err == CL_SUCCESS) {
		return;
	}
	printf("[ERROR] %s:%d: in function %s - %s\n", file, line, func, getErrorString(err));
	exit(1);
}

#define CHECK(err) _check(__FILE__, __LINE__, __func__, err)


const char* kernel_source = "__kernel void reduce_sum(__global const float* input, __global float* output, __local float* local_cache, unsigned int n) {\n\
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
}\n"; 

int main() {
    int n = 1024; // Array size
    float *h_input = (float*)calloc(n, sizeof(float));
    for(int i = 0; i < n; i++) h_input[i] = 1.0f; // Initialize data

    // 1. Setup Platform and Device
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    // 2. Create Context and Command Queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // 3. Create and Build Program
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "reduce_sum", NULL);

    // 4. Configure Work Sizes
    size_t local_size = 256; 
    size_t num_groups = (n + local_size - 1) / local_size;
    size_t global_size = num_groups * local_size;
    float *h_output = (float*)calloc(num_groups, sizeof(float));

    // 5. Create Buffers
    cl_mem d_input = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                    sizeof(float) * n, h_input, NULL);
    cl_mem d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                     sizeof(float) * num_groups, NULL, NULL);
	clEnqueueWriteBuffer(queue, d_output, CL_TRUE, 0, num_groups * sizeof(float), h_output, 0, NULL, NULL);
	
    // 6. Set Kernel Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, local_size * sizeof(float), NULL); // Local memory allocation
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    // 7. Execute Kernel
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));

    // 8. Read Result
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, num_groups * sizeof(float), h_output, 0, NULL, NULL);

    // Final CPU aggregation of work-group partial sums
    float total_sum = 0;
    for(int i = 0; i < (int)num_groups; i++) {
    	printf("%f\n", h_output[i]);
    	total_sum += h_output[i];
    }
    printf("Total Sum: %f\n", total_sum);

    // Cleanup
    clReleaseMemObject(d_input); clReleaseMemObject(d_output);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    free(h_input); free(h_output);

    return 0;
}
