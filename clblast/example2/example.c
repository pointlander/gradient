#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // to disable deprecation warnings

#include <CL/opencl.h> // Include the OpenCL header

// A function to load the kernel source code from a file (e.g., "kernel.cl")
char* load_kernel_source(const char* filename, size_t* length) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Failed to load kernel source file %s\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    *length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* source = (char*)malloc(*length + 1);
    if (!source) {
        printf("Error: Failed to allocate memory for kernel source\n");
        fclose(file);
        return NULL;
    }
    fread(source, 1, *length, file);
    fclose(file);
    source[*length] = 0;
    return source;
}

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
    size_t kernel_source_length;
    char* kernel_source = load_kernel_source("vector_add_kernel.cl", &kernel_source_length); // Load kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_length, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL); // Compile the program for the device
    free(kernel_source);

    // 5. Create the kernel
    cl_kernel kernel = clCreateKernel(program, "vecadd", NULL); // Name must match the kernel function in the .cl file

    // (Omitted for brevity: memory buffer creation and data transfer using clCreateBuffer and clEnqueueWriteBuffer)

    // 6. Set kernel arguments
    // Assuming 'memobj_A', 'memobj_B', 'memobj_C' are cl_mem objects created previously
    // clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_A);
    // clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_B);
    // clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_C);

    // 7. Configure NDRange and enqueue the kernel
    size_t global_work_size[] = {1024}; // Example for 1D array of 1024 elements
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL); // Execute the kernel

    // 8. Synchronize and read results
    clFinish(queue); // Ensure execution is complete

    // (Omitted for brevity: read results back to host using clEnqueueReadBuffer)

    // 9. Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
