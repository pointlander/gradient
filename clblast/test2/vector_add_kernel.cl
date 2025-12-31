// OpenCL C Kernel function (must return void)
__kernel void vecadd(__global int* A, __global int* B, __global int* C) {
    // Get the global unique ID of the current work item
    const int idx = get_global_id(0);

    // Perform the parallel operation
    C[idx] = A[idx] + B[idx];
}
