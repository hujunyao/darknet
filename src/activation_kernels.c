#ifdef GPU

#include "darknet.h"

#include <string.h>
#include "activation_layer.h"
#include "opencl.h"

#include "activation_kernels.cl"

cl_program opencl_activation_kernel_program = 0;
cl_kernel opencl_activate_array_kernel = 0;
cl_kernel opencl_gradient_array_kernel = 0;

void activation_kernel_init(void)
{
	opencl_load_buffer(activation_kernels_source, strlen(activation_kernels_source), &opencl_activation_kernel_program);
	opencl_create_kernel(&opencl_activation_kernel_program,
		"activate_array_kernel", &opencl_activate_array_kernel);
	opencl_create_kernel(&opencl_activation_kernel_program,
		"gradient_array_kernel", &opencl_gradient_array_kernel);
}

void activation_kernel_release(void)
{
	clReleaseKernel(opencl_activate_array_kernel);
	clReleaseKernel(opencl_gradient_array_kernel);
	clReleaseProgram(opencl_activation_kernel_program);

	opencl_activate_array_kernel = 0;
	opencl_gradient_array_kernel = 0;
	opencl_activation_kernel_program = 0;
}

void activate_array_offset_gpu(cl_mem_ext x, int offset, int n, ACTIVATION a)
{
	dim2 dimN;
	dimN = opencl_gridsize(n);
	opencl_kernel(opencl_activate_array_kernel, dimN, 8, &x, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &a, sizeof(cl_int));
}

void activate_array_gpu(cl_mem_ext x, int n, ACTIVATION a)
{
    activate_array_offset_gpu(x, 0, n, a);
}

void gradient_array_offset_gpu(cl_mem_ext x, int offset, int n, ACTIVATION a, cl_mem_ext delta)
{
	dim2 dimN;
	dimN = opencl_gridsize(n);
	opencl_kernel(opencl_gradient_array_kernel, dimN, 10, &x, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &a, sizeof(cl_int), &delta, sizeof(cl_mem));
}

void gradient_array_gpu(cl_mem_ext x, int n, ACTIVATION a, cl_mem_ext delta)
{
    gradient_array_offset_gpu(x, 0, n, a, delta);
}

#endif // GPU