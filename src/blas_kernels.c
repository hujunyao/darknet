#include <assert.h>
#include <string.h>

#include "darknet.h"

#include "blas.h"
#include "opencl.h"
#include "utils.h"
#include "blas_kernels.cl"

#ifdef GPU

#include "opencl.h"
#include "layer.h"

cl_program opencl_blas_kernel_program = 0;

cl_kernel test_kernel = 0;
cl_kernel softmax_device_kernel = 0;
cl_kernel opencl_scale_bias_kernel = 0;
cl_kernel opencl_backward_scale_kernel = 0;
cl_kernel opencl_add_bias_kernel = 0;
cl_kernel opencl_backward_bias_kernel = 0;
cl_kernel opencl_adam_kernel = 0;
cl_kernel opencl_normalize_kernel = 0;
cl_kernel opencl_normalize_delta_kernel = 0;
cl_kernel opencl_l2norm_kernel = 0;
cl_kernel opencl_variance_delta_kernel = 0;
cl_kernel opencl_accumulate_kernel = 0;
cl_kernel opencl_mean_delta_kernel = 0;
cl_kernel opencl_mean_kernel = 0;
cl_kernel opencl_variance_kernel = 0;
cl_kernel opencl_reorg_kernel = 0;
cl_kernel opencl_axpy_kernel = 0;
cl_kernel opencl_pow_kernel = 0;
cl_kernel opencl_const_kernel = 0;
cl_kernel opencl_constrain_kernel = 0;
cl_kernel opencl_supp_kernel = 0;
cl_kernel opencl_add_kernel = 0;
cl_kernel opencl_scal_kernel = 0;
cl_kernel opencl_fill_kernel = 0;
cl_kernel opencl_mask_kernel = 0;
cl_kernel opencl_copy_kernel = 0;
cl_kernel opencl_mul_kernel = 0;
//cl_kernel opencl_fast_mean_kernel = 0;
//cl_kernel opencl_fast_variance_kernel = 0;
//cl_kernel opencl_fast_mean_delta_kernel = 0;
//cl_kernel opencl_fast_variance_delta_kernel = 0;
cl_kernel opencl_flatten_kernel = 0;
cl_kernel opencl_shortcut_kernel = 0;
cl_kernel opencl_smooth_l1_kernel = 0;
cl_kernel opencl_softmax_x_ent_kernel = 0;
cl_kernel opencl_logistic_x_ent_kernel = 0;
cl_kernel opencl_l2_kernel = 0;
cl_kernel opencl_l1_kernel = 0;
cl_kernel opencl_wgan_kernel = 0;
cl_kernel opencl_inter_kernel = 0;
cl_kernel opencl_deinter_kernel = 0;
cl_kernel opencl_weighted_sum_kernel = 0;
cl_kernel opencl_weighted_delta_kernel = 0;
cl_kernel opencl_mult_add_into_kernel = 0;
cl_kernel opencl_softmax_tree_kernel = 0;
cl_kernel opencl_softmax_kernel = 0;
cl_kernel opencl_scale_mask_kernel = 0;
cl_kernel opencl_dot_kernel = 0;
cl_kernel opencl_upsample_kernel = 0;

void blas_kernel_init(void)
{
    opencl_load_buffer(blas_kernel_source, strlen(blas_kernel_source), &opencl_blas_kernel_program);

    opencl_create_kernel(&opencl_blas_kernel_program, "test_kernel", &test_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "softmax_device", &softmax_device_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "scale_bias_kernel", &opencl_scale_bias_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "backward_scale_kernel", &opencl_backward_scale_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "add_bias_kernel", &opencl_add_bias_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "backward_bias_kernel", &opencl_backward_bias_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "adam_kernel", &opencl_adam_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "normalize_kernel", &opencl_normalize_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "normalize_delta_kernel", &opencl_normalize_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "l2norm_kernel", &opencl_l2norm_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "variance_delta_kernel", &opencl_variance_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "accumulate_kernel", &opencl_accumulate_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mean_delta_kernel", &opencl_mean_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mean_kernel", &opencl_mean_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "variance_kernel", &opencl_variance_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "reorg_kernel", &opencl_reorg_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "axpy_kernel", &opencl_axpy_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "pow_kernel", &opencl_pow_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "const_kernel", &opencl_const_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "constrain_kernel", &opencl_constrain_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "supp_kernel", &opencl_supp_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "add_kernel", &opencl_add_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "scal_kernel", &opencl_scal_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "fill_kernel", &opencl_fill_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mask_kernel", &opencl_mask_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "copy_kernel", &opencl_copy_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mul_kernel", &opencl_mul_kernel);
    //opencl_create_kernel(&opencl_blas_kernel_program, "fast_mean_kernel", &opencl_fast_mean_kernel);
    //opencl_create_kernel(&opencl_blas_kernel_program, "fast_variance_kernel", &opencl_fast_variance_kernel);
    //opencl_create_kernel(&opencl_blas_kernel_program, "fast_mean_delta_kernel", &opencl_fast_mean_delta_kernel);
    //opencl_create_kernel(&opencl_blas_kernel_program, "fast_variance_delta_kernel", &opencl_fast_variance_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "flatten_kernel", &opencl_flatten_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "shortcut_kernel", &opencl_shortcut_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "smooth_l1_kernel", &opencl_smooth_l1_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "softmax_x_ent_kernel", &opencl_softmax_x_ent_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "logistic_x_ent_kernel", &opencl_logistic_x_ent_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "l2_kernel", &opencl_l2_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "l1_kernel", &opencl_l1_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "wgan_kernel", &opencl_wgan_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "deinter_kernel", &opencl_deinter_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "inter_kernel", &opencl_inter_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "weighted_sum_kernel", &opencl_weighted_sum_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "weighted_delta_kernel", &opencl_weighted_delta_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "mult_add_into_kernel", &opencl_mult_add_into_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "softmax_tree_kernel", &opencl_softmax_tree_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "softmax_kernel", &opencl_softmax_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "scale_mask_kernel", &opencl_scale_mask_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "dot_kernel", &opencl_dot_kernel);
    opencl_create_kernel(&opencl_blas_kernel_program, "upsample_kernel", &opencl_upsample_kernel);
}

void blas_kernel_release(void)
{
    clReleaseKernel(test_kernel); test_kernel = 0;
    clReleaseKernel(softmax_device_kernel); softmax_device_kernel = 0;
    clReleaseKernel(opencl_scale_bias_kernel); opencl_scale_bias_kernel = 0;
    clReleaseKernel(opencl_backward_scale_kernel); opencl_backward_scale_kernel = 0;
    clReleaseKernel(opencl_add_bias_kernel); opencl_add_bias_kernel = 0;
    clReleaseKernel(opencl_backward_bias_kernel); opencl_backward_bias_kernel = 0;
    clReleaseKernel(opencl_adam_kernel); opencl_adam_kernel = 0;
    clReleaseKernel(opencl_normalize_kernel); opencl_normalize_kernel = 0;
    clReleaseKernel(opencl_normalize_delta_kernel); opencl_normalize_delta_kernel = 0;
    clReleaseKernel(opencl_l2norm_kernel); opencl_l2norm_kernel = 0;
    clReleaseKernel(opencl_variance_delta_kernel); opencl_variance_delta_kernel = 0;
    clReleaseKernel(opencl_accumulate_kernel); opencl_accumulate_kernel = 0;
    clReleaseKernel(opencl_mean_delta_kernel); opencl_mean_delta_kernel = 0;
    clReleaseKernel(opencl_mean_kernel); opencl_mean_kernel = 0;
    clReleaseKernel(opencl_variance_kernel); opencl_variance_kernel = 0;
    clReleaseKernel(opencl_reorg_kernel); opencl_reorg_kernel = 0;
    clReleaseKernel(opencl_axpy_kernel); opencl_axpy_kernel = 0;
    clReleaseKernel(opencl_pow_kernel); opencl_pow_kernel = 0;
    clReleaseKernel(opencl_const_kernel); opencl_const_kernel = 0;
    clReleaseKernel(opencl_constrain_kernel); opencl_constrain_kernel = 0;
    clReleaseKernel(opencl_supp_kernel); opencl_supp_kernel = 0;
    clReleaseKernel(opencl_add_kernel); opencl_add_kernel = 0;
    clReleaseKernel(opencl_scal_kernel); opencl_scal_kernel = 0;
    clReleaseKernel(opencl_fill_kernel); opencl_fill_kernel = 0;
    clReleaseKernel(opencl_mask_kernel); opencl_mask_kernel = 0;
    clReleaseKernel(opencl_copy_kernel); opencl_copy_kernel = 0;
    clReleaseKernel(opencl_mul_kernel); opencl_mul_kernel = 0;
    //clReleaseKernel(opencl_fast_mean_kernel); opencl_fast_mean_kernel = 0;
    //clReleaseKernel(opencl_fast_variance_kernel); opencl_fast_variance_kernel = 0;
    //clReleaseKernel(opencl_fast_mean_delta_kernel); opencl_fast_mean_delta_kernel = 0;
    //clReleaseKernel(opencl_fast_variance_delta_kernel); opencl_fast_variance_delta_kernel = 0;
    clReleaseKernel(opencl_flatten_kernel); opencl_flatten_kernel = 0;
    clReleaseKernel(opencl_shortcut_kernel); opencl_shortcut_kernel = 0;
    clReleaseKernel(opencl_smooth_l1_kernel); opencl_smooth_l1_kernel = 0;
    clReleaseKernel(opencl_softmax_x_ent_kernel); opencl_softmax_x_ent_kernel = 0;
    clReleaseKernel(opencl_logistic_x_ent_kernel); opencl_logistic_x_ent_kernel = 0;
    clReleaseKernel(opencl_l2_kernel); opencl_l2_kernel = 0;
    clReleaseKernel(opencl_l1_kernel); opencl_l1_kernel = 0;
    clReleaseKernel(opencl_wgan_kernel); opencl_wgan_kernel = 0;
    clReleaseKernel(opencl_deinter_kernel); opencl_deinter_kernel = 0;
    clReleaseKernel(opencl_inter_kernel); opencl_inter_kernel = 0;
    clReleaseKernel(opencl_weighted_sum_kernel); opencl_weighted_sum_kernel = 0;
    clReleaseKernel(opencl_weighted_delta_kernel); opencl_weighted_delta_kernel = 0;
    clReleaseKernel(opencl_mult_add_into_kernel); opencl_mult_add_into_kernel = 0;
    clReleaseKernel(opencl_softmax_tree_kernel); opencl_softmax_tree_kernel = 0;
    clReleaseKernel(opencl_softmax_kernel); opencl_softmax_kernel = 0;
    clReleaseKernel(opencl_scale_mask_kernel); opencl_scale_mask_kernel = 0;
    clReleaseKernel(opencl_dot_kernel); opencl_dot_kernel = 0;
    clReleaseKernel(opencl_upsample_kernel); opencl_upsample_kernel = 0;

    clReleaseProgram(opencl_blas_kernel_program); opencl_blas_kernel_program = 0;
}

void test_kernel_gpu(int N, cl_mem_ext input, cl_mem_ext output, cl_mem_ext expected)
{
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(test_kernel, dimGrid, 8,
        &N, sizeof(cl_int),
        &input.mem, sizeof(cl_mem),
        &output.mem, sizeof(cl_mem),
        &expected.mem, sizeof(cl_mem)
    );
}

void scale_bias_gpu(cl_mem_ext output, cl_mem_ext biases, int batch, int n, int size)
{
    int N = batch * n * size;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_scale_bias_kernel, dimGrid, 12, &N, sizeof(cl_int), &output.mem, sizeof(cl_mem), &biases.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}


void backward_scale_gpu(cl_mem_ext x_norm, cl_mem_ext delta, int batch, int n, int size, cl_mem_ext scale_updates)
{
    int N = n * batch * size;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_backward_scale_kernel, dimGrid, 14, &N, sizeof(cl_int), &x_norm.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int), &scale_updates.mem, sizeof(cl_mem));
}


void add_bias_gpu(cl_mem_ext output, cl_mem_ext biases, int batch, int n, int size)
{
    int N = batch * n * size;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_add_bias_kernel, dimGrid, 12, &N, sizeof(cl_int), &output.mem, sizeof(cl_mem), &biases.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}


void backward_bias_gpu(cl_mem_ext bias_updates, cl_mem_ext delta, int batch, int n, int size)
{
    int N = batch * n * size;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_backward_bias_kernel, dimGrid, 12, &N, sizeof(cl_int), &bias_updates.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &n, sizeof(cl_int), &size, sizeof(cl_int));
}


void adam_gpu(int n, cl_mem_ext x, cl_mem_ext m, cl_mem_ext v, float B1, float B2, float rate, float eps, int t)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(n);

    opencl_kernel(opencl_adam_kernel, dimGrid, 18, &n, sizeof(cl_int), &x.mem, sizeof(cl_mem), &m.mem, sizeof(cl_mem), &v.mem, sizeof(cl_mem), &B1, sizeof(cl_float), &B2, sizeof(cl_float), &rate, sizeof(cl_float), &eps, sizeof(cl_float), &t, sizeof(cl_int));
}


void normalize_gpu(cl_mem_ext x, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    dim2 dimN;
    dimN = opencl_gridsize(N);

    opencl_kernel(opencl_normalize_kernel, dimN, 14, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int));
}


void normalize_delta_gpu(cl_mem_ext x, cl_mem_ext mean, cl_mem_ext variance, cl_mem_ext mean_delta, cl_mem_ext variance_delta, int batch, int filters, int spatial, cl_mem_ext delta)
{
    size_t N = batch*filters*spatial;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_normalize_delta_kernel, dimGrid, 20, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &mean_delta.mem, sizeof(cl_mem), &variance_delta.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &delta.mem, sizeof(cl_mem));
}


void mean_gpu(cl_mem_ext x, int batch, int filters, int spatial, cl_mem_ext mean)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_mean_kernel, dimGrid, 12, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean.mem, sizeof(cl_mem));
}

void variance_gpu(cl_mem_ext x, cl_mem_ext mean, int batch, int filters, int spatial, cl_mem_ext variance)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_variance_kernel, dimGrid, 14, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance.mem, sizeof(cl_mem));
}

void mean_delta_gpu(cl_mem_ext delta, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext mean_delta)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_mean_delta_kernel, dimGrid, 14, &N, sizeof(cl_int), &delta.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean_delta.mem, sizeof(cl_mem));
}

void variance_delta_gpu(cl_mem_ext x, cl_mem_ext delta, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext variance_delta)
{
    size_t N = filters;
    dim2 dimGrid;
    dimGrid = dim2_create(N, 1);

    opencl_kernel(opencl_variance_delta_kernel, dimGrid, 18, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance_delta.mem, sizeof(cl_mem));
}


void l2normalize_gpu(cl_mem_ext x, cl_mem_ext dx, int batch, int filters, int spatial)
{
    size_t N = batch*spatial;
    dim2 dimN;
    dimN = opencl_gridsize(N);

    opencl_kernel(opencl_l2norm_kernel, dimN, 12, &N, sizeof(cl_int), &x.mem, sizeof(cl_mem), &dx.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int));
}
/*
void fast_mean_gpu(cl_mem_ext x, int batch, int filters, int spatial, cl_mem_ext mean)
{
    dim2 dimGrid;
    dimGrid = dim2_create(BLOCK, filters);

    opencl_kernel(opencl_fast_mean_kernel, dimGrid, 10, &x.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean.mem, sizeof(cl_mem));
}

void fast_variance_gpu(cl_mem_ext x, cl_mem_ext mean, int batch, int filters, int spatial, cl_mem_ext variance)
{
    dim2 dimGrid;
    dimGrid = dim2_create(BLOCK, filters);

    opencl_kernel(opencl_fast_variance_kernel, dimGrid, 12, &x.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance.mem, sizeof(cl_mem));
}

void fast_mean_delta_gpu(cl_mem_ext delta, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext mean_delta)
{
    dim2 dimGrid;
    dimGrid = dim2_create(BLOCK, filters);

    opencl_kernel(opencl_fast_mean_delta_kernel, dimGrid, 12, &delta.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &mean_delta.mem, sizeof(cl_mem));
}

void fast_variance_delta_gpu(cl_mem_ext x, cl_mem_ext delta, cl_mem_ext mean, cl_mem_ext variance, int batch, int filters, int spatial, cl_mem_ext variance_delta)
{
    dim2 dimGrid;
    dimGrid = dim2_create(BLOCK, filters);

    opencl_kernel(opencl_fast_variance_delta_kernel, dimGrid, 16, &x.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &mean.mem, sizeof(cl_mem), &variance.mem, sizeof(cl_mem), &batch, sizeof(cl_int), &filters, sizeof(cl_int), &spatial, sizeof(cl_int), &variance_delta.mem, sizeof(cl_mem));
}
*/

void axpy_offset_gpu(int N, float ALPHA, cl_mem_ext  X, int OFFX, int INCX, cl_mem_ext  Y, int OFFY, int INCY)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_axpy_kernel, dimGrid, 16, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}

void axpy_gpu(int N, float ALPHA, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    axpy_offset_gpu(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

void pow_offset_gpu(int N, float ALPHA, cl_mem_ext X, int OFFX, int INCX, cl_mem_ext Y, int OFFY, int INCY)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_pow_kernel, dimGrid, 16, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}

void pow_gpu(int N, float ALPHA, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    pow_offset_gpu(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

void copy_offset_gpu(int N, cl_mem_ext X, int OFFX, int INCX, cl_mem_ext Y, int OFFY, int INCY)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_copy_kernel, dimGrid, 14, &N, sizeof(cl_int), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &OFFY, sizeof(cl_int), &INCY, sizeof(cl_int));
}

void copy_gpu(int N, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    copy_offset_gpu(N, X, 0, INCX, Y, 0, INCY);
}

void mul_gpu(int N, cl_mem_ext X, int INCX, cl_mem_ext Y, int INCY)
{
    assert(N <= X.len && N <= Y.len && X.len <= Y.len);
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_mul_kernel, dimGrid, 10, &N, sizeof(cl_int), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int), &Y.mem, sizeof(cl_mem), &INCY, sizeof(cl_int));
}

void flatten_gpu(cl_mem_ext x, int spatial, int layers, int batch, int forward, cl_mem_ext out)
{
    int size = spatial*batch*layers;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_flatten_kernel, dimGrid, 14, &size, sizeof(cl_int), &x.mem, sizeof(cl_mem), &spatial, sizeof(cl_int), &layers, sizeof(cl_int), &batch, sizeof(cl_int), &forward, sizeof(cl_int), &out.mem, sizeof(cl_mem));
}


void reorg_gpu(cl_mem_ext x, int w, int h, int c, int batch, int stride, int forward, cl_mem_ext out)
{
    int size = w*h*c*batch;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_reorg_kernel, dimGrid, 18, &size, sizeof(cl_int), &x.mem, sizeof(cl_mem), &w, sizeof(cl_int), &h, sizeof(cl_int), &c, sizeof(cl_int), &batch, sizeof(cl_int), &stride, sizeof(cl_int), &forward, sizeof(cl_int), &out.mem, sizeof(cl_mem));
}

void mask_gpu(int N, cl_mem_ext X, float mask_num, cl_mem_ext mask, float val)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_mask_kernel, dimGrid, 10, &N, sizeof(cl_int), &X.mem, sizeof(cl_mem), &mask_num, sizeof(cl_int), &mask.mem, sizeof(cl_mem), &val, sizeof(float));
}

void const_offset_gpu(int N, float ALPHA, cl_mem_ext X, int OFFX, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_const_kernel, dimGrid, 10, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int));
}


void const_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    const_offset_gpu(N, ALPHA, X, 0, INCX);
}


void constrain_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_constrain_kernel, dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void add_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_add_kernel, dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void scal_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_scal_kernel, dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void supp_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_supp_kernel, dimGrid, 8, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &INCX, sizeof(cl_int));
}


void fill_offset_gpu(int N, float ALPHA, cl_mem_ext X, int OFFX, int INCX)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(N);

    opencl_kernel(opencl_fill_kernel, dimGrid, 10, &N, sizeof(cl_int), &ALPHA, sizeof(cl_float), &X.mem, sizeof(cl_mem), &OFFX, sizeof(cl_int), &INCX, sizeof(cl_int));
}

void fill_gpu(int N, float ALPHA, cl_mem_ext X, int INCX)
{
    fill_offset_gpu(N, ALPHA, X, 0, INCX);
}

void shortcut_gpu(int batch, int w1, int h1, int c1, cl_mem_ext add, int w2, int h2, int c2, float s1, float s2, cl_mem_ext out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);
    opencl_kernel(opencl_shortcut_kernel, dimGrid, 34, &size, sizeof(cl_int), &minw, sizeof(cl_int), &minh, sizeof(cl_int), &minc, sizeof(cl_int), &stride, sizeof(cl_int), &sample, sizeof(cl_int), &batch, sizeof(cl_int), &w1, sizeof(cl_int), &h1, sizeof(cl_int), &c1, sizeof(cl_int), &add.mem, sizeof(cl_mem), &w2, sizeof(cl_int), &h2, sizeof(cl_int), &c2, sizeof(cl_int), &s1, sizeof(float), &s2, sizeof(float), &out.mem, sizeof(cl_mem));
}


void smooth_l1_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_smooth_l1_kernel, dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void softmax_x_ent_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_softmax_x_ent_kernel, dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void logistic_x_ent_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_logistic_x_ent_kernel, dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void l2_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_l2_kernel, dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}


void l1_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_l1_kernel, dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}


void wgan_gpu(int n, cl_mem_ext pred, cl_mem_ext truth, cl_mem_ext delta, cl_mem_ext error)
{
    dim2 dimN;
    dimN = opencl_gridsize(n);
    opencl_kernel(opencl_wgan_kernel, dimN, 10, &n, sizeof(cl_int), &pred.mem, sizeof(cl_mem), &truth.mem, sizeof(cl_mem), &delta.mem, sizeof(cl_mem), &error.mem, sizeof(cl_mem));
}

void deinter_gpu(int NX, cl_mem_ext X, int NY, cl_mem_ext Y, int B, cl_mem_ext OUT)
{
    dim2 dimN;
    dimN = opencl_gridsize((NX+NY)*B);

    opencl_kernel(opencl_deinter_kernel, dimN, 12, &NX, sizeof(cl_int), &X.mem, sizeof(cl_mem), &NY, &Y.mem, sizeof(cl_mem), &B, sizeof(cl_int), &OUT.mem, sizeof(cl_mem));
}

void inter_gpu(int NX, cl_mem_ext X, int NY, cl_mem_ext Y, int B, cl_mem_ext OUT)
{
    dim2 dimN;
    dimN = opencl_gridsize((NX+NY)*B);

    opencl_kernel(opencl_inter_kernel, dimN, 12, &NX, sizeof(cl_int), &X.mem, sizeof(cl_mem), &NY, &Y.mem, sizeof(cl_mem), &B, sizeof(cl_int), &OUT.mem, sizeof(cl_mem));
}

void weighted_sum_gpu(cl_mem_ext a, cl_mem_ext b, cl_mem_ext s, int num, cl_mem_ext c)
{
    dim2 dimNum;
    dimNum = opencl_gridsize(num);

    opencl_kernel(opencl_weighted_sum_kernel, dimNum, 10, &num, sizeof(cl_int), &a.mem, sizeof(cl_mem), &b.mem, sizeof(cl_mem), &s.mem, sizeof(cl_mem), &c.mem, sizeof(cl_mem));
}


void weighted_delta_gpu(cl_mem_ext a, cl_mem_ext b, cl_mem_ext s, cl_mem_ext da, cl_mem_ext db, cl_mem_ext ds, int num, cl_mem_ext dc)
{
    dim2 dimNum;
    dimNum = opencl_gridsize(num);

    opencl_kernel(opencl_weighted_delta_kernel, dimNum, 16, &num, sizeof(cl_int), &a.mem, sizeof(cl_mem), &b.mem, sizeof(cl_mem), &s.mem, sizeof(cl_mem), &da.mem, sizeof(cl_mem), &db.mem, sizeof(cl_mem), &ds.mem, sizeof(cl_mem), &dc.mem, sizeof(cl_mem));
}


void mult_add_into_gpu(int num, cl_mem_ext a, cl_mem_ext b, cl_mem_ext c)
{
    dim2 dimNum;
    dimNum = opencl_gridsize(num);

    opencl_kernel(opencl_mult_add_into_kernel, dimNum, 8, &num, sizeof(cl_int), &a.mem, sizeof(cl_mem), &b.mem, sizeof(cl_mem), &c.mem, sizeof(cl_mem));
}

void softmax_tree(cl_mem_ext input, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier)
{
    softmax_offset_tree(input, 0, spatial, batch, stride, temp, output, hier);
}

void softmax_offset_tree(cl_mem_ext input, int offset, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier)
{
    cl_mem_ext tree_groups_size = opencl_make_int_array(hier.group_size, hier.groups);
    cl_mem_ext tree_groups_offset = opencl_make_int_array(hier.group_offset, hier.groups);

    int num = spatial*batch*hier.groups;

    dim2 dimBatch;
    dimBatch = opencl_gridsize(num);

    opencl_kernel(opencl_softmax_tree_kernel, dimBatch, 20,
                  &input.mem, sizeof(cl_mem),
                  &offset, sizeof(cl_int),
                  &spatial, sizeof(cl_int),
                  &batch, sizeof(cl_int),
                  &stride, sizeof(cl_int),
                  &temp, sizeof(cl_float),
                  &output.mem, sizeof(cl_mem),
                  &hier.groups, sizeof(cl_int),
                  &tree_groups_size.mem, sizeof(cl_mem),
                  &tree_groups_offset.mem, sizeof(cl_mem)
    );

    opencl_free(tree_groups_size);
    opencl_free(tree_groups_offset);
}

void softmax_offset_gpu(cl_mem_ext input, int offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_ext output)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(batch * groups);
    opencl_kernel(opencl_softmax_kernel, dimBatch, 20, &input.mem, sizeof(cl_mem), &offset, sizeof(cl_int), &n, sizeof(cl_int), &batch, sizeof(cl_int), &batch_offset, sizeof(cl_int), &groups, sizeof(cl_int), &group_offset, sizeof(cl_int), &stride, sizeof(cl_int), &temp, sizeof(cl_float), &output.mem, sizeof(cl_mem));
}

void softmax_gpu(cl_mem_ext input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, cl_mem_ext output)
{
    softmax_offset_gpu(input, 0, n, batch, batch_offset, groups, group_offset, stride, temp, output);
}

void softmax_tree_gpu(cl_mem_ext input, int spatial, int batch, int stride, float temp, cl_mem_ext output, tree hier)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(batch * hier.groups);

    float * size = calloc(hier.groups, sizeof(float));
    float * offset = calloc(hier.groups, sizeof(float));

    cl_mem_ext tree_groups_size = opencl_make_array(size, hier.groups);
    cl_mem_ext tree_groups_offset = opencl_make_array(offset, hier.groups);

    opencl_kernel(opencl_softmax_tree_kernel, dimBatch, 18,
        &input.mem, sizeof(cl_mem),
        &spatial, sizeof(cl_int),
        &batch, sizeof(cl_int),
        &stride, sizeof(cl_int),
        &temp, sizeof(cl_float),
        &output.mem, sizeof(cl_mem),
        &hier.groups, sizeof(cl_int),
        &tree_groups_size.mem, sizeof(cl_mem),
        &tree_groups_offset.mem, sizeof(cl_mem)
    );

    opencl_free(tree_groups_size);
    opencl_free(tree_groups_offset);

    free(size);
    free(offset);
}

void scale_mask_gpu(int N, cl_mem_ext X, float mask_num, cl_mem_ext mask, float scale)
{
    dim2 dimBatch;
    dimBatch = opencl_gridsize(N);

    opencl_kernel(opencl_scale_mask_kernel, dimBatch, 10,
        &N, sizeof(cl_int),
        &X.mem, sizeof(cl_mem),
        &mask_num, sizeof(cl_float),
        &mask.mem, sizeof(cl_mem),
        &scale, sizeof(cl_float)
    );
}

void dot_error_gpu(layer l)
{
    dim2 dimGrid;
    dimGrid = opencl_gridsize(l.n*l.n);

    int size = l.out_w * l.out_h;

    opencl_kernel(opencl_dot_kernel, dimGrid, 12,
         &l.output_gpu.mem, sizeof(cl_mem),
         &l.dot, sizeof(cl_float),
         &l.batch, sizeof(cl_int),
         &l.n, sizeof(cl_int),
         &size, sizeof(cl_int),
         &l.delta_gpu.mem, sizeof(cl_mem));
}


void upsample_gpu(cl_mem_ext in, int w, int h, int c, int batch, int stride, int forward, float scale, cl_mem_ext out)
{
    size_t size = w*h*c*batch*stride*stride;

    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_upsample_kernel, dimGrid, 20,
                  &size, sizeof(cl_int),
                  &in.mem, sizeof(cl_mem),
                  &w, sizeof(cl_int),
                  &h, sizeof(cl_int),
                  &c, sizeof(cl_int),
                  &batch, sizeof(cl_int),
                  &stride, sizeof(cl_int),
                  &forward, sizeof(cl_int),
                  &scale, sizeof(cl_float),
                  &out.mem, sizeof(cl_mem));
}

#endif // GPU
