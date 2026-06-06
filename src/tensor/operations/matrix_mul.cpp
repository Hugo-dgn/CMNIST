#include <vector>
#include <algorithm>
#include <iostream>

#include "tensor.hpp"

bool check_input_shape_match(
        std::vector<std::size_t> shape1, 
        std::vector<std::size_t> shape2)
{
    if (shape1.size() != 2 || shape2.size() != 2)
    {
        return false;
    }
    if (shape1[1] != shape2[0])
    {
        return false;
    }
    return true;
};

bool check_output_shape_match(
        std::vector<std::size_t> shape1, 
        std::vector<std::size_t> shape2,
        std::vector<std::size_t> shape3)
{
    if (shape1.size() != 2 || shape2.size() != 2 || shape3.size() != 2)
    {
        return false;
    }
    if (shape1[0] != shape3[0] || shape2[1] != shape3[1])
    {
        return false;
    }
    return true;
};

void inplace_matmul(const Tensor& tensor1, const Tensor& tensor2, Tensor& tensor3)
{
    std::vector<std::size_t> shape1 = tensor1.shape();
    std::vector<std::size_t> shape2 = tensor2.shape();
    std::vector<std::size_t> shape3 = tensor3.shape();

    if (!check_input_shape_match(shape1, shape2))
    {
        throw std::runtime_error("Input shape mismatch.");
    }
    if (!check_output_shape_match(shape1, shape2, shape3))
    {
        throw std::runtime_error("Output shape mismatch.");
    }

    const float* data1 = tensor1.point();
    const float* data2 = tensor2.point();
    float* data3 = tensor3.point();

    const std::size_t stride1 = tensor1.stride()[0];
    const std::size_t stride2 = tensor2.stride()[0];
    const std::size_t stride3 = tensor3.stride()[0];

    constexpr std::size_t TILE_SIZE = 32;

    const std::size_t iMax = shape3[0];
    const std::size_t jMax = shape3[1];
    const std::size_t kMax = shape2[0];

    #pragma omp parallel for collapse(2)
    for (std::size_t ii = 0; ii < iMax; ii += TILE_SIZE)
    for (std::size_t jj = 0; jj < jMax; jj += TILE_SIZE)
    for (std::size_t kk = 0; kk < kMax; kk += TILE_SIZE)
    {

        std::size_t imax = std::min(iMax - ii, TILE_SIZE);
        std::size_t jmax = std::min(jMax - jj, TILE_SIZE);
        std::size_t kmax = std::min(kMax - kk, TILE_SIZE);

        for (std::size_t i = ii; i < imax; i++)
        for (std::size_t k = kk; k < kmax; k++)
        {

            const float x = data1[i*stride1 + k];

            #pragma omp simd
            for (std::size_t j = jj; j < jmax; j++)
            {
                data3[i*stride3 + j] += x * data2[k*stride2 + j];
            }
        }
    }
};

Tensor matmul(const Tensor tensor1, const Tensor tensor2)
{
    std::vector<std::size_t> shape1 = tensor1.shape();
    std::vector<std::size_t> shape2 = tensor2.shape();

    if (!check_input_shape_match(shape1, shape2))
    {
        throw std::runtime_error("Shape mismatch.");
    }

    std::vector<std::size_t> shape3 = {shape1[0], shape2[1]};
    Tensor tensor3 = allocateTensor(shape3);
    
    inplace_matmul(tensor1, tensor2, tensor3);
    return tensor3;
};

// Transposed

void inplace_matmul_transpose(const Tensor& tensor1, const Tensor& ttensor2, Tensor& tensor3)
{
    std::vector<std::size_t> shape1 = tensor1.shape();
    std::vector<std::size_t> tshape2 = ttensor2.shape();
    std::vector<std::size_t> shape3 = tensor3.shape();

    std::vector<std::size_t> shape2(tshape2.rbegin(), tshape2.rend());

    if (!check_input_shape_match(shape1, shape2))
    {
        throw std::runtime_error("Input shape mismatch.");
    }
    if (!check_output_shape_match(shape1, shape2, shape3))
    {
        throw std::runtime_error("Output shape mismatch.");
    }

    const float* data1 = tensor1.point();
    const float* tdata2 = ttensor2.point();
    float* data3 = tensor3.point();

    const std::size_t stride1 = tensor1.stride()[0];
    const std::size_t tstride2 = ttensor2.stride()[0];
    const std::size_t stride3 = tensor3.stride()[0];

    constexpr std::size_t TILE_SIZE = 32;

    const std::size_t iMax = shape3[0];
    const std::size_t jMax = shape3[1];
    const std::size_t kMax = shape2[0];

    #pragma omp parallel for collapse(2)
    for (std::size_t ii = 0; ii < iMax; ii += TILE_SIZE)
    for (std::size_t jj = 0; jj < jMax; jj += TILE_SIZE)
    for (std::size_t kk = 0; kk < kMax; kk += TILE_SIZE)
    {

        std::size_t imax = std::min(iMax - ii, TILE_SIZE);
        std::size_t jmax = std::min(jMax - jj, TILE_SIZE);
        std::size_t kmax = std::min(kMax - kk, TILE_SIZE);

        for (std::size_t i = ii; i < imax; i++)
        for (std::size_t j = jj; j < jmax; j++)
        {

            float x = 0;

            #pragma omp simd reduction(+:x)
            for (std::size_t k = kk; k < kmax; k++)
            {
                x += data1[i * stride1 + k] * tdata2[j * tstride2 + k];
            }
            data3[i * stride3 + j] = x;
        }
    }
}

Tensor matmul_transpose(const Tensor tensor1, const Tensor ttensor2)
{
    std::vector<std::size_t> shape1 = tensor1.shape();
    std::vector<std::size_t> tshape2 = ttensor2.shape();

    std::vector<std::size_t> shape2(tshape2.rbegin(), tshape2.rend());

    if (!check_input_shape_match(shape1, shape2))
    {
        throw std::runtime_error("Shape mismatch.");
    }

    std::vector<std::size_t> shape3 = {shape1[0], shape2[1]};
    Tensor tensor3 = allocateTensor(shape3);
    
    inplace_matmul_transpose(tensor1, ttensor2, tensor3);
    return tensor3;
};