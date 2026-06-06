#include <vector>
#include <numeric>
#include <functional>
#include <iostream>

#include "tensor.hpp"

std::size_t Tensor::numel() const
{
    std::size_t num = 1;
    for (const std::size_t& d : _shape)
    {
        num = num * d;
    }
    return num;
}

float* Tensor::point()
{
    return _storage->data.data() + _offset;
}

const float* Tensor::point() const
{
    return _storage->data.data() + _offset;
}

Tensor allocateTensor(std::vector<std::size_t> shape)
{

    std::size_t numel = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::vector<float> data(numel, 0.0f);

    return Tensor(data, shape, 0);
}

Tensor Tensor::copy() const
{
    return Tensor(_storage->data, _shape, _offset);
}

Tensor Tensor::transpose() const
{
    if (_shape.size() != 2)
    {
        throw std::runtime_error("Only 2D tensor can be transposed.");
    }

    std::vector<std::size_t> shape = {_shape[1], _shape[0]};
    std::vector<std::size_t> stride = {shape[1], 1};

    std::size_t numel = this->numel();
    std::vector<float> data(numel, 0.0f);

    const float* src = this->point();

    constexpr std::size_t BLOCK_SIZE = 64;

    #pragma omp parallel for collapse(2)
    for (std::size_t ii = 0; ii < _shape[0]; ii += BLOCK_SIZE)
    for (std::size_t jj = 0; jj < _shape[1]; jj += BLOCK_SIZE)
    {
        const std::size_t i_max = std::min(ii + BLOCK_SIZE, _shape[0]);

        const std::size_t j_max = std::min(jj + BLOCK_SIZE, _shape[1]);

        for (std::size_t i = ii; i < i_max; ++i)
        {
            for (std::size_t j = jj; j < j_max; ++j)
            {
                data[j * stride[0] + i] =
                    src[i * _stride[0] + j];
            }
        }
    }

    return Tensor(data, shape, 0);
}