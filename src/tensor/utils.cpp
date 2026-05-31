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

float* Tensor::point() const
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

    float* src = this->point();

    for (std::size_t i = 0; i < _shape[0]; i++)
    {
        for (std::size_t j = 0; j < _shape[1]; j++)
        {
            data[j*stride[0] + i] = src[i*_stride[0] + j];
        }
    }

    return Tensor(data, shape, 0);
}