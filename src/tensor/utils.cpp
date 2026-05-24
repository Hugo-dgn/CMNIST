#include <vector>

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