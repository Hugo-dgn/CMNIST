#include <vector>
#include <iostream>

#include "tensor.hpp"

Tensor Tensor::operator[](std::size_t i) const
{

    if (i >= _shape[0])
    {
        throw std::out_of_range(
            "Tensor index out of range: i=" + std::to_string(i) +
            ", valid range: [0, " + std::to_string(_shape[0] - 1) + "]"
        );
    }

    std::vector<std::size_t> new_shape;
    for (std::size_t i = 1; i < _shape.size(); i++)
    {
        new_shape.push_back(_shape[i]);
    }

    std::vector<std::size_t> new_stride;
    for (std::size_t i = 1; i < _stride.size(); i++)
    {
        new_stride.push_back(_stride[i]);
    }

    return Tensor(_storage, new_shape, new_stride, _offset + i*_stride[0]);
}