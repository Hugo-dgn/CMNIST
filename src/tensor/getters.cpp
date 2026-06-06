#include <vector>
#include <iostream>

#include "tensor.hpp"

const std::vector<std::size_t>& Tensor::shape() const
{
    return _shape;
}

const std::vector<std::size_t>& Tensor::stride() const
{
    return _stride;
}

size_t Tensor::offset() const
{
    return _offset;
}

const std::vector<float>& Tensor::data() const
{
    return _storage->data;
}

std::vector<float>& Tensor::rdata()
{
    return _storage->data;
}

const float& Tensor::item() const
{
    if (this->numel() == 1)
    {
        return _storage->data[_offset];
    }
    else
    {
        throw std::runtime_error("Only single element tensor support 'item' method");
    }
}