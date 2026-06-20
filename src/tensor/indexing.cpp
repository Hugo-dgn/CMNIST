#include <vector>
#include <iostream>

#include "tensor.hpp"

Tensor Tensor::operator[](std::size_t i) const
{

    if (i >= this->shape()[0])
    {
        throw std::out_of_range(
            "Tensor index out of range: i=" + std::to_string(i) +
            ", valid range: [0, " + std::to_string(this->shape()[0] - 1) + "]"
        );
    }

    std::vector<std::size_t> new_shape;
    for (std::size_t i = 1; i < this->shape().size(); i++)
    {
        new_shape.push_back(this->shape()[i]);
    }

    bool requires_grad = this->requires_grad();
    return Tensor(_handle->storage, new_shape, this->offset() + i*this->stride()[0], requires_grad);
}