#include <vector>
#include <numeric>
#include <functional>

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
    std::vector<float> data(numel);

    return Tensor(data, shape, 0);
}