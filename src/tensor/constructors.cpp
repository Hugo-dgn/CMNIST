#include <vector>
#include <iostream>

#include "tensor.hpp"

//Helpers

template <typename T>
void flatten(const T& x, std::vector<float>& out)
{
    if constexpr (std::is_same_v<T, float>)
    {
        out.push_back(x);
    }
    else
    {
        for (const auto& e :x)
        {
            flatten(e, out);
        }
    }
}

template <typename T>
void check_shape(const T& x, std::vector<std::size_t>& shape, std::size_t depth)
{
    if constexpr (std::is_same_v<T, float>)
    {
        return;
    }
    else
    {
        std::size_t size = x.size();
        if (shape.size() == depth)
        {
            shape.push_back(size);
        }
        else
        {
            if (shape[depth] != size)
            {
                throw std::invalid_argument("Non-homogeneous tensor");
            }
        }
        for (const auto& e : x)
        {
            check_shape(e, shape, depth + 1);
        }
    }
}

void set_stride(const std::vector<std::size_t>& shape, std::vector<std::size_t>& stride)
{
    if (shape.size() == 0)
    {
        stride = {};
    }
    else
    {
        std::size_t s = 1;
        stride.resize(shape.size());
        for (std::size_t i = shape.size(); i-- > 0; )
        {
            stride[i] = s;
            s = s*shape[i];
        }
    }
}

template <typename T>
void build_tensor(const T& data, 
    std::shared_ptr<TensorStorage>& storage, 
    std::vector<std::size_t>& shape, 
    std::vector<std::size_t>& stride,
    std::size_t& offset)
{

    storage = std::make_shared<TensorStorage>();

    check_shape(data, shape, 0);
    flatten(data, storage->data);
    set_stride(shape, stride);
    offset = 0;
}

// Class

// Constructors

Tensor::Tensor(std::vector<float> data)
{
    build_tensor<std::vector<float>>(data, _storage, _shape, _stride, _offset);
}

Tensor::Tensor(std::vector<std::vector<float>> data)
{
    build_tensor<std::vector<std::vector<float>>>(data, _storage, _shape, _stride, _offset);
}

Tensor::Tensor(std::vector<std::vector<std::vector<float>>> data)
{
    build_tensor<std::vector<std::vector<std::vector<float>>>>(data, _storage, _shape, _stride, _offset);
}

Tensor::Tensor(
                std::shared_ptr<TensorStorage> storage, 
                std::vector<std::size_t> shape,
                size_t offset
            )
{
    _storage = storage;
    _shape = shape;
    set_stride(shape, _stride);
    _offset = offset;
}

Tensor::Tensor(
                std::vector<float> data, 
                std::vector<std::size_t> shape, 
                size_t offset
            )
{

    auto storage = std::make_shared<TensorStorage>();
    storage->data = data;

    _storage = storage;
    _shape = shape;
    set_stride(shape, _stride);
    _offset = offset;
}