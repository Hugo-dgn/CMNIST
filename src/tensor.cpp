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
                std::vector<size_t> stride, 
                size_t offset
            )
{
    _storage = storage;
    _shape = shape;
    _stride = stride;
    _offset = offset;
}

// Methods

// getter

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

// Others

std::size_t Tensor::numel() const
{
    std::size_t num = 1;
    for (const std::size_t& d : _shape)
    {
        num = num * d;
    }
    return num;
}

// Overloads

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

// print

void print(
        std::ostream& os,
        const std::size_t* shape, 
        const std::size_t* stride,
        const float* data,
        std::size_t rest
    )
{

    std::size_t n = shape[0];
    std::size_t s = stride[0];

    os << "[";

    if (rest > 1)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            if (rest == 2)
                os << "\n ";
            print(os, shape + 1, stride + 1, data + i * s, rest-1);
        }
        if (rest == 2)
            os << "\n";
    }
    else
    {
        for (std::size_t i = 0; i < n; i++)
        {
            os << " " << data[i * s] << " ";
        }
    }

    os << "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{

    print(
        os, tensor.shape().data(), 
        tensor.stride().data(), 
        &tensor.data()[tensor.offset()], 
        tensor.shape().size()
    );

    return os;
}