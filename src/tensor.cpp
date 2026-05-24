#include <vector>
#include <iostream>

#include "tensor.h"

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

void set_stride(const std::vector<std::size_t> shape, std::vector<std::size_t>& stride)
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
void build_tensor(const T& raw_data, 
    std::vector<float>& data, 
    std::vector<std::size_t>& shape, 
    std::vector<std::size_t>& stride)
{
    check_shape(raw_data, shape, 0);
    flatten(raw_data, data);
    set_stride(shape, stride);
}

// Class

// Constructors

Tensor::Tensor(std::vector<float> data)
{
    build_tensor<std::vector<float>>(data, _data, _shape, _stride);
}

Tensor::Tensor(std::vector<std::vector<float>> data)
{
    build_tensor<std::vector<std::vector<float>>>(data, _data, _shape, _stride);
}

Tensor::Tensor(std::vector<std::vector<std::vector<float>>> data)
{
    build_tensor<std::vector<std::vector<std::vector<float>>>>(data, _data, _shape, _stride);
}

Tensor::Tensor(std::vector<float> data, std::vector<std::size_t> shape)
{

    std::size_t n = 1;
    for (std::size_t i : shape)
    {
        n = n * i;
    }

    if (n != data.size())
    {
        throw std::invalid_argument("The data can't match the shape.");
    }

    _data = data;
    _shape = shape;
    set_stride(_shape, _stride);
}

TensorView::TensorView(float* data, std::vector<std::size_t> shape, std::vector<std::size_t> stride, std::size_t offset, std::size_t dim)
{
    _data = data;
    _shape = shape;
    _stride = stride;
    _offset = offset;
    _dim = dim;
}

// Methods

// getter

template <typename T>
const float& get_item(const T& tensor, const std::size_t offset)
{
    if (tensor.numel() == 1)
    {
        return tensor.data()[offset];
    }
    else
    {
        throw std::runtime_error("Only single element tensor support 'item' method");
    }
}

const std::vector<std::size_t>& Tensor::shape() const
{
    return _shape;
}

const std::vector<std::size_t>& Tensor::stride() const
{
    return _stride;
}

const std::vector<float>& Tensor::data() const
{
    return _data;
}

const float& Tensor::item() const
{
    return get_item(*this, 0);
}

const std::vector<std::size_t>& TensorView::shape() const
{
    return _shape;
}

const std::vector<std::size_t>& TensorView::stride() const
{
    return _stride;
}

const float* TensorView::data() const
{
    return _data;
}

const float& TensorView::item() const
{
    return get_item(*this, _offset);
}

std::size_t TensorView::offset() const
{
    return _offset;
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

std::size_t TensorView::numel() const
{
    std::size_t num = 1;
    for (std::size_t i = _dim; i < _shape.size(); i++)
    {
        num = num * _shape[i];
    }
    return num;
}

// Overloads

TensorView Tensor::operator[](std::size_t i)
{
    const std::size_t dim0 = _shape[0];

    if (i >= dim0)
    {
        throw std::out_of_range(
            "Tensor index out of range: i=" + std::to_string(i) +
            ", valid range: [0, " + std::to_string(dim0 - 1) + "]"
        );
    }
    return TensorView(_data.data(), _shape, _stride, i * _stride[0], 1);
}

TensorView TensorView::operator[](std::size_t i)
{
    const std::size_t dim0 = _shape[_dim];

    if (i >= dim0)
    {
        throw std::out_of_range(
            "Tensor index out of range: i=" + std::to_string(i) +
            ", valid range: [0, " + std::to_string(dim0 - 1) + "]"
        );
    }
    return TensorView(_data, _shape, _stride, _offset + i * _stride[_dim], _dim + 1);
}

// print

template <typename T>
void print(std::ostream& os, std::size_t i, T& data, std::vector<std::size_t> stride, std::size_t start, std::size_t end)
{
    for (std::size_t j = 0; j + 1 < stride.size(); ++j)
        {
            if ((i% stride[j] == 0) | (i == start))
            {
                os << "[";
            }

            os << data[i];

            if (((i + 1)% stride[j] == 0) | (i + 1 == end))
            {
                os << "]";
            }
            else
            {
                os << " ";
            }
        }
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{

    std::vector<float> data = tensor.data();
    std::vector<std::size_t> stride = tensor.stride();

    std::size_t start = 0;
    std::size_t end = tensor.numel();

    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        print(os, i, data, stride, start, end);
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const TensorView& tensor)
{

    const float* data = tensor.data();
    std::vector<std::size_t> stride = tensor.stride();

    std::size_t start = tensor.offset();
    std::size_t end = start + tensor.numel();

    for (std::size_t i = tensor.offset(); i < tensor.offset() + tensor.numel(); i++)
    {
        print(os, i, data, stride, start, end);
    }
    return os;
}