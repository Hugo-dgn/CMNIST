#include <vector>
#include <iostream>
#include <functional>
#include <cstring>

#include "tensor.hpp"

template<typename Func>
void apply(
        float* src,
        float* dst,
        std::size_t numel,
        Func func
    )
{
    #pragma omp simd
    for (std::size_t i = 0; i < numel; i++)
    {
        src[i] = func(dst[i], src[i]);
    }
}

template<typename Func>
void inplace_apply(
        Tensor& tensor1, 
        const Tensor& tensor2, 
        Func func
    )
{

    if (tensor1.shape().size() != tensor2.shape().size())
    {
        throw std::runtime_error("Shape mismatch");
    }
    for (std::size_t i = 0; i < tensor1.shape().size(); i++)
    {
        if (tensor1.shape()[i] != tensor2.shape()[i])
        {
            throw std::runtime_error("Shape mismatch");
        }
    }

    apply(
        tensor1.point(),
        tensor2.point(),
        tensor1.numel(),
        func
    );
}

constexpr auto add = [](float a, float b) { return a + b; };

constexpr auto substract = [](float a, float b) { return b - a; };

constexpr auto multiply = [](float a, float b) { return a * b; };

constexpr auto divide = [](float a, float b) { return b / a; };

// Inplace

Tensor& Tensor::operator+=(const Tensor& tensor)
{
    inplace_apply(*this, tensor, add);
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& tensor)
{
    inplace_apply(*this, tensor, substract);
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& tensor)
{
    inplace_apply(*this, tensor, multiply);
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& tensor)
{
    inplace_apply(*this, tensor, divide);
    return *this;
}

// New tensor

Tensor Tensor::operator+(const Tensor& tensor) const
{
    Tensor new_tensor = this->copy();
    new_tensor += tensor;
    return new_tensor;
}

Tensor Tensor::operator-(const Tensor& tensor) const
{
    Tensor new_tensor = this->copy();
    new_tensor -= tensor;
    return new_tensor;
}

Tensor Tensor::operator*(const Tensor& tensor) const
{
    Tensor new_tensor = this->copy();
    new_tensor *= tensor;
    return new_tensor;
}

Tensor Tensor::operator/(const Tensor& tensor) const
{
    Tensor new_tensor = this->copy();
    new_tensor /= tensor;
    return new_tensor;
}