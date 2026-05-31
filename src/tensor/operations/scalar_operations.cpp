#include <vector>
#include <iostream>
#include <functional>
#include <cstring>

#include "tensor.hpp"

template<bool Swap, typename Func>
void apply(
        float* src,
        float* dst,
        std::size_t numel,
        float x,
        Func func
    )
{
    #pragma omp simd
    for (std::size_t i = 0; i < numel; i++)
    {
        if constexpr (Swap)
            dst[i] = func(src[i], x);
        else
            dst[i] = func(x, src[i]);
    }
}

template<typename Func>
void inplace_apply(
        Tensor& tensor, 
        float x,
        Func func
    )
{
    apply<true>(
        tensor.point(),
        tensor.point(),
        tensor.numel(),
        x,
        func
    );
}

template<bool Swap, typename Func>
Tensor create_apply(
                    float x, 
                    const Tensor& tensor, 
                    Func func
                )
{
    Tensor new_tensor = allocateTensor(tensor.shape());

    apply<Swap>(
            tensor.point(),
            new_tensor.point(),
            tensor.numel(),
            x,
            func
    );
    
    return new_tensor;
}

constexpr auto add = [](float a, float b) { return a + b; };

constexpr auto substract = [](float a, float b) { return a - b; };

constexpr auto multiply = [](float a, float b) { return a * b; };

constexpr auto divide = [](float a, float b) { return a / b; };

// Create

// right

Tensor operator+(float x, const Tensor& tensor)
{
    return create_apply<false>(x, tensor, add);
}

Tensor operator-(float x, const Tensor& tensor)
{  
    return create_apply<false>(x, tensor, substract);
}

Tensor operator*(float x, const Tensor& tensor)
{
    return create_apply<false>(x, tensor, multiply);
}

Tensor operator/(float x, const Tensor& tensor)
{
    return create_apply<false>(x, tensor, divide);
}

// left

Tensor operator+(const Tensor& tensor, float x)
{
    return create_apply<true>(x, tensor, add);
}

Tensor operator-(const Tensor& tensor, float x)
{
    return create_apply<true>(x, tensor, substract);
}

Tensor operator*(const Tensor& tensor, float x)
{
    return create_apply<true>(x, tensor, multiply);
}

Tensor operator/(const Tensor& tensor, float x)
{
    return create_apply<true>(1.0f / x, tensor, multiply);
}

// Inplace

Tensor& Tensor::operator+=(float x)
{
    inplace_apply(*this, x, add);
    return *this;
}

Tensor& Tensor::operator-=(float x)
{
    inplace_apply(*this, x, substract);
    return *this;
}

Tensor& Tensor::operator*=(float x)
{
    inplace_apply(*this, x, multiply);
    return *this;
}

Tensor& Tensor::operator/=(float x)
{
    inplace_apply(*this, 1.0f / x, multiply);
    return *this;
}