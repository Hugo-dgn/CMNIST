#include <vector>
#include <iostream>
#include <functional>

#include "tensor.hpp"

void apply(
        const float& x,
        std::function<float(float, float)> func,
        std::vector<float>& res,
        const std::size_t* shape, 
        const std::size_t* stride,
        const float* data,
        std::size_t rest,
        bool swap
    )
{

    std::size_t n = shape[0];
    std::size_t s = stride[0];

    if (rest > 1)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            apply(x, func, res, shape + 1, stride + 1, data + i * s, rest-1, swap);
        } 
    }
    else
    {
        for (std::size_t i = 0; i < n; i++)
        {
            if (swap)
            {
                res.push_back(func(data[s*i], x));
            }
            else
            {
                res.push_back(func(x, data[s*i]));
            }
        } 
    }
}

float add(float x, float y)
{
    return x + y;
}

float substract(float x, float y)
{
    return y - x;
}

float multiply(float x, float y)
{
    return x*y;
}

float divide(float x, float y)
{
    return y / x;
}

// right

Tensor operator+(const float& x, const Tensor& tensor)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, add, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), true);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}

Tensor operator-(const float& x, const Tensor& tensor)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, substract, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), true);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}

Tensor operator*(const float& x, const Tensor& tensor)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, multiply, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), true);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}

Tensor operator/(const float& x, const Tensor& tensor)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, divide, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), true);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}

// left

Tensor operator+(const Tensor& tensor, const float& x)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, add, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), false);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}

Tensor operator-(const Tensor& tensor, const float& x)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, substract, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), false);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}

Tensor operator*(const Tensor& tensor, const float& x)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, multiply, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), false);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}

Tensor operator/(const Tensor& tensor, const float& x)
{

    std::size_t offset = 0;

    std::vector<float> res = {};
    apply(x, divide, res, tensor.shape().data(), tensor.stride().data(), tensor.point(), tensor.shape().size(), false);

    return Tensor(res, tensor.shape(), tensor.stride(), offset);
}