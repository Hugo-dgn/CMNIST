#include <vector>
#include <iostream>
#include <functional>
#include <cstring>

#include "tensor.hpp"

template<typename Func>
void apply(
        float* src,
        const float* dst,
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
    Tensor res = this->copy();
    res += tensor;

    if (tensor.requires_grad() || this->requires_grad())
    {

        std::shared_ptr<TensorHandle> handle1 = this->handle();
        std::shared_ptr<TensorHandle> handle2 = tensor.handle();

        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {handle1, handle2};
        GradFn grad_fn =
        [handle1, handle2](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            Tensor tmp1 = Tensor(handle1);
            Tensor tmp2 = Tensor(handle2);
            Tensor grad1 = upstream;
            Tensor grad2 = upstream;
            const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad1.handle(), grad2.handle()};
            return grad_handles;
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }

    return res;
}

Tensor Tensor::operator-(const Tensor& tensor) const
{
    Tensor res = this->copy();
    res -= tensor;

    if (tensor.requires_grad() || this->requires_grad())
    {

        std::shared_ptr<TensorHandle> handle1 = this->handle();
        std::shared_ptr<TensorHandle> handle2 = tensor.handle();

        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {handle1, handle2};
        GradFn grad_fn =
        [handle1, handle2](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            Tensor tmp1 = Tensor(handle1);
            Tensor tmp2 = Tensor(handle2);
            Tensor grad1 = upstream;
            Tensor grad2 = -1.0f * upstream;
            const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad1.handle(), grad2.handle()};
            return grad_handles;
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }

    return res;
}

Tensor Tensor::operator*(const Tensor& tensor) const
{
    Tensor res = this->copy();
    res *= tensor;
    res.handle()->storage->requires_grad = tensor.handle()->storage->requires_grad;

    if (tensor.requires_grad() || this->requires_grad())
    {

        std::shared_ptr<TensorHandle> handle1 = this->handle();
        std::shared_ptr<TensorHandle> handle2 = tensor.handle();

        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {handle1, handle2};
        GradFn grad_fn =
        [handle1, handle2](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            Tensor tmp1 = Tensor(handle1);
            Tensor tmp2 = Tensor(handle2);
            Tensor grad1 = tmp2 * upstream;
            Tensor grad2 = tmp1 * upstream;
            const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad1.handle(), grad2.handle()};
            return grad_handles;
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }

    return res;
}

Tensor Tensor::operator/(const Tensor& tensor) const
{
    Tensor res = this->copy();
    res /= tensor;

    if (tensor.requires_grad() || this->requires_grad())
    {

        std::shared_ptr<TensorHandle> handle1 = this->handle();
        std::shared_ptr<TensorHandle> handle2 = tensor.handle();

        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {handle1, handle2};
        GradFn grad_fn =
        [handle1, handle2](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            Tensor tmp1 = Tensor(handle1);
            Tensor tmp2 = Tensor(handle2);
            Tensor grad1 = 1.0f / tmp2 * upstream;
            Tensor grad2 = -1.0f * tmp1 / (tmp2*tmp2) * upstream;
            const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad1.handle(), grad2.handle()};
            return grad_handles;
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }

    return res;
}