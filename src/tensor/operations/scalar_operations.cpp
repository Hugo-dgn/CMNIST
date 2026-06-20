#include <vector>
#include <iostream>
#include <functional>
#include <cstring>

#include "tensor.hpp"

template<bool Swap, typename Func>
void apply(
        const float* src,
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


// grad fn

void set_plus_grad_fn(const Tensor& tensor, Tensor& res)
{
    if (tensor.requires_grad())
    {
        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {tensor.handle()};

        GradFn grad_fn =
        [](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            return {upstream};
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }
}

template<bool swap>
void set_minus_grad_fn(const Tensor& tensor, Tensor& res)
{
    if (tensor.requires_grad())
    {
        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {tensor.handle()};

        float alpha = -1;
        if (swap)
        {
            alpha = 1;
        }

        GradFn grad_fn =
        [alpha](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            if constexpr (swap)
            {
                return {upstream};
            }
            else
            {
                Tensor tmp = Tensor(upstream);
                tmp *= 1;
                return {tmp.handle()};
            }
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }
}

void set_multiply_grad_fn(const Tensor& tensor, Tensor& res, const float& x)
{
    if (tensor.requires_grad())
    {

        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {tensor.handle()};
        GradFn grad_fn =
        [x](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            Tensor tmp = Tensor(upstream);
            Tensor grad = x * tmp;
            const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad.handle()};
            return grad_handles;
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }
}

template<bool swap>
void set_divide_grad_fn(const Tensor& tensor, Tensor& res, const float& x)
{
    if (tensor.requires_grad())
    {
        
        std::shared_ptr<TensorHandle> handle = res.handle();
        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {tensor.handle()};
        GradFn grad_fn =
        [x, handle](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            if constexpr (swap)
            {
                Tensor tmp = Tensor(upstream);
                Tensor grad = 1.0 / x * tmp;
                const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad.handle()};
                return grad_handles;
            }
            else
            {
                Tensor tmp1 = Tensor(handle);
                Tensor tmp2 = Tensor(upstream);
                Tensor grad = -1.0f * x / (tmp1 * tmp1) * tmp2;
                const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad.handle()};
                return grad_handles;
            }
            
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }
}

// right

Tensor operator+(float x, const Tensor& tensor)
{

    constexpr bool swap = false;
    Tensor res = create_apply<swap>(x, tensor, add);
    set_plus_grad_fn(tensor, res);
    return res;
}

Tensor operator-(float x, const Tensor& tensor)
{  
    constexpr bool swap = false;
    Tensor res = create_apply<swap>(x, tensor, substract);
    set_minus_grad_fn<swap>(tensor, res);

    return res;
}

Tensor operator*(float x, const Tensor& tensor)
{
    constexpr bool swap = false;
    Tensor res = create_apply<swap>(x, tensor, multiply);

    set_multiply_grad_fn(tensor, res, x);

    return res;
}

Tensor operator/(float x, const Tensor& tensor)
{
    constexpr bool swap = false;
    Tensor res = create_apply<swap>(x, tensor, divide);
    set_divide_grad_fn<swap>(tensor, res, x);
    return res;
}

// left

Tensor operator+(const Tensor& tensor, float x)
{
    constexpr bool swap = true;
    Tensor res = create_apply<swap>(x, tensor, add);
    set_plus_grad_fn(tensor, res);
    return res;
}

Tensor operator-(const Tensor& tensor, float x)
{
    constexpr bool swap = true;
    Tensor res = create_apply<swap>(x, tensor, substract);
    set_minus_grad_fn<swap>(tensor, res);
    return res;
}

Tensor operator*(const Tensor& tensor, float x)
{
    constexpr bool swap = true;
    Tensor res = create_apply<swap>(x, tensor, multiply);
    set_multiply_grad_fn(tensor, res, x);
    return res;
}

Tensor operator/(const Tensor& tensor, float x)
{
    constexpr bool swap = true;
    Tensor res = create_apply<swap>(1.0f / x, tensor, multiply);
    set_divide_grad_fn<swap>(tensor, res, x);
    return res;
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