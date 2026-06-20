#include <iostream>

#include "tensor.hpp"

void Tensor::backward()
{

    if (this->numel() != 1 || this->shape().size() > 0)
    {
        throw std::runtime_error("the method backward() can only be used on scalar tensor.");
    }

    Tensor root_grad = Tensor(1.0f);

    std::shared_ptr<TensorHandle> upstream_grad_handle = root_grad.handle();

    this->backward(upstream_grad_handle);
}

void Tensor::backward(const std::shared_ptr<TensorHandle>& upstream_grad_handle)
{
    Tensor grad = this->grad();
    Tensor incoming_grad = Tensor(upstream_grad_handle);
    grad += incoming_grad;

    if (this->handle()->grad_fn)
    {
        const std::vector<std::shared_ptr<TensorHandle>> grad_handles = this->handle()->grad_fn(upstream_grad_handle);
        std::vector<std::shared_ptr<TensorHandle>> parents = this->handle()->parents;
        for (size_t i = 0; i < parents.size(); i++)
        {
            Tensor tmp = Tensor(parents[i]);
            tmp.backward(grad_handles[i]);
        }
    }
}