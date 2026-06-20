#include <vector>
#include <iostream>

#include "tensor.hpp"

const std::vector<std::size_t>& Tensor::shape() const
{
    return _handle->shape;
}

const std::vector<std::size_t>& Tensor::stride() const
{
    return _handle->stride;
}

const size_t& Tensor::offset() const
{
    return _handle->offset;
}

const std::vector<float>& Tensor::data() const
{
    return _handle->storage->data;
}

const std::shared_ptr<TensorHandle> Tensor::handle() const
{
    return _handle;
}

std::vector<float>& Tensor::rdata()
{
    return _handle->storage->data;
}

const float& Tensor::item() const
{
    if (this->numel() == 1)
    {
        return _handle->storage->data[_handle->offset];
    }
    else
    {
        throw std::runtime_error("Only single element tensor support 'item' method");
    }
}

const bool& Tensor::requires_grad() const
{
    return _handle->storage->requires_grad;
}

const Tensor Tensor::grad() const
{
    std::shared_ptr<TensorStorage> storage = this->handle()->storage->grad_storage;
    if (!storage)
    {   
        storage = std::make_shared<TensorStorage>();
        this->handle()->storage->grad_storage = storage;
        float value = 0.0f;
        std::vector<float> data(this->numel(), value);
        storage->data = data;
    }
    return Tensor(storage, this->shape(), this->offset());
}