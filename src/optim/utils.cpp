#include "optim.hpp"
#include "tensor.hpp"

void SGD::zero_grad()
{
    for (std::shared_ptr<TensorHandle>& parameter : _parameters)
    {
        Tensor tensor = Tensor(parameter);
        Tensor grad = tensor.grad();
        grad.fill(0.0f);
    }
}