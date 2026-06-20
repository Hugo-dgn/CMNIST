#include "optim.hpp"

void SGD::step()
{
    for (std::shared_ptr<TensorHandle>& parameter : _parameters)
    {
        Tensor tensor = Tensor(parameter);
        Tensor grad = tensor.grad();
        tensor -= _lr * grad;
    }
}