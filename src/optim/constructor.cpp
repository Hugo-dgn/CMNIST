#include <vector>
#include <iostream>

#include "optim.hpp"

SGD::SGD(std::vector<std::shared_ptr<TensorHandle>> parameters, float lr)
{
    _parameters = parameters;
    _lr = lr;
}