#pragma once
#include <vector>

#include "tensor.hpp"

class SGD
{
    private:
        std::vector<std::shared_ptr<TensorHandle>> _parameters;
        float _lr;
    public:
        SGD(std::vector<std::shared_ptr<TensorHandle>> paramters, float lr);
        void zero_grad();
        void step();
};