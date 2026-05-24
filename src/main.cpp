#include <iostream>
#include <vector>

#include "tensor.hpp"

int main()
{
    std::vector<std::vector<std::vector<float>>> data = {{{1.0f, 2.0f}, {4.0f, 5.0f}}, {{6.0f, 7.0f}, {8.0f, 9.0f}}};
    Tensor tensor1 = Tensor(data);
    Tensor tensor2 = tensor1[0];
    std::cout << tensor1 << std::endl;
    return 0;
}