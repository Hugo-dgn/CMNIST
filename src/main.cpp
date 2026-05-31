#include <iostream>
#include <vector>

#include "tensor.hpp"

int main()
{
    std::vector<std::vector<float>> data1 = {
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    };
    std::vector<std::vector<float>> data2 = {
        {7.f, 8.f},
        {9.f, 10.f},
        {11.f, 12.f}
    };
    Tensor tensor1 = Tensor(data1);
    Tensor tensor2 = Tensor(data2);
    Tensor tensor3 = matmul(tensor1, tensor2);
    std::cout << tensor1.transpose() << std::endl;
    return 0;
}