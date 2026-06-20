#include <iostream>
#include <vector>

#include "tensor.hpp"

int main()
{
    std::vector<float> data1 = {1.f, 2.f, 3.f};
    std::vector<size_t> shape = {3, 1};
    std::size_t offset = 0;
    Tensor x = Tensor(data1, shape, offset);

    std::vector<float> data2 = {1.f, 3.f, -1.f};
    Tensor target = Tensor(data2, shape, offset);

    std::vector<std::vector<float>> data3 = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
    bool requires_grad = true;
    Tensor parameters = Tensor(data3, requires_grad);

    Tensor y = matmul(parameters, x);
    Tensor tmp1 = y - target;
    Tensor tmp2 = tmp1 * tmp1;
    Tensor loss = tmp2.sum();

    loss.backward();
    std::cout << parameters.grad() << std::endl;

    Tensor t({1.f, 2.f, 4.f});
    auto r = 8.f / t;
    loss.backward();
    std::cout << r << std::endl;

    return 0;
}