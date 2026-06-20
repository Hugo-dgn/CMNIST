#include <iostream>
#include <vector>

#include "tensor.hpp"
#include "optim.hpp"

Tensor mse(Tensor x, Tensor target)
{
    Tensor tmp1 = x - target;
    Tensor tmp2 = tmp1 * tmp1;
    Tensor loss = tmp2.sum();
    return loss;
}

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
    Tensor tensor = Tensor(data3, requires_grad);

    float lr = 0.01f;
    std::vector<std::shared_ptr<TensorHandle>> parameters = {tensor.handle()};
    SGD optimizer = SGD(parameters, lr);

    int epochs = 100;

    for (int i = 0; i < epochs; i++)
    {
        optimizer.zero_grad();
        Tensor y = matmul(tensor, x);
        Tensor loss = mse(y, target);
        std::cout << loss << std::endl;
        loss.backward();
        optimizer.step();
    }

    Tensor y = matmul(tensor, x);

    std::cout << tensor << std::endl;
    std::cout << y << std::endl;

    return 0;
}