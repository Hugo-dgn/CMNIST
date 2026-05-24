#include <iostream>
#include <vector>

#include "tensor.h"

int main()
{
    std::vector<std::vector<float>> data = {{1.0f, 2.0f, 3.0f}, {3.0f, 4.0f, 5.0f}};
    Tensor tensor = Tensor(data);
    std::cout << tensor[0][0].item() << std::endl;
    return 0;
}