#include <vector>
#include <iostream>

#include "tensor.hpp"

void print(
        std::ostream& os,
        const std::size_t* shape, 
        const std::size_t* stride,
        const float* data,
        std::size_t rest
    )
{

    std::size_t n = shape[0];
    std::size_t s = stride[0];

    os << "[";

    if (rest > 1)
    {
        for (std::size_t i = 0; i < n; i++)
        {
            if (rest == 2)
                os << "\n ";
            print(os, shape + 1, stride + 1, data + i * s, rest-1);
        }
        if (rest == 2)
            os << "\n";
    }
    else
    {
        for (std::size_t i = 0; i < n; i++)
        {
            os << " " << data[i * s] << " ";
        }
    }

    os << "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{

    print(
        os, tensor.shape().data(), 
        tensor.stride().data(), 
        tensor.point(),
        tensor.shape().size()
    );

    return os;
}