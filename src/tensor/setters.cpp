#include "tensor.hpp"

void Tensor::fill(const float x)
{
    float* data = this->point();

    #pragma omp simd
    for (std::size_t i = 0; i < this->numel(); i++)
    {
        data[i] = x;
    }
}