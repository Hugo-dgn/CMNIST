#include "tensor.hpp"

Tensor Tensor::sum()
{
    const float* data = this->point();

    float x = 0;
    #pragma omp simd reduction(+:x)
    for (std::size_t i = 0; i < this->numel(); i++)
    {
        x += data[i];
    }

    Tensor res = Tensor(x, this->requires_grad());

    if (this->requires_grad())
    {

        const std::vector<std::shared_ptr<TensorHandle>> parent_handles = {this->handle()};
        std::vector<std::size_t> input_shape = this->shape();

        GradFn grad_fn =
        [input_shape](const std::shared_ptr<TensorHandle>& upstream)
        -> std::vector<std::shared_ptr<TensorHandle>>
        {
            Tensor tmp = Tensor(upstream);
            Tensor grad = allocateTensor(input_shape, tmp.item());
            const std::vector<std::shared_ptr<TensorHandle>> grad_handles = {grad.handle()};
            return grad_handles;
        };

        res.set_grad_fn(parent_handles, grad_fn);
    }

    return res;
}