#pragma once

#include <vector>
#include <memory>
#include <functional>

struct TensorStorage
{
    bool requires_grad;
    std::vector<float> data;
    std::shared_ptr<TensorStorage> grad_storage;
};


struct TensorHandle
{
    std::shared_ptr<TensorStorage> storage;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    size_t offset;

    std::function<const std::vector<std::shared_ptr<TensorHandle>>
    (const std::shared_ptr<TensorHandle>& upstream_grad)> grad_fn;

    std::vector<std::shared_ptr<TensorHandle>> parents;
};

using GradFn = std::function<
    std::vector<std::shared_ptr<TensorHandle>>(
        const std::shared_ptr<TensorHandle>&
    )
>;

class Tensor
{
    private:
        std::shared_ptr<TensorHandle> _handle;

    public:
        Tensor(std::shared_ptr<TensorHandle> handle);
        Tensor(float x, bool requires_grad = false);
        Tensor(std::vector<float> data, bool requires_grad = false);
        Tensor(std::vector<std::vector<float>> data, bool requires_grad = false);
        Tensor(std::vector<std::vector<std::vector<float>>> data, bool requires_grad = false);
        Tensor(
                std::shared_ptr<TensorStorage> storage, 
                std::vector<std::size_t> shape,
                size_t offset,
                bool requires_grad = false
            );
        Tensor(
                std::vector<float> data, 
                std::vector<std::size_t> shape,
                size_t offset,
                bool requires_grad = false
            );

        const std::vector<std::size_t>& shape() const;
        const std::vector<std::size_t>& stride() const;
        const size_t& offset() const;
        const std::vector<float>& data() const;
        const std::shared_ptr<TensorHandle> handle() const;
        const Tensor grad() const;

        std::vector<float>& rdata();
        Tensor copy() const;
        Tensor transpose() const;

        const float& item() const;

        std::size_t numel() const;
        float* point();
        const float* point() const;
        const bool& requires_grad() const;

        Tensor operator[](std::size_t i) const;

        Tensor sum();

        Tensor& operator+=(float x);
        Tensor& operator-=(float x);
        Tensor& operator*=(float x);
        Tensor& operator/=(float x);

        Tensor& operator+=(const Tensor& tensor);
        Tensor& operator-=(const Tensor& tensor);
        Tensor& operator*=(const Tensor& tensor);
        Tensor& operator/=(const Tensor& tensor);

        Tensor operator+(const Tensor& tensor) const;
        Tensor operator-(const Tensor& tensor) const;
        Tensor operator*(const Tensor& tensor) const;
        Tensor operator/(const Tensor& tensor) const;

        void fill(const float x);

        void set_grad_fn(const std::vector<std::shared_ptr<TensorHandle>> parents, GradFn& backward);

        void backward(const std::shared_ptr<TensorHandle>& upstream_grad);
        void backward();
};

Tensor allocateTensor(std::vector<std::size_t> shape, float value = 0.0f);

Tensor matmul(const Tensor& tensor1, const Tensor& tensor2);
Tensor matmul_transpose(const Tensor& tensor1, const Tensor& ttensor2);
void inplace_matmul(const Tensor& tensor1, const Tensor& tensor2, Tensor& tensor3);
void inplace_matmul_transpose(const Tensor& tensor1, const Tensor& ttensor2, Tensor& tensor3);

//right scalar operations

Tensor operator+(float x, const Tensor& tensor);
Tensor operator-(float x, const Tensor& tensor);
Tensor operator*(float x, const Tensor& tensor);
Tensor operator/(float x, const Tensor& tensor);

//left scalar operations

Tensor operator+(const Tensor& tensor, float x);
Tensor operator-(const Tensor& tensor, float x);
Tensor operator*(const Tensor& tensor, float x);
Tensor operator/(const Tensor& tensor, float x);

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);