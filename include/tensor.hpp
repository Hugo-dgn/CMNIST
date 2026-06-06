#pragma once

#include <vector>
#include <memory>

struct TensorStorage
{
    std::vector<float> data;
};

class Tensor
{
    private:
        std::shared_ptr<TensorStorage> _storage;
        std::vector<size_t> _shape;
        std::vector<size_t> _stride;
        size_t _offset;
    public:
        Tensor(std::vector<float> data);
        Tensor(std::vector<std::vector<float>> data);
        Tensor(std::vector<std::vector<std::vector<float>>> data);
        Tensor(
                std::shared_ptr<TensorStorage> storage, 
                std::vector<std::size_t> shape,
                size_t offset
            );
        Tensor(
                std::vector<float> data, 
                std::vector<std::size_t> shape,
                size_t offset
            );

        const std::vector<std::size_t>& shape() const;
        const std::vector<std::size_t>& stride() const;
        size_t offset() const;
        const std::vector<float>& data() const;
        std::vector<float>& rdata();
        Tensor copy() const;
        Tensor transpose() const;

        const float& item() const;

        std::size_t numel() const;
        float* point();
        const float* point() const;

        Tensor operator[](std::size_t i) const;

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
};

Tensor allocateTensor(std::vector<std::size_t> shape);

Tensor matmul(const Tensor tensor1, const Tensor tensor2);
Tensor matmul_transpose(const Tensor tensor1, const Tensor ttensor2);
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