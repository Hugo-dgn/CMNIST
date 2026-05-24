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
                std::vector<size_t> stride, 
                size_t offset
            );

        const std::vector<std::size_t>& shape() const;
        const std::vector<std::size_t>& stride() const;
        size_t offset() const;
        const std::vector<float>& data() const;

        const float& item() const;

        std::size_t numel() const;

        Tensor operator[](std::size_t i) const;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
