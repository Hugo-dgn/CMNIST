#pragma once

#include <vector>

class TensorView
{
    private:
        float* _data;
        std::vector<std::size_t> _shape;
        std::vector<std::size_t> _stride;
        std::size_t _offset;
        std::size_t _dim;
    public:
        TensorView(float* data, std::vector<std::size_t> shape, std::vector<std::size_t> stride, std::size_t offset, std::size_t dim);

        const std::vector<std::size_t>& shape() const;
        const std::vector<std::size_t>& stride() const;
        const float* data() const;
        std::size_t offset() const;

        const float& item() const;

        std::size_t numel() const;

        TensorView operator[](std::size_t i);
};

class Tensor
{
    private:
        std::vector<std::size_t> _shape;
        std::vector<std::size_t> _stride;
        std::vector<float> _data;
    public:
        Tensor(std::vector<float> data);
        Tensor(std::vector<std::vector<float>> data);
        Tensor(std::vector<std::vector<std::vector<float>>> data);
        Tensor(std::vector<float> data, std::vector<std::size_t> shape);

        const std::vector<std::size_t>& shape() const;
        const std::vector<std::size_t>& stride() const;
        const std::vector<float>& data() const;

        const float& item() const;

        std::size_t numel() const;

        TensorView operator[](std::size_t i);
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
std::ostream& operator<<(std::ostream& os, const TensorView& tensor);