#include <gtest/gtest.h>

#include "tensor.hpp"

TEST(TensorTest, IndexingBasic) {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor tensor1 = Tensor(data1);

    for (std::size_t i = 0; i < data1.size(); i++)
    {
        EXPECT_FLOAT_EQ(tensor1[i].item(), data1[i]);
    }

    EXPECT_THROW(tensor1[4], std::out_of_range);

    std::vector<std::vector<float>> data2 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    Tensor tensor2 = Tensor(data2);

    for (std::size_t i = 0; i < data2.size(); i++)
    {
        for (std::size_t j = 0; j < data2[0].size(); j++)
        {
            EXPECT_FLOAT_EQ(tensor2[i][j].item(), data2[i][j]);
        }
    }

    for (std::size_t i = 0; i < data2.size(); i++)
    {
        EXPECT_THROW(tensor2[i][3], std::out_of_range);
    }

    std::vector<std::vector<std::vector<float>>> data3 = {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}};
    Tensor tensor3 = Tensor(data3);

    for (std::size_t i = 0; i < data3.size(); i++)
    {
        for (std::size_t j = 0; j < data3[i].size(); j++)
        {
            for (std::size_t k = 0; k < data3[j].size(); k++)
            {
                EXPECT_FLOAT_EQ(tensor3[i][j][k].item(), data3[i][j][k]);
            }
        }
    }

    for (std::size_t i = 0; i < data3.size(); i++)
    {
        for (std::size_t j = 0; j < data3[i].size(); j++)
        {
            EXPECT_THROW(tensor3[i][j][2], std::out_of_range);
        }
    }
}