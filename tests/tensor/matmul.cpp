#include <gtest/gtest.h>
#include <vector>
#include "tensor.hpp"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

static constexpr float EPS = 1e-5f;

static void collect(const Tensor& t, std::vector<float>& out)
{
    if (t.shape().empty())
    {
        out.push_back(t.item());
        return;
    }

    for (size_t i = 0; i < t.shape()[0]; ++i)
        collect(t[i], out);
}

static std::vector<float> elements(const Tensor& t)
{
    std::vector<float> v;
    collect(t, v);
    return v;
}

static void expect_near(const std::vector<float>& got,
                        const std::vector<float>& expected)
{
    ASSERT_EQ(got.size(), expected.size());

    for (size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], expected[i], EPS);
}

// ─────────────────────────────────────────────
// MatMul tests (2D matrices only)
// ─────────────────────────────────────────────

TEST(MatMul, Square2x2)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor b({
        {5.f, 6.f},
        {7.f, 8.f}
    });

    Tensor c = matmul(a, b);

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    expect_near(elements(c), {
        19.f, 22.f,
        43.f, 50.f
    });
}

TEST(MatMulTranspose, Square2x2)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor b({
        {5.f, 6.f},
        {7.f, 8.f}
    });

    Tensor c = matmul_transpose(a, b.transpose());

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    expect_near(elements(c), {
        19.f, 22.f,
        43.f, 50.f
    });
}

TEST(MatMul, Rectangular2x3_3x2)
{
    Tensor a({
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    });

    Tensor b({
        {7.f, 8.f},
        {9.f, 10.f},
        {11.f, 12.f}
    });

    Tensor c = matmul(a, b);

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    expect_near(elements(c), {
        58.f, 64.f,
        139.f, 154.f
    });
}

TEST(MatMulTranspose, Rectangular2x3_3x2)
{
    Tensor a({
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    });

    Tensor b({
        {7.f, 8.f},
        {9.f, 10.f},
        {11.f, 12.f}
    });

    Tensor c = matmul_transpose(a, b.transpose());

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    expect_near(elements(c), {
        58.f, 64.f,
        139.f, 154.f
    });
}

TEST(MatMul, Rectangular3x2_2x4)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f},
        {5.f, 6.f}
    });

    Tensor b({
        {7.f, 8.f, 9.f, 10.f},
        {11.f, 12.f, 13.f, 14.f}
    });

    Tensor c = matmul(a, b);

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);

    expect_near(elements(c), {
        29.f, 32.f, 35.f, 38.f,
        65.f, 72.f, 79.f, 86.f,
        101.f, 112.f, 123.f, 134.f
    });
}

TEST(MatMulTranspose, Rectangular3x2_2x4)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f},
        {5.f, 6.f}
    });

    Tensor b({
        {7.f, 8.f, 9.f, 10.f},
        {11.f, 12.f, 13.f, 14.f}
    });

    Tensor c = matmul_transpose(a, b.transpose());

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 3);
    EXPECT_EQ(c.shape()[1], 4);

    expect_near(elements(c), {
        29.f, 32.f, 35.f, 38.f,
        65.f, 72.f, 79.f, 86.f,
        101.f, 112.f, 123.f, 134.f
    });
}

TEST(MatMul, IdentityRight)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor identity({
        {1.f, 0.f},
        {0.f, 1.f}
    });

    Tensor c = matmul(a, identity);

    expect_near(elements(c), {
        1.f, 2.f,
        3.f, 4.f
    });
}

TEST(MatMulTranspose, IdentityRight)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor identity({
        {1.f, 0.f},
        {0.f, 1.f}
    });

    Tensor c = matmul_transpose(a, identity.transpose());

    expect_near(elements(c), {
        1.f, 2.f,
        3.f, 4.f
    });
}

TEST(MatMul, IdentityLeft)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor identity({
        {1.f, 0.f},
        {0.f, 1.f}
    });

    Tensor c = matmul(identity, a);

    expect_near(elements(c), {
        1.f, 2.f,
        3.f, 4.f
    });
}

TEST(MatMulTranspose, IdentityLeft)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor identity({
        {1.f, 0.f},
        {0.f, 1.f}
    });

    Tensor c = matmul_transpose(identity, a.transpose());

    expect_near(elements(c), {
        1.f, 2.f,
        3.f, 4.f
    });
}

TEST(MatMul, ZeroMatrixRight)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor zero({
        {0.f, 0.f},
        {0.f, 0.f}
    });

    Tensor c = matmul(a, zero);

    expect_near(elements(c), {
        0.f, 0.f,
        0.f, 0.f
    });
}

TEST(MatMulTranspose, ZeroMatrixRight)
{
    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor zero({
        {0.f, 0.f},
        {0.f, 0.f}
    });

    Tensor c = matmul_transpose(a, zero.transpose());

    expect_near(elements(c), {
        0.f, 0.f,
        0.f, 0.f
    });
}

TEST(MatMul, ZeroMatrixLeft)
{
    Tensor zero({
        {0.f, 0.f},
        {0.f, 0.f}
    });

    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor c = matmul(zero, a);

    expect_near(elements(c), {
        0.f, 0.f,
        0.f, 0.f
    });
}

TEST(MatMulTranspose, ZeroMatrixLeft)
{
    Tensor zero({
        {0.f, 0.f},
        {0.f, 0.f}
    });

    Tensor a({
        {1.f, 2.f},
        {3.f, 4.f}
    });

    Tensor c = matmul_transpose(zero, a.transpose());

    expect_near(elements(c), {
        0.f, 0.f,
        0.f, 0.f
    });
}

TEST(MatMul, SliceMultiplication)
{
    Tensor batch({
        {
            {1.f, 2.f},
            {3.f, 4.f}
        },
        {
            {5.f, 6.f},
            {7.f, 8.f}
        }
    });

    Tensor identity({
        {1.f, 0.f},
        {0.f, 1.f}
    });

    Tensor c = matmul(batch[1], identity);

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    expect_near(elements(c), {
        5.f, 6.f,
        7.f, 8.f
    });
}

TEST(MatMulTranspose, SliceMultiplication)
{
    Tensor batch({
        {
            {1.f, 2.f},
            {3.f, 4.f}
        },
        {
            {5.f, 6.f},
            {7.f, 8.f}
        }
    });

    Tensor identity({
        {1.f, 0.f},
        {0.f, 1.f}
    });

    Tensor c = matmul_transpose(batch[1], identity.transpose());

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    expect_near(elements(c), {
        5.f, 6.f,
        7.f, 8.f
    });
}

TEST(MatMul, SingleElementMatrices)
{
    Tensor a(std::vector<std::vector<float>>{{3.f}});
    Tensor b(std::vector<std::vector<float>>{{4.f}});

    Tensor c = matmul(a, b);

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 1);
    EXPECT_EQ(c.shape()[1], 1);

    expect_near(elements(c), {12.f});
}

TEST(MatMulTranspose, SingleElementMatrices)
{
    Tensor a(std::vector<std::vector<float>>{{3.f}});
    Tensor b(std::vector<std::vector<float>>{{4.f}});

    Tensor c = matmul_transpose(a, b.transpose());

    ASSERT_EQ(c.shape().size(), 2);
    EXPECT_EQ(c.shape()[0], 1);
    EXPECT_EQ(c.shape()[1], 1);

    expect_near(elements(c), {12.f});
}