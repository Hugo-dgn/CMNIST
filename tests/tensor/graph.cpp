#include <gtest/gtest.h>
#include <vector>
#include <memory>
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

static void expect_shape(const Tensor& t, const std::vector<size_t>& expected)
{
    ASSERT_EQ(t.shape().size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(t.shape()[i], expected[i]);
}


// ─────────────────────────────────────────────
// Arithmetic backward tests
// ─────────────────────────────────────────────

TEST(Backward, Add)
{
    Tensor x(std::vector<float>{1.f, 2.f, 3.f}, true);
    Tensor y(std::vector<float>{4.f, 5.f, 6.f}, true);

    Tensor loss = (x + y).sum();
    loss.backward();

    expect_near(elements(x.grad()), {1.f, 1.f, 1.f});
    expect_near(elements(y.grad()), {1.f, 1.f, 1.f});
    EXPECT_NEAR(loss.item(), 21.f, EPS);
}

TEST(Backward, Sub)
{
    Tensor x(std::vector<float>{1.f, 2.f, 3.f}, true);
    Tensor y(std::vector<float>{4.f, 5.f, 6.f}, true);

    Tensor loss = (x - y).sum();
    loss.backward();

    expect_near(elements(x.grad()), {1.f, 1.f, 1.f});
    expect_near(elements(y.grad()), {-1.f, -1.f, -1.f});
    EXPECT_NEAR(loss.item(), -9.f, EPS);
}

TEST(Backward, Mul)
{
    Tensor x(std::vector<float>{1.f, 2.f, 3.f}, true);
    Tensor y(std::vector<float>{4.f, 5.f, 6.f}, true);

    Tensor loss = (x * y).sum();
    loss.backward();

    expect_near(elements(x.grad()), {4.f, 5.f, 6.f});
    expect_near(elements(y.grad()), {1.f, 2.f, 3.f});
    EXPECT_NEAR(loss.item(), 32.f, EPS);
}

TEST(Backward, Div)
{
    Tensor x(std::vector<float>{4.f, 8.f, 16.f}, true);
    Tensor y(std::vector<float>{2.f, 4.f, 8.f}, true);

    Tensor loss = (x / y).sum();
    loss.backward();

    expect_near(elements(x.grad()), {0.5f, 0.25f, 0.125f});
    expect_near(elements(y.grad()), {-1.f, -0.5f, -0.25f});
    EXPECT_NEAR(loss.item(), 6.f, EPS);
}

TEST(Backward, ScalarRightAdd)
{
    Tensor x(std::vector<float>{1.f, 2.f, 3.f}, true);

    Tensor loss = (x + 2.f).sum();
    loss.backward();

    expect_near(elements(x.grad()), {1.f, 1.f, 1.f});
    EXPECT_NEAR(loss.item(), 12.f, EPS);
}

TEST(Backward, ScalarLeftMul)
{
    Tensor x(std::vector<float>{1.f, 2.f, 3.f}, true);

    Tensor loss = (2.f * x).sum();
    loss.backward();

    expect_near(elements(x.grad()), {2.f, 2.f, 2.f});
    EXPECT_NEAR(loss.item(), 12.f, EPS);
}

// ─────────────────────────────────────────────
// MatMul backward test
// ─────────────────────────────────────────────

TEST(Backward, MatMul)
{
    Tensor A(std::vector<std::vector<float>>{
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    }, true);

    Tensor B(std::vector<std::vector<float>>{
        {7.f, 8.f},
        {9.f, 10.f},
        {11.f, 12.f}
    }, true);

    Tensor C = matmul(A, B);
    Tensor loss = C.sum();
    loss.backward();

    // dL/dA = ones(2x2) * B^T
    expect_shape(A.grad(), {2, 3});
    expect_near(elements(A.grad()), {
        15.f, 19.f, 23.f,
        15.f, 19.f, 23.f
    });

    // dL/dB = A^T * ones(2x2)
    expect_shape(B.grad(), {3, 2});
    expect_near(elements(B.grad()), {
        5.f, 5.f,
        7.f, 7.f,
        9.f, 9.f
    });

    EXPECT_NEAR(loss.item(), 415.f, EPS);
}