#include <gtest/gtest.h>
#include <cmath>
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
// 1. 1D inplace tensor ops
// ─────────────────────────────────────────────

TEST(TensorTensorInplace1D, Add)
{
    Tensor a({1.f, 2.f, 3.f});
    Tensor b({4.f, 5.f, 6.f});

    a += b;

    expect_near(elements(a), {5.f, 7.f, 9.f});
}

TEST(TensorTensorInplace1D, Sub)
{
    Tensor a({5.f, 7.f, 9.f});
    Tensor b({1.f, 2.f, 3.f});

    a -= b;

    expect_near(elements(a), {4.f, 5.f, 6.f});
}

TEST(TensorTensorInplace1D, Mul)
{
    Tensor a({1.f, 2.f, 3.f});
    Tensor b({4.f, 5.f, 6.f});

    a *= b;

    expect_near(elements(a), {4.f, 10.f, 18.f});
}

TEST(TensorTensorInplace1D, Div)
{
    Tensor a({4.f, 10.f, 18.f});
    Tensor b({2.f, 5.f, 3.f});

    a /= b;

    expect_near(elements(a), {2.f, 2.f, 6.f});
}

// ─────────────────────────────────────────────
// 2. 2D inplace tensor ops
// ─────────────────────────────────────────────

TEST(TensorTensorInplace2D, Add)
{
    Tensor a({{1.f, 2.f}, {3.f, 4.f}});
    Tensor b({{10.f, 20.f}, {30.f, 40.f}});

    a += b;

    expect_near(elements(a), {11.f, 22.f, 33.f, 44.f});
}

TEST(TensorTensorInplace2D, Sub)
{
    Tensor a({{11.f, 22.f}, {33.f, 44.f}});
    Tensor b({{1.f, 2.f}, {3.f, 4.f}});

    a -= b;

    expect_near(elements(a), {10.f, 20.f, 30.f, 40.f});
}

TEST(TensorTensorInplace2D, Mul)
{
    Tensor a({{1.f, 2.f}, {3.f, 4.f}});
    Tensor b({{2.f, 3.f}, {4.f, 5.f}});

    a *= b;

    expect_near(elements(a), {2.f, 6.f, 12.f, 20.f});
}

TEST(TensorTensorInplace2D, Div)
{
    Tensor a({{2.f, 6.f}, {12.f, 20.f}});
    Tensor b({{2.f, 3.f}, {4.f, 5.f}});

    a /= b;

    expect_near(elements(a), {1.f, 2.f, 3.f, 4.f});
}

// ─────────────────────────────────────────────
// 3. 3D inplace tensor ops
// ─────────────────────────────────────────────

TEST(TensorTensorInplace3D, Add)
{
    Tensor a({
        {{1.f, 2.f}, {3.f, 4.f}},
        {{5.f, 6.f}, {7.f, 8.f}}
    });

    Tensor b({
        {{10.f, 20.f}, {30.f, 40.f}},
        {{50.f, 60.f}, {70.f, 80.f}}
    });

    a += b;

    expect_near(elements(a),
        {11.f,22.f,33.f,44.f,55.f,66.f,77.f,88.f});
}

TEST(TensorTensorInplace3D, Sub)
{
    Tensor a({
        {{11.f,22.f},{33.f,44.f}},
        {{55.f,66.f},{77.f,88.f}}
    });

    Tensor b({
        {{10.f,20.f},{30.f,40.f}},
        {{50.f,60.f},{70.f,80.f}}
    });

    a -= b;

    expect_near(elements(a),
        {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f});
}

TEST(TensorTensorInplace3D, Mul)
{
    Tensor a({
        {{1.f,2.f},{3.f,4.f}},
        {{5.f,6.f},{7.f,8.f}}
    });

    Tensor b({
        {{2.f,2.f},{2.f,2.f}},
        {{2.f,2.f},{2.f,2.f}}
    });

    a *= b;

    expect_near(elements(a),
        {2.f,4.f,6.f,8.f,10.f,12.f,14.f,16.f});
}

TEST(TensorTensorInplace3D, Div)
{
    Tensor a({
        {{2.f,4.f},{6.f,8.f}},
        {{10.f,12.f},{14.f,16.f}}
    });

    Tensor b({
        {{2.f,2.f},{2.f,2.f}},
        {{2.f,2.f},{2.f,2.f}}
    });

    a /= b;

    expect_near(elements(a),
        {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f});
}

// ─────────────────────────────────────────────
// 4. Non-inplace tensor ops
// ─────────────────────────────────────────────

TEST(TensorTensor1D, Add)
{
    Tensor a({1.f, 2.f, 3.f});
    Tensor b({4.f, 5.f, 6.f});

    Tensor c = a + b;

    expect_near(elements(c), {5.f, 7.f, 9.f});
}

TEST(TensorTensor1D, Sub)
{
    Tensor a({5.f, 7.f, 9.f});
    Tensor b({1.f, 2.f, 3.f});

    Tensor c = a - b;

    expect_near(elements(c), {4.f, 5.f, 6.f});
}

TEST(TensorTensor1D, Mul)
{
    Tensor a({1.f, 2.f, 3.f});
    Tensor b({4.f, 5.f, 6.f});

    Tensor c = a * b;

    expect_near(elements(c), {4.f, 10.f, 18.f});
}

TEST(TensorTensor1D, Div)
{
    Tensor a({4.f, 10.f, 18.f});
    Tensor b({2.f, 5.f, 3.f});

    Tensor c = a / b;

    expect_near(elements(c), {2.f, 2.f, 6.f});
}

// ─────────────────────────────────────────────
// 5. Slice tensor ops
// ─────────────────────────────────────────────

TEST(TensorTensorSlice, Add)
{
    Tensor a({
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    });

    Tensor b({10.f, 20.f, 30.f});

    a[0] += b;

    expect_near(elements(a),
        {11.f,22.f,33.f,4.f,5.f,6.f});
}

TEST(TensorTensorSlice, Mul)
{
    Tensor a({
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f}
    });

    Tensor b({2.f, 3.f, 4.f});

    a[1] *= b;

    expect_near(elements(a),
        {1.f,2.f,3.f,8.f,15.f,24.f});
}