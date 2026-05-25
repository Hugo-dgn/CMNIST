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
// 1. 1D inplace ops
// ─────────────────────────────────────────────

TEST(TensorInplace1D, Add)
{
    Tensor t({1.f, 2.f, 3.f});
    t += 1.f;
    expect_near(elements(t), {2.f, 3.f, 4.f});
}

TEST(TensorInplace1D, Sub)
{
    Tensor t({5.f, 6.f, 7.f});
    t -= 2.f;
    expect_near(elements(t), {3.f, 4.f, 5.f});
}

TEST(TensorInplace1D, Mul)
{
    Tensor t({1.f, 2.f, 3.f});
    t *= 3.f;
    expect_near(elements(t), {3.f, 6.f, 9.f});
}

TEST(TensorInplace1D, Div)
{
    Tensor t({2.f, 4.f, 8.f});
    t /= 2.f;
    expect_near(elements(t), {1.f, 2.f, 4.f});
}

// ─────────────────────────────────────────────
// 2. 2D inplace ops
// ─────────────────────────────────────────────

TEST(TensorInplace2D, Add)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    t += 1.f;
    expect_near(elements(t), {2.f, 3.f, 4.f, 5.f});
}

TEST(TensorInplace2D, Sub)
{
    Tensor t({{10.f, 20.f}, {30.f, 40.f}});
    t -= 10.f;
    expect_near(elements(t), {0.f, 10.f, 20.f, 30.f});
}

TEST(TensorInplace2D, Mul)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    t *= 2.f;
    expect_near(elements(t), {2.f, 4.f, 6.f, 8.f});
}

TEST(TensorInplace2D, Div)
{
    Tensor t({{2.f, 4.f}, {6.f, 8.f}});
    t /= 2.f;
    expect_near(elements(t), {1.f, 2.f, 3.f, 4.f});
}

// ─────────────────────────────────────────────
// 3. 3D inplace ops
// ─────────────────────────────────────────────

TEST(TensorInplace3D, Add)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    t += 10.f;
    expect_near(elements(t),
        {11.f,12.f,13.f,14.f,15.f,16.f,17.f,18.f});
}

TEST(TensorInplace3D, Sub)
{
    Tensor t({{{2.f, 3.f}, {4.f, 5.f}},
              {{6.f, 7.f}, {8.f, 9.f}}});
    t -= 2.f;
    expect_near(elements(t),
        {0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f});
}

TEST(TensorInplace3D, Mul)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    t *= 2.f;
    expect_near(elements(t),
        {2.f,4.f,6.f,8.f,10.f,12.f,14.f,16.f});
}

TEST(TensorInplace3D, Div)
{
    Tensor t({{{2.f, 4.f}, {6.f, 8.f}},
              {{10.f, 12.f}, {14.f, 16.f}}});
    t /= 2.f;
    expect_near(elements(t),
        {1.f,2.f,3.f,4.f,5.f,6.f,7.f,8.f});
}

// ─────────────────────────────────────────────
// 4. Slice inplace (minimal sanity check)
// ─────────────────────────────────────────────

TEST(TensorInplaceSlice, RowAdd)
{
    Tensor t({{1.f, 2.f, 3.f},
              {4.f, 5.f, 6.f}});

    t[0] += 10.f;
    expect_near(elements(t), {11.f,12.f,13.f,4.f,5.f,6.f});
}

TEST(TensorInplaceSlice, RowMul)
{
    Tensor t({{1.f, 2.f, 3.f},
              {4.f, 5.f, 6.f}});

    t[1] *= 2.f;
    expect_near(elements(t), {1.f,2.f,3.f,8.f,10.f,12.f});
}