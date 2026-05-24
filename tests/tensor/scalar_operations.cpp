#include <gtest/gtest.h>
#include <cmath>
#include "tensor.hpp"

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

static constexpr float EPS = 1e-5f;

// Recursively collect every element reachable through operator[] down to item()
static void collect(const Tensor& t, std::vector<float>& out)
{
    if (t.shape().empty())
    {
        out.push_back(t.item());
        return;
    }
    for (std::size_t i = 0; i < t.shape()[0]; ++i)
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
    for (std::size_t i = 0; i < got.size(); ++i)
        EXPECT_NEAR(got[i], expected[i], EPS) << "  at index " << i;
}

// ─────────────────────────────────────────────
// 1. Construction & metadata
// ─────────────────────────────────────────────

TEST(TensorConstruct, Shape1D)
{
    Tensor t({1.f, 2.f, 3.f});
    ASSERT_EQ(t.shape().size(), 1u);
    EXPECT_EQ(t.shape()[0], 3u);
    EXPECT_EQ(t.numel(), 3u);
    EXPECT_EQ(t.offset(), 0u);
}

TEST(TensorConstruct, Shape2D)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    ASSERT_EQ(t.shape().size(), 2u);
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_EQ(t.numel(), 4u);
}

TEST(TensorConstruct, Shape3D)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    ASSERT_EQ(t.shape().size(), 3u);
    EXPECT_EQ(t.shape()[0], 2u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_EQ(t.shape()[2], 2u);
    EXPECT_EQ(t.numel(), 8u);
}

TEST(TensorConstruct, Stride1D)
{
    Tensor t({1.f, 2.f, 3.f});
    ASSERT_EQ(t.stride().size(), 1u);
    EXPECT_EQ(t.stride()[0], 1u);
}

TEST(TensorConstruct, Stride2D)
{
    Tensor t({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    // shape [2,3] → strides [3,1]
    ASSERT_EQ(t.stride().size(), 2u);
    EXPECT_EQ(t.stride()[0], 3u);
    EXPECT_EQ(t.stride()[1], 1u);
}

TEST(TensorConstruct, Stride3D)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    // shape [2,2,2] → strides [4,2,1]
    ASSERT_EQ(t.stride().size(), 3u);
    EXPECT_EQ(t.stride()[0], 4u);
    EXPECT_EQ(t.stride()[1], 2u);
    EXPECT_EQ(t.stride()[2], 1u);
}

// ─────────────────────────────────────────────
// 2. operator[] and item()
// ─────────────────────────────────────────────

TEST(TensorIndex, Index1D)
{
    Tensor t({10.f, 20.f, 30.f});
    EXPECT_FLOAT_EQ(t[0].item(), 10.f);
    EXPECT_FLOAT_EQ(t[1].item(), 20.f);
    EXPECT_FLOAT_EQ(t[2].item(), 30.f);
}

TEST(TensorIndex, Index2D)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    EXPECT_FLOAT_EQ(t[0][0].item(), 1.f);
    EXPECT_FLOAT_EQ(t[0][1].item(), 2.f);
    EXPECT_FLOAT_EQ(t[1][0].item(), 3.f);
    EXPECT_FLOAT_EQ(t[1][1].item(), 4.f);
}

TEST(TensorIndex, Index3D)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    EXPECT_FLOAT_EQ(t[0][0][0].item(), 1.f);
    EXPECT_FLOAT_EQ(t[0][0][1].item(), 2.f);
    EXPECT_FLOAT_EQ(t[0][1][0].item(), 3.f);
    EXPECT_FLOAT_EQ(t[0][1][1].item(), 4.f);
    EXPECT_FLOAT_EQ(t[1][0][0].item(), 5.f);
    EXPECT_FLOAT_EQ(t[1][0][1].item(), 6.f);
    EXPECT_FLOAT_EQ(t[1][1][0].item(), 7.f);
    EXPECT_FLOAT_EQ(t[1][1][1].item(), 8.f);
}

// Sliced tensor metadata
TEST(TensorIndex, SliceShape1D)
{
    Tensor t({1.f, 2.f, 3.f});
    Tensor s = t[1];               // scalar view
    EXPECT_EQ(s.shape().size(), 0u);
    EXPECT_FLOAT_EQ(s.item(), 2.f);
}

TEST(TensorIndex, SliceShape2D)
{
    Tensor t({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    Tensor row = t[1];
    ASSERT_EQ(row.shape().size(), 1u);
    EXPECT_EQ(row.shape()[0], 3u);
    EXPECT_FLOAT_EQ(row[0].item(), 4.f);
    EXPECT_FLOAT_EQ(row[1].item(), 5.f);
    EXPECT_FLOAT_EQ(row[2].item(), 6.f);
}

TEST(TensorIndex, SliceShape3D)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    Tensor mat = t[1];
    ASSERT_EQ(mat.shape().size(), 2u);
    EXPECT_EQ(mat.shape()[0], 2u);
    EXPECT_EQ(mat.shape()[1], 2u);
    EXPECT_FLOAT_EQ(mat[0][0].item(), 5.f);
    EXPECT_FLOAT_EQ(mat[0][1].item(), 6.f);
    EXPECT_FLOAT_EQ(mat[1][0].item(), 7.f);
    EXPECT_FLOAT_EQ(mat[1][1].item(), 8.f);
}

TEST(TensorIndex, SliceOffset2D)
{
    Tensor t({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    Tensor row0 = t[0];
    Tensor row1 = t[1];
    // row0 starts at element 0, row1 at element 3
    EXPECT_EQ(row0.offset(), 0u);
    EXPECT_EQ(row1.offset(), 3u);
}

TEST(TensorIndex, SliceOffset3D)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    // stride[0] == 4, so t[1] has offset 4
    EXPECT_EQ(t[0].offset(), 0u);
    EXPECT_EQ(t[1].offset(), 4u);
    // t[1][0] has offset 4 + 0*2 = 4, t[1][1] = 4 + 1*2 = 6
    EXPECT_EQ(t[1][0].offset(), 4u);
    EXPECT_EQ(t[1][1].offset(), 6u);
}

// ─────────────────────────────────────────────
// 3. Scalar arithmetic — 1D tensor
// ─────────────────────────────────────────────

TEST(TensorArith1D, AddRight)
{
    Tensor t({1.f, 2.f, 3.f});
    auto r = t + 10.f;
    expect_near(elements(r), {11.f, 12.f, 13.f});
}

TEST(TensorArith1D, AddLeft)
{
    Tensor t({1.f, 2.f, 3.f});
    auto r = 10.f + t;
    expect_near(elements(r), {11.f, 12.f, 13.f});
}

TEST(TensorArith1D, SubRight)
{
    // tensor - scalar: result[i] = tensor[i] - scalar
    Tensor t({5.f, 10.f, 15.f});
    auto r = t - 3.f;
    expect_near(elements(r), {2.f, 7.f, 12.f});
}

TEST(TensorArith1D, SubLeft)
{
    // scalar - tensor: result[i] = scalar - tensor[i]
    Tensor t({1.f, 2.f, 3.f});
    auto r = 10.f - t;
    expect_near(elements(r), {9.f, 8.f, 7.f});
}

TEST(TensorArith1D, MulRight)
{
    Tensor t({1.f, 2.f, 3.f});
    auto r = t * 3.f;
    expect_near(elements(r), {3.f, 6.f, 9.f});
}

TEST(TensorArith1D, MulLeft)
{
    Tensor t({1.f, 2.f, 3.f});
    auto r = 3.f * t;
    expect_near(elements(r), {3.f, 6.f, 9.f});
}

TEST(TensorArith1D, DivRight)
{
    // tensor / scalar
    Tensor t({2.f, 4.f, 8.f});
    auto r = t / 2.f;
    expect_near(elements(r), {1.f, 2.f, 4.f});
}

TEST(TensorArith1D, DivLeft)
{
    // scalar / tensor
    Tensor t({1.f, 2.f, 4.f});
    auto r = 8.f / t;
    expect_near(elements(r), {8.f, 4.f, 2.f});
}

// ─────────────────────────────────────────────
// 4. Scalar arithmetic — 2D tensor
// ─────────────────────────────────────────────

TEST(TensorArith2D, AddRight)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    auto r = t + 1.f;
    expect_near(elements(r), {2.f, 3.f, 4.f, 5.f});
}

TEST(TensorArith2D, AddLeft)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    auto r = 1.f + t;
    expect_near(elements(r), {2.f, 3.f, 4.f, 5.f});
}

TEST(TensorArith2D, SubRight)
{
    Tensor t({{10.f, 20.f}, {30.f, 40.f}});
    auto r = t - 5.f;
    expect_near(elements(r), {5.f, 15.f, 25.f, 35.f});
}

TEST(TensorArith2D, SubLeft)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    auto r = 10.f - t;
    expect_near(elements(r), {9.f, 8.f, 7.f, 6.f});
}

TEST(TensorArith2D, MulRight)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    auto r = t * 2.f;
    expect_near(elements(r), {2.f, 4.f, 6.f, 8.f});
}

TEST(TensorArith2D, MulLeft)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    auto r = 2.f * t;
    expect_near(elements(r), {2.f, 4.f, 6.f, 8.f});
}

TEST(TensorArith2D, DivRight)
{
    Tensor t({{2.f, 4.f}, {6.f, 8.f}});
    auto r = t / 2.f;
    expect_near(elements(r), {1.f, 2.f, 3.f, 4.f});
}

TEST(TensorArith2D, DivLeft)
{
    Tensor t({{1.f, 2.f}, {4.f, 8.f}});
    auto r = 8.f / t;
    expect_near(elements(r), {8.f, 4.f, 2.f, 1.f});
}

// ─────────────────────────────────────────────
// 5. Scalar arithmetic — 3D tensor
// ─────────────────────────────────────────────

TEST(TensorArith3D, AddRight)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    auto r = t + 10.f;
    expect_near(elements(r), {11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f});
}

TEST(TensorArith3D, AddLeft)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    auto r = 10.f + t;
    expect_near(elements(r), {11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f});
}

TEST(TensorArith3D, SubRight)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    auto r = t - 1.f;
    expect_near(elements(r), {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
}

TEST(TensorArith3D, SubLeft)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    auto r = 9.f - t;
    expect_near(elements(r), {8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f});
}

TEST(TensorArith3D, MulRight)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    auto r = t * 2.f;
    expect_near(elements(r), {2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f, 16.f});
}

TEST(TensorArith3D, MulLeft)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    auto r = 2.f * t;
    expect_near(elements(r), {2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 14.f, 16.f});
}

TEST(TensorArith3D, DivRight)
{
    Tensor t({{{2.f, 4.f}, {6.f, 8.f}},
              {{10.f, 12.f}, {14.f, 16.f}}});
    auto r = t / 2.f;
    expect_near(elements(r), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
}

TEST(TensorArith3D, DivLeft)
{
    Tensor t({{{1.f, 2.f}, {4.f, 8.f}},
              {{1.f, 2.f}, {4.f, 8.f}}});
    auto r = 8.f / t;
    expect_near(elements(r), {8.f, 4.f, 2.f, 1.f, 8.f, 4.f, 2.f, 1.f});
}

// ─────────────────────────────────────────────
// 6. Operations on 2D slices (operator[] + arith)
// ─────────────────────────────────────────────

// Slice a row from a 2D tensor, then do every arithmetic op on the slice
class SliceArith2D : public ::testing::Test
{
protected:
    // t = [[1, 2, 3],
    //      [4, 5, 6]]
    Tensor t{std::vector<std::vector<float>>{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}};
};

TEST_F(SliceArith2D, Row0AddRight)
{
    auto r = t[0] + 10.f;
    expect_near(elements(r), {11.f, 12.f, 13.f});
}

TEST_F(SliceArith2D, Row0AddLeft)
{
    auto r = 10.f + t[0];
    expect_near(elements(r), {11.f, 12.f, 13.f});
}

TEST_F(SliceArith2D, Row1SubRight)
{
    auto r = t[1] - 1.f;
    expect_near(elements(r), {3.f, 4.f, 5.f});
}

TEST_F(SliceArith2D, Row1SubLeft)
{
    auto r = 10.f - t[1];
    expect_near(elements(r), {6.f, 5.f, 4.f});
}

TEST_F(SliceArith2D, Row0MulRight)
{
    auto r = t[0] * 3.f;
    expect_near(elements(r), {3.f, 6.f, 9.f});
}

TEST_F(SliceArith2D, Row0MulLeft)
{
    auto r = 3.f * t[0];
    expect_near(elements(r), {3.f, 6.f, 9.f});
}

TEST_F(SliceArith2D, Row1DivRight)
{
    auto r = t[1] / 2.f;
    expect_near(elements(r), {2.f, 2.5f, 3.f});
}

TEST_F(SliceArith2D, Row1DivLeft)
{
    auto r = 12.f / t[1];
    expect_near(elements(r), {3.f, 12.f/5.f, 2.f});
}

// ─────────────────────────────────────────────
// 7. Operations on 3D slices
// ─────────────────────────────────────────────

// t = [[[1,2],[3,4]],
//      [[5,6],[7,8]]]
class SliceArith3D : public ::testing::Test
{
protected:
    Tensor t{std::vector<std::vector<std::vector<float>>>{
        {{1.f, 2.f}, {3.f, 4.f}},
        {{5.f, 6.f}, {7.f, 8.f}}}};
};

// t[0] is the 2D matrix [[1,2],[3,4]]
TEST_F(SliceArith3D, Mat0AddRight)
{
    auto r = t[0] + 1.f;
    expect_near(elements(r), {2.f, 3.f, 4.f, 5.f});
}

TEST_F(SliceArith3D, Mat0AddLeft)
{
    auto r = 1.f + t[0];
    expect_near(elements(r), {2.f, 3.f, 4.f, 5.f});
}

TEST_F(SliceArith3D, Mat1SubRight)
{
    auto r = t[1] - 4.f;
    expect_near(elements(r), {1.f, 2.f, 3.f, 4.f});
}

TEST_F(SliceArith3D, Mat1SubLeft)
{
    auto r = 9.f - t[1];
    expect_near(elements(r), {4.f, 3.f, 2.f, 1.f});
}

TEST_F(SliceArith3D, Mat0MulRight)
{
    auto r = t[0] * 2.f;
    expect_near(elements(r), {2.f, 4.f, 6.f, 8.f});
}

TEST_F(SliceArith3D, Mat0MulLeft)
{
    auto r = 2.f * t[0];
    expect_near(elements(r), {2.f, 4.f, 6.f, 8.f});
}

TEST_F(SliceArith3D, Mat1DivRight)
{
    auto r = t[1] / 2.f;
    expect_near(elements(r), {2.5f, 3.f, 3.5f, 4.f});
}

TEST_F(SliceArith3D, Mat1DivLeft)
{
    auto r = 10.f / t[1];
    expect_near(elements(r), {2.f, 10.f/6.f, 10.f/7.f, 10.f/8.f});
}

// Nested slice: t[0][1] is the 1D row [3, 4]
TEST_F(SliceArith3D, NestedSliceAdd)
{
    auto r = t[0][1] + 10.f;
    expect_near(elements(r), {13.f, 14.f});
}

TEST_F(SliceArith3D, NestedSliceSub)
{
    auto r = 5.f - t[0][1];
    expect_near(elements(r), {2.f, 1.f});
}

TEST_F(SliceArith3D, NestedSliceMul)
{
    auto r = t[1][0] * 3.f;    // [5,6] * 3
    expect_near(elements(r), {15.f, 18.f});
}

TEST_F(SliceArith3D, NestedSliceDiv)
{
    auto r = 12.f / t[1][1];   // 12 / [7,8]
    expect_near(elements(r), {12.f/7.f, 12.f/8.f});
}

// ─────────────────────────────────────────────
// 8. numel()
// ─────────────────────────────────────────────

TEST(TensorNumel, Numel1D) { EXPECT_EQ(Tensor({1.f, 2.f, 3.f}).numel(), 3u); }
TEST(TensorNumel, Numel2D) { EXPECT_EQ(Tensor({{1.f,2.f},{3.f,4.f}}).numel(), 4u); }
TEST(TensorNumel, Numel3D)
{
    Tensor t({{{1.f,2.f},{3.f,4.f}},{{5.f,6.f},{7.f,8.f}}});
    EXPECT_EQ(t.numel(), 8u);
}

// ─────────────────────────────────────────────
// 9. Edge cases
// ─────────────────────────────────────────────

TEST(TensorEdge, AddZero)
{
    Tensor t({1.f, 2.f, 3.f});
    expect_near(elements(t + 0.f), {1.f, 2.f, 3.f});
    expect_near(elements(0.f + t), {1.f, 2.f, 3.f});
}

TEST(TensorEdge, MulOne)
{
    Tensor t({1.f, 2.f, 3.f});
    expect_near(elements(t * 1.f), {1.f, 2.f, 3.f});
    expect_near(elements(1.f * t), {1.f, 2.f, 3.f});
}

TEST(TensorEdge, MulZero)
{
    Tensor t({1.f, 2.f, 3.f});
    expect_near(elements(t * 0.f), {0.f, 0.f, 0.f});
}

TEST(TensorEdge, NegativeScalar)
{
    Tensor t({1.f, 2.f, 3.f});
    expect_near(elements(t + (-1.f)), {0.f, 1.f, 2.f});
    expect_near(elements(t * (-1.f)), {-1.f, -2.f, -3.f});
}

TEST(TensorEdge, SubSelf)
{
    // t - scalar where scalar equals first element
    Tensor t({5.f, 5.f, 5.f});
    expect_near(elements(t - 5.f), {0.f, 0.f, 0.f});
}

TEST(TensorEdge, DivByOne)
{
    Tensor t({3.f, 6.f, 9.f});
    expect_near(elements(t / 1.f), {3.f, 6.f, 9.f});
}

// ─────────────────────────────────────────────
// 10. Result shape is preserved after arithmetic
// ─────────────────────────────────────────────

TEST(TensorResultShape, Add2DPreservesShape)
{
    Tensor t({{1.f, 2.f}, {3.f, 4.f}});
    auto r = t + 1.f;
    ASSERT_EQ(r.shape().size(), 2u);
    EXPECT_EQ(r.shape()[0], 2u);
    EXPECT_EQ(r.shape()[1], 2u);
}

TEST(TensorResultShape, Add3DPreservesShape)
{
    Tensor t({{{1.f, 2.f}, {3.f, 4.f}},
              {{5.f, 6.f}, {7.f, 8.f}}});
    auto r = t * 2.f;
    ASSERT_EQ(r.shape().size(), 3u);
    EXPECT_EQ(r.shape()[0], 2u);
    EXPECT_EQ(r.shape()[1], 2u);
    EXPECT_EQ(r.shape()[2], 2u);
}