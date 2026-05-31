#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

#include "tensor.hpp"

static void BM_Matmul(benchmark::State& state)
{

    const std::size_t N = state.range(0);

    std::vector<float> data(N*N, 1);
    std::vector<std::size_t> shape = {N, N};

    Tensor tensor1 = Tensor(data, shape, 0);
    Tensor tensor2 = Tensor(data, shape, 0);

    for (auto _ : state)
    {
        Tensor mul = matmul(tensor1, tensor2);

        benchmark::DoNotOptimize(tensor1);
        benchmark::DoNotOptimize(tensor2);
        benchmark::DoNotOptimize(mul);
        benchmark::ClobberMemory();
    }
}

static void BM_MatmulInplace(benchmark::State& state)
{

    const std::size_t N = state.range(0);

    std::vector<float> data(N*N, 1);
    std::vector<std::size_t> shape = {N, N};

    Tensor tensor1 = Tensor(data, shape, 0);
    Tensor tensor2 = Tensor(data, shape, 0);
    Tensor tensor3 = allocateTensor(shape);

    for (auto _ : state)
    {
        inplace_matmul(tensor1, tensor2, tensor3);

        benchmark::DoNotOptimize(tensor1);
        benchmark::DoNotOptimize(tensor2);
        benchmark::DoNotOptimize(tensor3);
        benchmark::ClobberMemory();

        std::fill(tensor3.point(),
          tensor3.point() + N*N,
          0.0f);
    }
}

BENCHMARK(BM_Matmul)
    ->Arg(64)
    ->Arg(512)
    ->Arg(1024);

BENCHMARK(BM_MatmulInplace)
    ->Arg(64)
    ->Arg(512)
    ->Arg(1024);