#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

#include "tensor.hpp"

static void BM_TensorTensorSUM(benchmark::State& state)
{
    std::vector<float> v(state.range(0), 1);
    Tensor tensor1 = Tensor(v);
    Tensor tensor2 = Tensor(v);

    for (auto _ : state)
    {
        Tensor sum = tensor1 + tensor2;

        benchmark::DoNotOptimize(tensor1);
        benchmark::DoNotOptimize(tensor2);
        benchmark::DoNotOptimize(sum);
    }
}

static void BM_TensorTensorInplaceSUM(benchmark::State& state)
{
    std::vector<float> v(state.range(0), 1);
    Tensor tensor1 = Tensor(v);
    Tensor tensor2 = Tensor(v);

    for (auto _ : state)
    {
        tensor1 += tensor2;

        benchmark::DoNotOptimize(tensor1);
        benchmark::DoNotOptimize(tensor2);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_TensorTensorSUM)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

BENCHMARK(BM_TensorTensorInplaceSUM)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);