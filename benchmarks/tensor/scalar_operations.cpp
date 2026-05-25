#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

#include "tensor.hpp"

static void BM_TensorScalarSUM(benchmark::State& state)
{
    std::vector<float> v(state.range(0), 1);
    Tensor tensor = Tensor(v);

    for (auto _ : state)
    {
        Tensor sum = tensor + 1.0f;

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_TensorInplaceScalarSUM(benchmark::State& state)
{
    std::vector<float> v(state.range(0), 1);
    Tensor tensor = Tensor(v);

    for (auto _ : state)
    {
        tensor += 1.0f;

        benchmark::DoNotOptimize(tensor);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_TensorScalarSUM)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);

BENCHMARK(BM_TensorInplaceScalarSUM)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);