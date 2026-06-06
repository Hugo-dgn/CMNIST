#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

#include "tensor.hpp"

static void BM_MatrixTranspose(benchmark::State& state)
{

    const std::size_t N = state.range(0);

    std::vector<float> data(N*N, 1);
    std::vector<std::size_t> shape = {N, N};

    Tensor tensor1 = Tensor(data, shape, 0);

    for (auto _ : state)
    {
        Tensor tensor2 = tensor1.transpose();

        benchmark::DoNotOptimize(tensor1);
        benchmark::DoNotOptimize(tensor2);
        benchmark::ClobberMemory();
    }

}


BENCHMARK(BM_MatrixTranspose)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);