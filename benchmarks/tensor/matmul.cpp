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

    const double flops = 2.0 * N * N * N;

    state.counters["FLOPS"] =
        benchmark::Counter(
            flops,
            benchmark::Counter::kIsIterationInvariantRate);

}

static void BM_MatmulTranspose(benchmark::State& state)
{

    const std::size_t N = state.range(0);

    std::vector<float> data(N*N, 1);
    std::vector<std::size_t> shape = {N, N};

    Tensor tensor1 = Tensor(data, shape, 0);
    Tensor tensor2 = Tensor(data, shape, 0);

    for (auto _ : state)
    {
        Tensor mul = matmul_transpose(tensor1, tensor2.transpose());

        benchmark::DoNotOptimize(tensor1);
        benchmark::DoNotOptimize(tensor2);
        benchmark::DoNotOptimize(mul);
        benchmark::ClobberMemory();
    }

    const double flops = 2.0 * N * N * N;

    state.counters["FLOPS"] =
        benchmark::Counter(
            flops,
            benchmark::Counter::kIsIterationInvariantRate);

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

    const double flops = 2.0 * N * N * N;

    state.counters["FLOPS"] =
        benchmark::Counter(
            flops,
            benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_MatmulInplaceTranspose(benchmark::State& state)
{

    const std::size_t N = state.range(0);

    std::vector<float> data(N*N, 1);
    std::vector<std::size_t> shape = {N, N};

    Tensor tensor1 = Tensor(data, shape, 0);
    Tensor ttensor2 = Tensor(data, shape, 0).transpose();
    Tensor tensor3 = allocateTensor(shape);

    for (auto _ : state)
    {
        inplace_matmul_transpose(tensor1, ttensor2, tensor3);

        benchmark::DoNotOptimize(tensor1);
        benchmark::DoNotOptimize(ttensor2);
        benchmark::DoNotOptimize(tensor3);
        benchmark::ClobberMemory();

        std::fill(tensor3.point(),
          tensor3.point() + N*N,
          0.0f);
    }

    const double flops = 2.0 * N * N * N;

    state.counters["FLOPS"] =
        benchmark::Counter(
            flops,
            benchmark::Counter::kIsIterationInvariantRate);
}


BENCHMARK(BM_Matmul)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatmulTranspose)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatmulInplace)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_MatmulInplaceTranspose)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Arg(4096)
    ->Unit(benchmark::kMicrosecond);