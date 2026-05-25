#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

// A simple benchmark
static void BM_VectorSum(benchmark::State& state)
{
    std::vector<int> v(state.range(0), 1);

    for (auto _ : state)
    {
        int sum = std::accumulate(v.begin(), v.end(), 0);

        benchmark::DoNotOptimize(sum);
    }
}

// Run benchmark with different vector sizes
BENCHMARK(BM_VectorSum)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(100000);