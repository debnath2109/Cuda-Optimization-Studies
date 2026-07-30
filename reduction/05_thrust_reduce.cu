#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <cstdio>

int main() {
    int n = 32'000'000;

    // thrust::device_vector manages GPU memory for you (RAII — allocates on device)
    thrust::device_vector<float> d_data(n, 1.0f);   // n elements, all 1.0

    // reduce: sum all elements. Args: begin, end, initial value, operation
    float sum = thrust::reduce(d_data.begin(), d_data.end(), 0.0f, thrust::plus<float>());

    printf("Thrust sum: %.1f\n", sum);   // expect 32000000.0
    return 0;
}