#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <numeric>
#include <gtest/gtest.h>
#include "env.h"

template <typename T>
void GeneratorIndata(std::vector<int> &shape, T **output)
{
    *output =
        (T *)malloc(std::accumulate(shape.begin(), shape.end(), 0) * sizeof(T));
}

template <>
void GeneratorIndata(std::vector<int> &shape, int8_t **output);
template <>
void GeneratorIndata(std::vector<int> &shape, int16_t **output);
template <>
void GeneratorIndata(std::vector<int> &shape, int32_t **output);
template <>
void GeneratorIndata(std::vector<int> &shape, float **output);

int main(int argc, char **argv)
{
    printf("splice main testing\r\n");
    auto cpu_count = utils::ORTGetCPUCount();
    std::cout << "scpu count is: " << cpu_count << " " << std::endl;
    return 0;
}