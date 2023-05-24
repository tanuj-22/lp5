#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

void maxi(vector<int> arr)
{
    int max = arr[0];
    double start = omp_get_wtime();
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
        }
    }
    double end = omp_get_wtime();
    cout << "Max Seq: " << max << endl;
    cout << "Time Required: " << end - start << endl;

    max = arr[0];
    start = omp_get_wtime();
#pragma omp parallel for reduction(max : max)
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
        }
    }
    end = omp_get_wtime();
    cout << "Max Par: " << max << endl;
    cout << "Time Required: " << end - start << endl;
}

void mini(vector<int> arr)
{
    int min = arr[0];
    double start = omp_get_wtime();
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }
    double end = omp_get_wtime();
    cout << "Min Seq: " << min << endl;
    cout << "Time Required: " << end - start << endl;

    min = arr[0];
    start = omp_get_wtime();
#pragma omp parallel for reduction(min : min)
    for (int i = 0; i < arr.size(); i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }
    end = omp_get_wtime();

    cout << "Min Par: " << min << endl;
    cout << "Time Required: " << end - start << endl;
}

void sum(vector<int> arr)
{
    int sum = 0;
    double start = omp_get_wtime();
    for (int i = 0; i < arr.size(); i++)
    {
        sum += arr[i];
    }
    double end = omp_get_wtime();
    cout << "Sum Seq: " << sum << endl;
    cout << "Time Required: " << end - start << endl;

    sum = 0;
    start = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < arr.size(); i++)
    {
        sum += arr[i];
    }
    end = omp_get_wtime();

    cout << "Sum Par: " << sum << endl;
    cout << "Time Required: " << end - start << endl;
}

void average(vector<int> arr)
{

    int sum = 0;
    double start = omp_get_wtime();
    for (int i = 0; i < arr.size(); i++)
    {
        sum += arr[i];
    }
    double end = omp_get_wtime();
    cout << "Average Seq: " << (double)sum / arr.size() << endl;
    cout << "Time Required: " << end - start << endl;

    sum = 0;
    start = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < arr.size(); i++)
    {
        sum += arr[i];
    }
    end = omp_get_wtime();

    cout << "Average Par: " << (double)sum / arr.size() << endl;
    cout << "Time Required: " << end - start << endl;
}

int main()
{

    int size;
    cout << "Enter the size of the array: ";
    cin >> size;

    vector<int> arr(size);
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 1000;
    }

    cout << "Array: ";

    maxi(arr);
    mini(arr);
    sum(arr);
    average(arr);
}






























// Efficiency, speedup, and throughput are performance metrics used to evaluate the effectiveness and efficiency of parallel algorithms and systems.

// Efficiency:
// Efficiency measures how well a parallel algorithm or system utilizes the available computational resources. It is defined as the ratio of the speedup achieved by parallel execution to the maximum possible speedup.
// Efficiency formula:
// Efficiency = (Speedup / Number of Processors) * 100%

// Speedup:
// Speedup measures the improvement in performance achieved by parallel execution compared to sequential execution. It is defined as the ratio of the execution time of the sequential algorithm to the execution time of the parallel algorithm.
// Speedup formula:
// Speedup = Sequential Execution Time / Parallel Execution Time

// Throughput:
// Throughput measures the amount of work completed per unit of time. It represents the rate at which a system or algorithm can process tasks or data.
// Throughput formula:
// Throughput = Number of Tasks / Execution Time

// Here's an example to illustrate the calculation of efficiency, speedup, and throughput:

// Suppose we have a sequential algorithm that takes 10 seconds to process 100 tasks. We parallelize the algorithm and run it on 4 processors, and it takes 2 seconds to process the same 100 tasks.

// Sequential Execution Time = 10 seconds
// Parallel Execution Time = 2 seconds
// Number of Processors = 4
// Number of Tasks = 100

// Efficiency:
// Efficiency = (Speedup / Number of Processors) * 100%
// Speedup = Sequential Execution Time / Parallel Execution Time
// Speedup = 10 seconds / 2 seconds = 5
// Efficiency = (5 / 4) * 100% = 125%

// Speedup:
// Speedup = Sequential Execution Time / Parallel Execution Time
// Speedup = 10 seconds / 2 seconds = 5

// Throughput:
// Throughput = Number of Tasks / Execution Time
// Throughput = 100 tasks / 2 seconds = 50 tasks per second

// In this example, the efficiency is 125%, indicating that the parallel algorithm is utilizing the available resources effectively. The speedup is 5, meaning the parallel execution is 5 times faster than the sequential execution. The throughput is 50 tasks per second, indicating the rate at which the tasks are processed.

// These metrics are useful for evaluating the performance of parallel algorithms and systems, helping to identify the effectiveness of parallelization and resource utilization.