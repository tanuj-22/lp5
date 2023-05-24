#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

void seq_bubble_sort(vector<int> arr, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void par_bubble_sort(vector<int> arr, int n)
{
    for (int k = 0; k < n; k++)
    {
        if (k % 2 == 0)
        {
#pragma omp parallel for
            for (int i = 1; i < n - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    swap(arr[i], arr[i + 1]);
                }
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < n - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    swap(arr[i], arr[i + 1]);
                }
            }
        }
    }
}

void merge(vector<int> arr, int l, int m, int r)
{
    vector<int> temp;
    int i = l;
    int j = m + 1;

    while (i <= m && j <= r)
    {
        if (arr[i] < arr[j])
        {
            temp.push_back(arr[i]);
            i++;
        }
        else
        {
            temp.push_back(arr[j]);
            j++;
        }
    }

    while (i <= m)
    {
        temp.push_back(arr[i]);
        i++;
    }

    while (j <= r)
    {
        temp.push_back(arr[j]);
        j++;
    }

    for (int i = l; i <= r; i++)
    {
        arr[i] = temp[i - l];
    }
}

void seq_merge_sort(vector<int> arr, int l, int r)
{
    if (l >= r)
    {
        return;
    }
    int mid = (l + r) / 2;
    seq_merge_sort(arr, l, mid);
    seq_merge_sort(arr, mid + 1, r);
    merge(arr, l, mid, r);
}

void par_merge_sort(vector<int> arr, int l, int r)
{
    if (l >= r)
    {
        return;
    }
    int mid = (l + r) / 2;
#pragma omp parallel sections
    {
#pragma omp section
        {
            par_merge_sort(arr, l, mid);
        }
#pragma omp section
        {
            par_merge_sort(arr, mid + 1, r);
        }
    }
    merge(arr, l, mid, r);
}

int main()
{

    int size;
    cout << "Enter the size of the array: ";
    cin >> size;

    vector<int> arr(size);
    for (int i = 0; i < size; i++)
    {
        arr[i] = rand() % 100;
    }

    cout << endl;

    double start = omp_get_wtime();
    seq_bubble_sort(arr, size);
    double end = omp_get_wtime();

    cout << "Time Required Seq: " << end - start << endl;

    start = omp_get_wtime();
    par_bubble_sort(arr, size);
    end = omp_get_wtime();

    cout << "Time Required Par: " << end - start << endl;

    cout << endl;

    start = omp_get_wtime();
    seq_merge_sort(arr, 0, size - 1);
    end = omp_get_wtime();

    cout << "Time Required Seq: " << end - start << endl;

    start = omp_get_wtime();
    par_merge_sort(arr, 0, size - 1);
    end = omp_get_wtime();

    cout << "Time Required Par: " << end - start << endl;

    // get number of processors in this system
    int n = omp_get_num_procs();
    cout << "Number of processors: " << n << endl;

    return 0;
}

























// nlogn
// logn

// odd even sort
// n^2

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