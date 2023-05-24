#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

class Graph
{
public:
    int vertices = 6;
    int edges = 5;
    vector<vector<int>> graph = {{1}, {0, 2, 3}, {1, 4, 5}, {1, 4}, {2, 3}, {2}};
    // vector<vector<int>> graph = {{1, 2, 3}, {0, 2, 4}, {0, 1, 3, 4}, {0, 2, 5}, {1, 2, 6, 7}, {3, 7}, {4, 8}, {4, 5, 9}, {6, 9}, {7, 8}, {11, 12}, {10, 12}, {10, 11, 13}, {12, 14}, {13, 15}, {15, 16, 17}, {14, 16}, {14, 15, 17}, {15, 16, 18, 19}, {17, 19}, {19, 20, 21}, {19, 20, 22}, {21, 22, 23}, {21, 22, 24}, {23, 24, 25}, {23, 24, 26}, {25, 26, 27}, {25, 26, 28, 29}, {27, 29}, {29, 30}, {29, 30, 31}, {31, 32}, {31, 32, 33}, {33, 34}, {33, 34, 35}, {35, 36}, {35, 36, 37}, {37, 38}, {37, 38, 39}, {39, 40}, {39, 40, 41}, {41, 42}, {41, 42, 43}, {43, 44}, {43, 44, 45}, {45, 46}, {45, 46, 47}, {47, 48}, {47, 48, 49}};
    vector<bool> visited;

    void printGraph()
    {
        for (int i = 0; i < vertices; i++)
        {
            cout << i << " -> ";
            for (auto j = graph[i].begin(); j != graph[i].end(); j++)
            {
                cout << *j << " ";
            }
            cout << endl;
        }
    }

    void initialize_visited()
    {
        visited.assign(vertices, false);
    }

    void dfs(int i)
    {
        stack<int> s;
        s.push(i);
        visited[i] = true;

        while (s.empty() != true)
        {
            int current = s.top();
            cout << current << " ";
            s.pop();
            for (auto j = graph[current].begin(); j != graph[current].end(); j++)
            {
                if (visited[*j] == false)
                {
                    s.push(*j);
                    visited[*j] = true;
                }
            }
        }
    }

    void parallel_dfs(int i)
    {
        stack<int> s;
        s.push(i);
        visited[i] = true;

        while (s.empty() != true)
        {
            int current;
#pragma omp critical
            {
                current = s.top();
                cout << current << " ";
                s.pop();
            }
#pragma omp parallel for
            for (auto j = graph[current].begin(); j != graph[current].end(); j++)
            {
                if (visited[*j] == false)
                {
#pragma omp critical
                    {
                        s.push(*j);
                        visited[*j] = true;
                    }
                }
            }
        }
    }

    void bfs(int i)
    {
        queue<int> q;
        q.push(i);
        visited[i] = true;

        while (q.empty() != true)
        {
            int current = q.front();
            q.pop();
            cout << current << " ";
            for (auto j = graph[current].begin(); j != graph[current].end(); j++)
            {
                if (visited[*j] == false)
                {
                    q.push(*j);
                    visited[*j] = true;
                }
            }
        }
    }

    void parallel_bfs(int i)
    {
        queue<int> q;
        q.push(i);
        visited[i] = true;

        while (q.empty() != true)
        {

            int current;
#pragma omp critical
            current = q.front();
            q.pop();
            cout << current << " ";

#pragma omp parallel for
            for (auto j = graph[current].begin(); j != graph[current].end(); j++)
            {
                if (visited[*j] == false)
                {
#pragma omp critical
                    q.push(*j);
                    visited[*j] = true;
                }
            }
        }
    }
};

int main(int argc, char const *argv[])
{
    Graph g;
    cout << "Adjacency List:\n";
    g.printGraph();
    g.initialize_visited();
    cout << "Depth First Search: \n";
    double start = omp_get_wtime();
    g.dfs(0);
    cout << endl;
    double end = omp_get_wtime();
    cout << "Time taken: " << (end - start) << " seconds" << endl;
    cout << "Parallel Depth First Search: \n";
    g.initialize_visited();
    start = omp_get_wtime();
    g.parallel_dfs(0);
    cout << endl;
    end = omp_get_wtime();
    cout << "Time taken: " << end - start << " seconds" << endl;
    start = omp_get_wtime();
    cout << "Breadth First Search: \n";
    g.initialize_visited();
    g.bfs(0);
    cout << endl;
    end = omp_get_wtime();
    cout << "Time taken: " << end - start << " seconds" << endl;
    start = omp_get_wtime();
    cout << "Parallel Breadth First Search: \n";
    g.initialize_visited();
    g.parallel_bfs(0);
    cout << endl;
    end = omp_get_wtime();
    cout << "Time taken: " << end - start << " seconds" << endl;

    return 0;
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


















// #include <iostream>
// #include <queue>
// #include <stack>
// #include <chrono>
// #include <omp.h>

// using namespace std;

// struct Node {
//     int data;
//     Node* left;
//     Node* right;

//     Node(int value) : data(value), left(nullptr), right(nullptr) {}
// };

// class BinaryTree {
// public:
//     Node* root;

//     BinaryTree() : root(nullptr) {}

//     void insert(int value) {
//         root = insertRecursive(root, value);
//     }

//     Node* insertRecursive(Node* node, int value) {
//         if (node == nullptr) {
//             node = new Node(value);
//         } else {
//             if (value <= node->data) {
//                 node->left = insertRecursive(node->left, value);
//             } else {
//                 node->right = insertRecursive(node->right, value);
//             }
//         }
//         return node;
//     }

//     void printInorder() {
//         printInorderRecursive(root);
//     }

//     void printInorderRecursive(Node* node) {
//         if (node != nullptr) {
//             printInorderRecursive(node->left);
//             cout << node->data << " ";
//             printInorderRecursive(node->right);
//         }
//     }

//     void dfs() {
//         dfsRecursive(root);
//     }

//     void dfsRecursive(Node* node) {
//         if (node != nullptr) {
//             cout << node->data << " ";
//             dfsRecursive(node->left);
//             dfsRecursive(node->right);
//         }
//     }

//     void parallelDfs() {
//         #pragma omp parallel
//         {
//             #pragma omp single
//             parallelDfsRecursive(root);
//         }
//     }

//     void parallelDfsRecursive(Node* node) {
//         if (node != nullptr) {
//             cout << node->data << " ";

//             #pragma omp task
//             parallelDfsRecursive(node->left);

//             #pragma omp task
//             parallelDfsRecursive(node->right);
//         }
//     }

//     void bfs() {
//         queue<Node*> q;
//         q.push(root);

//         while (!q.empty()) {
//             Node* node = q.front();
//             q.pop();

//             if (node != nullptr) {
//                 cout << node->data << " ";
//                 q.push(node->left);
//                 q.push(node->right);
//             }
//         }
//     }

//     void parallelBfs() {
//         queue<Node*> q;
//         q.push(root);

//         #pragma omp parallel
//         {
//             while (!q.empty()) {
//                 #pragma omp for
//                 for (int i = 0; i < q.size(); i++) {
//                     Node* node;
//                     #pragma omp critical
//                     {
//                         node = q.front();
//                         q.pop();
//                     }

//                     if (node != nullptr) {
//                         cout << node->data << " ";

//                         #pragma omp critical
//                         {
//                             q.push(node->left);
//                             q.push(node->right);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// };

// int main() {
//     BinaryTree tree;
//     tree.insert(4);
//     tree.insert(2);
//     tree.insert(1);
//     tree.insert(3);
//     tree.insert(6);
//     tree.insert(5);
//     tree.insert(7);

//     cout << "Inorder Traversal: ";
//     tree.printInorder();
//     cout << endl;

//     cout << "Depth First Search (DFS): ";
//     auto start = chrono::high_resolution_clock::now();
//     tree.dfs();
//     auto end = chrono::high_resolution_clock::now();
//     cout << "\nTime taken (DFS): " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl
//     cout << "Parallel Depth First Search (DFS): ";
//     start = chrono::high_resolution_clock::now();
//     tree.parallelDfs();
//     end = chrono::high_resolution_clock::now();
//     cout << "\nTime taken (Parallel DFS): " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

//     cout << "Breadth First Search (BFS): ";
//     start = chrono::high_resolution_clock::now();
//     tree.bfs();
//     end = chrono::high_resolution_clock::now();
//     cout << "\nTime taken (BFS): " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

//     cout << "Parallel Breadth First Search (BFS): ";
//     start = chrono::high_resolution_clock::now();
//     tree.parallelBfs();
//     end = chrono::high_resolution_clock::now();
//     cout << "\nTime taken (Parallel BFS): " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

//     return 0;
// }