#include <iostream>
#include <thread>
#include <vector>

void hello(int thread_id)
{
    std::cout << "Hello from thread " << thread_id
              << " (native id: " << std::this_thread::get_id() << ")\n";
}

int main()
{
    const int num_threads = 8;
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i)
        threads.emplace_back(hello, i);

    for (auto &t : threads)
        t.join();

    return 0;
}