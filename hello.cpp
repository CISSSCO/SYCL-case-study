#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q;

    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    q.single_task([]() {
        // SYCL kernel
    }).wait();

    std::cout << "Hello World from SYCL!\n";

    return 0;
}
