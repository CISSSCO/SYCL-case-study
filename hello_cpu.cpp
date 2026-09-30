#include <sycl/sycl.hpp>
#include <iostream>

int main()
{
    sycl::queue q(sycl::cpu_selector_v);

    auto device = q.get_device();

    std::cout << "CPU Device: "
              << device.get_info<
                     sycl::info::device::name>()
              << std::endl;

    q.parallel_for(
        sycl::range<1>(100),
        [=](sycl::id<1> i) {
            // CPU work
        }
    ).wait();

    return 0;
}
