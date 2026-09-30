#include <sycl/sycl.hpp>
#include <iostream>

int main()
{
    sycl::queue q(sycl::gpu_selector_v);

    auto device = q.get_device();

    std::cout << "Device : "
              << device.get_info<
                     sycl::info::device::name>()
              << std::endl;

    std::cout << "Vendor : "
              << device.get_info<
                     sycl::info::device::vendor>()
              << std::endl;

    std::cout << "Driver : "
              << device.get_info<
                     sycl::info::device::driver_version>()
              << std::endl;

    return 0;
}
