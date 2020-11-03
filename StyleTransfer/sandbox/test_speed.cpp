#include <memory>

#include <boost/filesystem.hpp>
#include <torch/script.h>

#include "utility.hpp"

namespace fs = boost::filesystem;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage : test_speed <path-to-exported-scrpit-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    fs::path input_path(
        "/home/kitamura/dataset/COCO/train2014/Original/COCO_train2014_000000000009.jpg");
    at::Tensor input_tensor = loadImageToTensor(input_path.generic_string());

    auto run_module = [&module, &input_tensor]() {
        module.forward({input_tensor}).toTensor();
    };
    speedTest(run_module, 100);
    // auto max_idx = output_tensor.argmax().item<int>();
    // auto max_conf = output_tensor.max().item<float>();
    // std::cout << "max index : " << max_idx << ", max confidence : " << max_conf << std::endl;
}
