#include <iostream>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <torch/script.h>

#include "utility.hpp"

namespace fs = boost::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage : test_torchscript <path-to-exported-scrpit-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loadint the model\n";
        return -1;
    }

    fs::path content_path(
        "/home/kitamura/dataset/COCO/train2014/Original/COCO_train2014_000000000009.jpg");
    fs::path style_path(
        "/home/kitamura/dataset/AbstractGallary/Abstract_image_1030.jpg");
    fs::path save_dir = fs::path(__FILE__).parent_path() / "output";
    fs::create_directory(save_dir);

    at::Tensor content_tensor = loadImageToTensor(content_path.generic_string());
    at::Tensor style_tensor = loadImageToTensor(style_path.generic_string());

    auto func = [&module, &content_tensor, &style_tensor]() {
        module.forward({content_tensor, style_tensor}).toTensor();
    };
    speedTest(func, 100);
    // saveTensorImage((save_dir / "output.png").generic_string(), output_tensor);
}
