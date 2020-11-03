/*
 * utility.hpp
 *
 *
 * Create Date : 2020-08-16 21:21:36
 * Copyright (c) 2019- Yusuke Kitamura <ymyk6602@gmail.com>
 */

#ifndef UTILITY_HPP__
#define UTILITY_HPP__

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <torch/script.h>


std::vector<float> kMEANS({0.406, 0.456, 0.485});  // imagenet parameter (BGR order)
std::vector<float> kSTDS({0.225, 0.224, 0.229});  // imagenet parameter (BGR order)


at::Tensor loadImageToTensor(std::string filename) {
    /*
     * TODO: speed test (and refactor)
     */
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    assert(image.empty());
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(image.channels());
    float data[image.cols * image.rows * image.channels()];
    int offset = image.cols * image.rows;
    int num_channels = channels.size();
    cv::split(image, channels);
    for (auto i = 0; i < num_channels; i++) {
        channels[i] = (channels[i] - kMEANS[i]) / kSTDS[i];
        std::memcpy((void*)(data + (num_channels - 1 - i) * offset),
                    (void*)channels[i].data, sizeof(float) * offset);
    }
    auto tensor = at::from_blob(data, {1, 3, image.rows, image.cols}, torch::kF32);
    return tensor.clone();
}


void saveTensorImage(std::string save_path, const at::Tensor& tensor) {
    auto shape = tensor.sizes();
    assert(shape.size() == 4);
    auto output_tensor = tensor.squeeze().permute({1, 2, 0}).contiguous();
    output_tensor = (output_tensor).mul(255).clamp(0, 255).to(torch::kU8);

    cv::Mat image(shape[2], shape[3], CV_8UC3);
    std::memcpy((void*)image.data, output_tensor.data_ptr(), sizeof(torch::kU8) * output_tensor.numel());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(save_path, image);
}


void speedTest(std::function<void()> func, int loop=1e3) {
    auto start_time = std::chrono::system_clock::now();
    for (auto i = 0; i < loop; i++) {
        func();
    }
    auto end_time = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << elapsed.count() / 1000.0 << " (sec)" << std::endl;
}

#endif // UTILITY_HPP__
