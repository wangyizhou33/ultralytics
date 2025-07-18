// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"
#include <filesystem>



using namespace std;
using namespace cv;

void postProcess(Inference &inf, const std::string &image_path, bool onnxRuntime, const std::string &trtModelPath) {
    cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);
    // Inference starts here...
    std::vector<Detection> output = inf.runInference(frame, onnxRuntime, trtModelPath);
    int detections = output.size();

    for (int i = 0; i < detections; ++i)
    {
        Detection detection = output[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(frame, box, color, 2);

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);
        cv::Rect textBox(box.x, box.y - 20, textSize.width + 10, textSize.height + 10);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 5), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1, 0);
    }
    // Inference ends here...

    // This is only for preview purposes
    // float scale = 0.8;
    // cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
    // cv::imshow("Inference", frame);

    // cv::waitKey(-1);
    if (!trtModelPath.empty()) {
        std::cout << "Writing result to output_trt_cpp.jpg" << std::endl;
        cv::imwrite("output_trt_cpp.jpg", frame);
    } else {
        if (onnxRuntime) {
            cv::imwrite("output_onnx_cpp_onnx_runtime.jpg", frame);
        } else {
            cv::imwrite("output_onnx_cpp.jpg", frame);
        }
    }
}

int main(int argc, char **argv)
{
    // std::string projectBasePath = "/home/user/ultralytics"; // Set your ultralytics base path
    std::string projectBasePath = "/home/yizhouw/Repositories/ultralytics/cpp-tensorrt";

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    // Inference inf(projectBasePath + "/yolov8s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);
    Inference inf(projectBasePath + "/best.engine", cv::Size(640, 640), "classes.txt", runOnGPU);
 

    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images";

    std::vector<std::string> imageNames;
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/bus.jpg");
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");
    

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            imageNames.push_back(img_path);
        }
    }

    std::string trtModelPath = "/home/yizhouw/Repositories/ultralytics/cpp-tensorrt/best.engine";

    for (int i = 0; i < imageNames.size(); ++i)
    {
        // postProcess(inf, imageNames[i], true, "");
        // postProcess(inf, imageNames[i], false, "");
        postProcess(inf, imageNames[i], false, trtModelPath);
    }
}
