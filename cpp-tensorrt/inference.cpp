// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << msg << std::endl;
    }
} gLogger;

void Inference::doInference(IExecutionContext& context, float* inputHost, float* outputHost, int inputH, int inputW, int batchSize, int rows, int dimensions) {
    const ICudaEngine& engine = context.getEngine();
    // int inputIndex = engine.getBindingIndex("images");
    // int outputIndex = engine.getBindingIndex("output0");

    // 分配设备内存
    void* buffers[2];
    cudaMalloc(&buffers[0], batchSize * 3 * inputH * inputW * sizeof(float));
    cudaMalloc(&buffers[1], batchSize * dimensions * rows * sizeof(float));

    // Host -> Device 拷贝
    cudaMemcpy(buffers[0], inputHost, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice);

    // 推理
    context.executeV2(buffers);

    // Device -> Host 拷贝
    cudaMemcpy(outputHost, buffers[1], batchSize * dimensions * rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    cv::Mat outputMat = cv::Mat(dimensions, rows, CV_32F, outputHost);
    cv::Mat transposedMat = outputMat.t();
    memcpy(outputHost, transposedMat.data, dimensions * rows * sizeof(float));
}

template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return nullptr;
}

float* Inference::runTrt(cv::Mat &modelInput, const std::string &trtModelPath, int rows, int dimensions) {
    // 加载引擎
    IRuntime* runtime = createInferRuntime(gLogger);
    std::ifstream file(trtModelPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open TensorRT model file: " << trtModelPath << std::endl;
        return nullptr; // Or throw an exception
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    if (size == static_cast<size_t>(-1)) { // Check if tellg() returned an error
        std::cerr << "Error: Could not determine size of TensorRT model file: " << trtModelPath << std::endl;
        file.close();
        return nullptr;
    }
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    if (!file.read(engine_data.data(), size)) {
        std::cerr << "Error: Could not read TensorRT model file: " << trtModelPath << std::endl;
        file.close();
        return nullptr;
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    IExecutionContext* context = engine->createExecutionContext();

    // 预处理
    int inputH = static_cast<int>(modelShape.height);
    int inputW = static_cast<int>(modelShape.width);
    float* inputHost = new float[3 * inputH * inputW];
    float* outputHost = new float[dimensions * rows];
    BlobFromImage(modelInput, inputHost);

    // 推理
    doInference(*context, inputHost, outputHost, inputH, inputW, 1, rows, dimensions);

    // context->destroy();
    // engine->destroy();
    // runtime->destroy();
    return outputHost;
}

Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    // loadOnnxNetwork();
    // loadOnnxNetworkOnnxRuntime();
    // loadClassesFromFile(); The classes are hard-coded for this example
}

std::vector<Detection> Inference::runInference(const cv::Mat &input, const bool &onnxRuntime, const std::string &trtModelPath)
{
    cv::Mat modelInput = input;
    int pad_x, pad_y;
    float scale;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput, &pad_x, &pad_y, &scale);
    int rows, dimensions;
    bool yolov8 = false;
    float *data;

    if (!trtModelPath.empty()) {
        // tensorrt inference
        rows = 8400;
        dimensions = 84;
        yolov8 = true;

        data = runTrt(modelInput, trtModelPath, rows, dimensions);
    }
    // } else {
    //     if (onnxRuntime) {
    //         // onnx inference with onnx runtime
    //         float* blob_new = new float[modelInput.total() * 3];
    //         BlobFromImage(modelInput, blob_new);
    //         std::vector<int64_t> YOLO_input_node_dims = { 1, 3, modelShape.width, modelShape.height };
    //         Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    //             Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob_new, 3 * modelShape.width * modelShape.height,
    //             YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
    //         auto outputTensor = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
    //             outputNodeNames.size());
    //         Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    //         auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    //         std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    //         auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<float>::type>();
    //         rows = outputNodeDims[1];//84
    //         dimensions = outputNodeDims[2];//8400
    //         cv::Mat rawData = cv::Mat(rows, dimensions, CV_32F, output);
    //         rawData = rawData.t();
    //         data = (float*)rawData.data;
    //         if (dimensions > rows) 
    //         {
    //             yolov8 = true;
    //             int t = rows;
    //             rows = dimensions;
    //             dimensions = t;
    //         }
    //     } else {
    //         // onnx inference with opencv
    //         cv::Mat blob;
    //         cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
    //         net.setInput(blob);

    //         std::vector<cv::Mat> outputs;
    //         net.forward(outputs, net.getUnconnectedOutLayersNames());

    //         rows = outputs[0].size[1];
    //         dimensions = outputs[0].size[2];

    //         // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    //         // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    //         if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    //         {
    //             yolov8 = true;
    //             rows = outputs[0].size[2];
    //             dimensions = outputs[0].size[1];

    //             outputs[0] = outputs[0].reshape(1, dimensions);
    //             cv::transpose(outputs[0], outputs[0]);
    //         }
    //         data = (float *)outputs[0].data;
    //     }
    // }

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data+4;

            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > modelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w - pad_x) / scale);
                int top = int((y - 0.5 * h - pad_y) / scale);

                int width = int(w / scale);
                int height = int(h / scale);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // yolov5
        {
            float confidence = data[4];

            if (confidence >= modelConfidenceThreshold)
            {
                float *classes_scores = data+5;

                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > modelScoreThreshold)
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w - pad_x) / scale);
                    int top = int((y - 0.5 * h - pad_y) / scale);

                    int width = int(w / scale);
                    int height = int(h / scale);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}

void Inference::loadClassesFromFile()
{
    std::ifstream inputFile(classesPath);
    if (inputFile.is_open())
    {
        std::string classLine;
        while (std::getline(inputFile, classLine))
            classes.push_back(classLine);
        inputFile.close();
    }
}

// void Inference::loadOnnxNetwork()
// {
//     net = cv::dnn::readNetFromONNX(modelPath);
//     if (cudaEnabled)
//     {
//         std::cout << "\nRunning on CUDA" << std::endl;
//         net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//         net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
//     }
//     else
//     {
//         std::cout << "\nRunning on CPU" << std::endl;
//         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//         net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//     }
// }

// void Inference::loadOnnxNetworkOnnxRuntime()
// {
//     env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
//     Ort::SessionOptions sessionOption;
//     sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
//     sessionOption.SetIntraOpNumThreads(1);
//     sessionOption.SetLogSeverityLevel(3);
//     session = new Ort::Session(env, modelPath.c_str(), sessionOption);
//     Ort::AllocatorWithDefaultOptions allocator;
//     size_t inputNodesNum = session->GetInputCount();
//     for (size_t i = 0; i < inputNodesNum; i++)
//     {
//         Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
//         char* temp_buf = new char[50];
//         strcpy(temp_buf, input_node_name.get());
//         inputNodeNames.push_back(temp_buf);
//     }
//     size_t OutputNodesNum = session->GetOutputCount();
//     for (size_t i = 0; i < OutputNodesNum; i++)
//     {
//         Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
//         char* temp_buf = new char[10];
//         strcpy(temp_buf, output_node_name.get());
//         outputNodeNames.push_back(temp_buf);
//     }
//     options = Ort::RunOptions{ nullptr };
// }

cv::Mat Inference::formatToSquare(const cv::Mat &source, int *pad_x, int *pad_y, float *scale)
{
    int col = source.cols;
    int row = source.rows;
    int m_inputWidth = modelShape.width;
    int m_inputHeight = modelShape.height;

    *scale = std::min(m_inputWidth / (float)col, m_inputHeight / (float)row);
    int resized_w = col * *scale;
    int resized_h = row * *scale;
    *pad_x = (m_inputWidth - resized_w) / 2;
    *pad_y = (m_inputHeight - resized_h) / 2;

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(resized_w, resized_h));
    cv::Mat result = cv::Mat::zeros(m_inputHeight, m_inputWidth, source.type());
    resized.copyTo(result(cv::Rect(*pad_x, *pad_y, resized_w, resized_h)));
    resized.release();
    return result;
}
