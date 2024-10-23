#include "../include/inference.hpp"
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <mutex>
#include <thread>

static std::once_flag initialize_flag;
static ov::Core core;
static ov::CompiledModel compiled_model;
static ov::InferRequest infer_request;
static ov::Output<const ov::Node> input_port;


static void initializeModel() {
    compiled_model = core.compile_model(MODEL_PATH, "CPU");
    infer_request = compiled_model.create_infer_request();
    input_port = compiled_model.input();
    std::cout << "Model initialized successfully!" << std::endl;
}


PreprocessedImage preprocessImage(const cv::Mat& img, cv::Size new_shape, cv::Scalar color) {
    cv::Size original_size = img.size();
    double scaling_factor = std::min((double)new_shape.width / original_size.width, 
                                     (double)new_shape.height / original_size.height);
    cv::Size new_unpad(int(original_size.width * scaling_factor), int(original_size.height * scaling_factor));

    cv::Mat resized_img;
    cv::resize(img, resized_img, new_unpad);

    int dw = new_shape.width - new_unpad.width;
    int dh = new_shape.height - new_unpad.height;

    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img, 0, dh, 0, dw, cv::BORDER_CONSTANT, color);

    cv::Mat blob = cv::dnn::blobFromImage(padded_img, 1.0 / 255.0, new_shape, cv::Scalar(), true, false);

    PreprocessedImage data;
    data.scale = scaling_factor;
    data.blob = blob;
    data.input_image = img;

    return data;
}


std::vector<DetectedObject> runInferenceAndPostprocess(const PreprocessedImage& preprocessed_data) {
    std::call_once(initialize_flag, initializeModel);

    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), preprocessed_data.blob.ptr<float>());
    infer_request.set_input_tensor(input_tensor);

    infer_request.infer();

    auto output_tensor = infer_request.get_output_tensor();
    auto output_shape = output_tensor.get_shape();
    float* output_data = output_tensor.data<float>();

    std::vector<DetectedObject> detected_objects;
    for (size_t i = 0; i < output_shape[1]; ++i) {
        float confidence = output_data[i * output_shape[2] + 4];  
        if (confidence > CONF_THRESHOLD) {
            float cx = output_data[i * output_shape[2] + 0];
            float cy = output_data[i * output_shape[2] + 1];
            float width = output_data[i * output_shape[2] + 2];
            float height = output_data[i * output_shape[2] + 3];

            int left = static_cast<int>((cx - width / 2) * preprocessed_data.scale);
            int top = static_cast<int>((cy - height / 2) * preprocessed_data.scale);
            int right = static_cast<int>(width * preprocessed_data.scale);
            int bottom = static_cast<int>(height * preprocessed_data.scale);

            DetectedObject obj;
            obj.confidence = confidence;
            obj.box = cv::Rect(left, top, right, bottom);
            obj.class_id = static_cast<int>(output_data[i * output_shape[2] + 5]); 
            detected_objects.push_back(obj);
        }
    }

    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (const auto& obj : detected_objects) {
        boxes.push_back(obj.box);
        confidences.push_back(obj.confidence);
    }
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<DetectedObject> final_detections;
    for (int idx : indices) {
        final_detections.push_back(detected_objects[idx]);
    }

    return final_detections;
}
