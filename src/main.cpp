#include "../include/inference.hpp"

int main() {
    cv::VideoCapture cap(0);  
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频流！" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        PreprocessedImage preprocessed_data = preprocessImage(frame);
        std::vector<DetectedObject> detections = runInferenceAndPostprocess(preprocessed_data);

        for (const auto& obj : detections) {
            cv::rectangle(frame, obj.box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, std::to_string(obj.class_id), obj.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Object Detection", frame);
        if (cv::waitKey(30) == 27) {  
            break;
        }
    }

    return 0;
}
