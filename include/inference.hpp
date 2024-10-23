#ifndef OBJECT_INFERENCE_H
#define OBJECT_INFERENCE_H

#include <iostream>
#include <string>
#include <vector>
#include <openvino/openvino.hpp>  
#include <opencv2/opencv.hpp>     

#define MODEL_PATH "../model/object_detection_model.xml"
#define CONF_THRESHOLD 0.8   
#define NMS_THRESHOLD 0.4     
#define INPUT_SIZE 416        

struct PreprocessedImage {
    float scale;
    cv::Mat blob;
    cv::Mat input_image;
};

struct DetectedObject {
    float confidence;
    cv::Rect box;
    int class_id;
};

PreprocessedImage preprocessImage(const cv::Mat& img, cv::Size new_shape = cv::Size(INPUT_SIZE, INPUT_SIZE));
std::vector<DetectedObject> runInferenceAndPostprocess(const PreprocessedImage& preprocessed_data);

#endif 
