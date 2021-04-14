#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace InferenceEngine;

std::vector<float> anchor = {
	10, 13,  16, 30,  33, 23,	// 8倍降采样下的anchor比率		-	小目标
	30, 61,  62, 45,  59,119,	// 16倍降采样下的anchor比率	-	中目标
	116,90, 156,198, 373,326,	// 32倍降采样下的anchor比率	-	大目标
};

int get_anchor_index(int scale_w, int scale_h) {
	if (scale_w == 20) {	// 32倍降采样
		return 12;
	}
	if (scale_w == 40) {	// 16倍降采样
		return 6;
	}
	if (scale_w == 80) {	// 8倍降采样
		return 0;
	}
	return -1;
}

float get_stride(int scale_w, int scale_h) {
	if (scale_w == 20) {	// 32倍降采样
		return 32.0;
	}
	if (scale_w == 40) {	// 16倍降采样
		return 16.0;
	}
	if (scale_w == 80) {	// 8倍降采样
		return 8.0;
	}
	return -1;
}

float sigmoid_function(float a) {
	return 1.f / (1.f + exp(-a));
}

// https://gitee.com/opencv_ai/opencv_tutorial_data/blob/master/source/openvino/openvino_yolov5s_demo.cpp
int main() {
	int i = 0, j = 0, c = 0, d = 0;

	// 测试图像
	Mat src = imread("test1.jpg");
	int image_height	= src.rows;
	int image_width		= src.cols;

	// 创建ie插件，查询支持硬件设备
	Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (i = 0; i < availableDevices.size(); ++i) {
		cout << "supported device name: " << availableDevices[i] << endl;
	}
	// 加载检测模型
	auto network = ie.ReadNetwork("yolov5s.xml", "yolov5s.bin");
	//auto network = ie.ReadNetwork("*.onnx");

	// 设置输入格式
	InputsDataMap input_info(network.getInputsInfo());
	for (auto &item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
		// 显示输入维度
		cout << "input_name : " << item.first << endl;
		auto input_shape = input_data->getTensorDesc().getDims();
		cout << input_shape[0] << " " << input_shape[1] << " " << input_shape[2] << " "
			<< input_shape[3] << " " << endl;
	}
	// 设置输出格式
	OutputsDataMap output_info(network.getOutputsInfo());
	for (auto &item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
		// 显示输出维度
		auto out_shape = output_data->getTensorDesc().getDims();
		cout << "output_name : " << item.first << endl;
		cout << out_shape[0] << " " << out_shape[1] << " " << out_shape[2] << " "
			<< out_shape[3] << " " << out_shape[4] << " " << endl;
	}
	// Loading model to the device
	//auto executable_network = ie.LoadNetwork(network, "GPU");
	auto executable_network = ie.LoadNetwork(network, "CPU");

	// 处理解析输出结果
	vector<Rect>	boxes;
	vector<int>		classIds;
	vector<float>	confidences;

	// Create infer request
	InferRequest infer_request = executable_network.CreateInferRequest();
	float scale_x = image_width / 640.0;
	float scale_y = image_height / 640.0;

	// 设置输入图像数据并实现推理预测
	int64 start = getTickCount();
	/* Iterating over all input blobs */
	for (auto &item : input_info) {
		auto input_name = item.first;

		/* Getting input blob */
		auto input = infer_request.GetBlob(input_name);
		size_t num_channels = input->getTensorDesc().getDims()[1];
		size_t h = input->getTensorDesc().getDims()[2];
		size_t w = input->getTensorDesc().getDims()[3];
		size_t image_size = h * w;
		Mat blob_image;
		resize(src, blob_image, Size(w, h));
		cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

		// NCHW
		float *data = static_cast<float*>(input->buffer());
		for (size_t row = 0; row < h; row++) {
			for (size_t col = 0; col < w; col++) {
				for (size_t ch = 0; ch < num_channels; ch++) {
					data[image_size * ch + row * w + col] = float(blob_image.at<Vec3b>(row, col)[ch] / 255.0);
				}
			}
		}
	}
	// 执行预测
	infer_request.Infer();

	// 输出解析
	for (auto &item : output_info) {
		auto output_name = item.first;
		cout << "output_name: " << output_name.c_str() << endl;
		auto output = infer_request.GetBlob(output_name);

		const float *output_blob = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
		const SizeVector outputDims = output->getTensorDesc().getDims();
		const int out_n = outputDims[0];
		const int out_c = outputDims[1];
		const int side_h = outputDims[2];
		const int side_w = outputDims[3];
		const int side_data = outputDims[4];
		float stride = get_stride(side_h, side_h);
		int anchor_index = get_anchor_index(side_h, side_h);
		cout << "number of images: " << out_n << ", channels: " << out_c << ", height: " << side_h <<
			", width: " << side_w << ", out_data: " << side_data << endl;
		int side_square = side_h * side_w;
		int side_data_square = side_square * side_data;
		int side_data_w = side_w * side_data;
		for (i = 0; i < side_square; ++i) {
			for (c = 0; c < out_c; c++) {	// 3个anchor
				int row = i / side_h;
				int col = i % side_h;
				int object_index = c * side_data_square + row * side_data_w + col * side_data;

				// 阈值过滤
				float conf = sigmoid_function(output_blob[object_index + 4]);
				if (conf < 0.25) {
					continue;
				}

				// 解析cx,cy,width,height
				float x = (sigmoid_function(output_blob[object_index]) * 2 - 0.5 + col) * stride;
				float y = (sigmoid_function(output_blob[object_index + 1]) * 2 - 0.5 + row) * stride;
				float w = pow(sigmoid_function(output_blob[object_index + 2]) * 2, 2) * anchor[anchor_index + c * 2];
				float h = pow(sigmoid_function(output_blob[object_index + 3]) * 2, 2) * anchor[anchor_index + c * 2 + 1];
				float max_prob = -1;
				int class_index = -1;

				// 解析类别
				for (d = 5; d < 85; ++d) {
					float prob = sigmoid_function(output_blob[object_index + d]);
					if (prob > max_prob) {
						max_prob = prob;
						class_index = d - 5;
					}
				}

				// 转换为top-left, bottom-right坐标
				int x1 = saturate_cast<int>((x - w / 2) * scale_x);	// top left x
				int y1 = saturate_cast<int>((y - h / 2) * scale_y);	// top left y
				int x2 = saturate_cast<int>((x + w / 2) * scale_x);	// bottom right x
				int y2 = saturate_cast<int>((y + h / 2) * scale_y);	// bottom right y

				// 解析输出
				classIds.push_back(class_index);
				confidences.push_back((float)conf);
				boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
			}
		}
		vector<int> indices;
		NMSBoxes(boxes, confidences, 0.25, 0.5, indices);
		for (i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			Rect box = boxes[idx];
			rectangle(src, box, Scalar(140, 190, 0), 4, 8, 0);
		}
		float fps = getTickFrequency() / (getTickCount() - start);
		float time = (getTickCount() - start) / getTickFrequency();

		ostringstream ss;
		ss << "FPS: " << fps << " detection time: " << time * 1000 << " ms";
		putText(src, ss.str(), Point(20, 50), 0, 1.0, Scalar(0, 0, 255), 2);

		imshow("OpenVINO2021R2+YOLOv5", src);
		waitKey(0);
	}

	return 0;
}