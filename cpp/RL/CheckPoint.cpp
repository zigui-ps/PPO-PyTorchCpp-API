#include "RL/CheckPoint.h"

template<typename T>
void stringToData(const char* str, std::vector<T> &var){
	std::stringstream s(str);
	std::vector<T> res; T tmp;
	var.clear(); while(s >> tmp) var.push_back(tmp);
}

template<typename T>
void stringToData(const char* str, T &var){
	std::stringstream s(str);
	s >> var;
}

void stringToTorch(const char* str, torch::Tensor &tensor){
	std::vector<float> res; stringToData(str, res);
	tensor = torch::empty(res.size());
	auto tensor_data = tensor.accessor<float,1>();
	for(int i = 0; i < res.size(); i++) tensor_data[i] = res[i];
}

std::string torchToString(torch::Tensor &tensor_in){
	torch::Tensor tensor = tensor_in.to(torch::kCPU);
	int size = tensor.size(0);
	auto tensor_data = tensor.accessor<float,1>();
	std::stringstream str;
	for(int i = 0; i < size; i++){
		str << tensor_data[i];
		if(i != size-1) str << " ";
	}
	return str.str();
}

template void stringToData(const char* str, std::vector<int> &var);
template void stringToData(const char* str, std::vector<float> &var);
template void stringToData(const char* str, std::vector<double> &var);
template void stringToData(const char* str, int &var);
template void stringToData(const char* str, float &var);
template void stringToData(const char* str, double &var);