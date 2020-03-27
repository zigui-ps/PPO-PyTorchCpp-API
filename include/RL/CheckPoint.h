#ifndef CHECKPOINT
#define CHECKPOINT

#include<vector>
#include<tinyxml2.h>
#include<torch/torch.h>

template<typename T> void stringToData(const char* str, std::vector<T> &var);
template<typename T> void stringToData(const char* str, T &var);
void stringToTorch(const char* str, torch::Tensor &tensor);
std::string torchToString(torch::Tensor &tensor);

class CheckPoint{
public:
	virtual void set_xml(tinyxml2::XMLElement *xml) = 0;
	virtual tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc) = 0;
};

#endif