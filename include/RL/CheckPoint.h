#ifndef CHECKPOINT
#define CHECKPOINT

#include<vector>
#include<tinyxml.h>
#include<torch/torch.h>

template<typename T> void stringToData(const char* str, std::vector<T> &var);
template<typename T> void stringToData(const char* str, T &var);
void stringToTorch(const char* str, torch::Tensor &tensor);
std::string torchToString(torch::Tensor &tensor);

class CheckPoint{
public:
	virtual void set_xml(TiXmlElement *xml) = 0;
	virtual TiXmlElement* get_xml(const std::string &prefix) = 0;
};

#endif