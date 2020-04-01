#ifndef STATEMODIFIER
#define STATEMODIFIER

#include <torch/torch.h>
#include <mutex>
#include "RL/CheckPoint.h"

class StateModifier : public CheckPoint{
	public:
	virtual torch::Tensor apply(const torch::Tensor &state);
	virtual torch::Tensor modify(const torch::Tensor &state);

	virtual void to(torch::Device dev);
	virtual void set_xml(tinyxml2::XMLElement *xml);
	virtual tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc);
};

class ClassicModifier : public StateModifier{
	public:
	std::mutex mtx;

	ClassicModifier(int observation_size);
	
	virtual torch::Tensor apply(const torch::Tensor &state);
	virtual torch::Tensor modify(const torch::Tensor &state);
	int n;
	torch::Tensor mean, std;
	
	virtual void to(torch::Device dev);
	virtual void set_xml(tinyxml2::XMLElement *xml);
	virtual tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc);
};

using StateModifierPtr = std::shared_ptr<StateModifier>;

#endif
