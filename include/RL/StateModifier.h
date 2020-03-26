#ifndef STATEMODIFIER
#define STATEMODIFIER

#include <torch/torch.h>
#include "RL/CheckPoint.h"

class StateModifier : public CheckPoint{
	public:
	virtual torch::Tensor apply(const torch::Tensor &state);
	virtual torch::Tensor modify(const torch::Tensor &state);

	virtual void to(torch::Device dev);
	virtual void set_xml(TiXmlElement *xml);
	virtual TiXmlElement* get_xml(const std::string &prefix);
};

class ClassicModifier : public StateModifier{
	public:
	ClassicModifier(int observation_size);
	
	torch::Tensor apply(const torch::Tensor &state);
	torch::Tensor modify(const torch::Tensor &state);
	int n;
	torch::Tensor mean, std;
	
	void to(torch::Device dev);
	void set_xml(TiXmlElement *xml);
	TiXmlElement* get_xml(const std::string &prefix);
};

using StateModifierPtr = std::shared_ptr<StateModifier>;

#endif