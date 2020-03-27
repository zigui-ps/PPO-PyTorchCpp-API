#ifndef DISTRIBUTION
#define DISTRIBUTION

#include <torch/torch.h>
#include<memory>
#include<tinyxml2.h>
#include "RL/CheckPoint.h"

class DistributionInterface : public CheckPoint{
	public:
	virtual torch::Tensor get_scale(const torch::Tensor &state) = 0;
	virtual torch::Tensor sample(const torch::Tensor &mu, const torch::Tensor &state) = 0;
	virtual torch::Tensor log_prob(const torch::Tensor &act, const torch::Tensor &mu, const torch::Tensor &state) = 0;
	virtual torch::Tensor entropy(const torch::Tensor &mu, const torch::Tensor &state) = 0;
	
	virtual void train();
	virtual void eval();
	virtual void zero_grad();
	virtual void step();
	virtual void to(torch::Device dev);
};

class GaussianDistribution : public DistributionInterface{
	public:
	GaussianDistribution(torch::Tensor scale);
	torch::Tensor scale, logScale;
	torch::Tensor get_scale(const torch::Tensor &state);
	torch::Tensor sample(const torch::Tensor &mu, const torch::Tensor &state);
	torch::Tensor log_prob(const torch::Tensor &act, const torch::Tensor &mu, const torch::Tensor &state);
	torch::Tensor entropy(const torch::Tensor &mu, const torch::Tensor &state);
	
	void to(torch::Device dev);
	void set_xml(tinyxml2::XMLElement *xml);
	tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc);
};

using DistributionInterfacePtr = std::shared_ptr<DistributionInterface>;

#endif
