#include "RL/Distribution.h"

const double PI = acos(-1);

void DistributionInterface::train(){}
void DistributionInterface::eval(){}
void DistributionInterface::zero_grad(){}
void DistributionInterface::step(){}
void DistributionInterface::to(torch::Device dev){}

GaussianDistribution::GaussianDistribution(torch::Tensor scale):scale(scale), logScale(scale.log()){	
}

torch::Tensor GaussianDistribution::get_scale(const torch::Tensor &state){
	return scale;
}

torch::Tensor GaussianDistribution::sample(const torch::Tensor &mu, const torch::Tensor &state){
	return at::normal(mu, scale);
}

torch::Tensor GaussianDistribution::log_prob(const torch::Tensor &act, const torch::Tensor &mu, const torch::Tensor &state){
	return -(act - mu).mul(act - mu).div(2 * scale) - logScale - log(2*PI) / 2;
}

torch::Tensor GaussianDistribution::entropy(const torch::Tensor &mu, const torch::Tensor &state){
	return 0.5 + 0.5 * log(2*PI) + logScale;
}

void GaussianDistribution::to(torch::Device dev){
	scale = scale.to(dev);
	logScale = logScale.to(dev);
}

void GaussianDistribution::set_xml(tinyxml2::XMLElement *xml){
	auto cur = xml->FirstChildElement("GaussianDistribution");
	stringToTorch(cur->Attribute("scale"), scale);
	logScale = scale.log();
}

tinyxml2::XMLElement* GaussianDistribution::get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc){
	tinyxml2::XMLElement* out = doc.NewElement("GaussianDistribution");
	out->SetAttribute("scale", torchToString(scale).c_str());
	return out;
}