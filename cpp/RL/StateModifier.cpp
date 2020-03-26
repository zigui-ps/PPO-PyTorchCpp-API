#include "RL/StateModifier.h"

torch::Tensor StateModifier::apply(const torch::Tensor &state){
	return state;
}

torch::Tensor StateModifier::modify(const torch::Tensor &state){
	return state;
}

void StateModifier::to(torch::Device dev){
}

void StateModifier::set_xml(TiXmlElement *xml){
}

TiXmlElement* StateModifier::get_xml(const std::string &prefix){
	return new TiXmlElement("StateModifier");
}

ClassicModifier::ClassicModifier(int observation_size) : n(0){
	mean = torch::zeros(observation_size);
	std = torch::zeros(observation_size);
}

torch::Tensor ClassicModifier::apply(const torch::Tensor &state){
	n += 1;
	torch::Tensor prev_mean = mean.clone();
	mean = prev_mean + (state - prev_mean) / n;
	std = std + (state - prev_mean) * (state - mean);
	return modify(state);
}

torch::Tensor ClassicModifier::modify(const torch::Tensor &state){
	torch::Tensor norm;
	if(n == 0) return state;
	else if(n == 1) norm = torch::zeros(state.size(0), state.device());
	else norm = (state - mean) / (std.div(n-1).sqrt().add(1e-8));
	return norm.clamp(-5, 5);
}

void ClassicModifier::to(torch::Device dev){
	mean = mean.to(dev);
	std = std.to(dev);
}

void ClassicModifier::set_xml(TiXmlElement *xml){
	TiXmlElement* cur = xml->FirstChildElement("ClassicModifier");
	stringToData(cur->Attribute("n"), n);
	stringToTorch(cur->Attribute("mean"), mean);
	stringToTorch(cur->Attribute("std"), std);
}

TiXmlElement* ClassicModifier::get_xml(const std::string &prefix){
	TiXmlElement* out = new TiXmlElement("ClassicModifier");
	out->SetAttribute("n", n);
	out->SetAttribute((std::string)"mean", torchToString(mean));
	out->SetAttribute((std::string)"std", torchToString(std));
	return out;
}