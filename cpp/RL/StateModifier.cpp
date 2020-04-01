#include "RL/StateModifier.h"

torch::Tensor StateModifier::apply(const torch::Tensor &state){
	return state;
}

torch::Tensor StateModifier::modify(const torch::Tensor &state){
	return state;
}

void StateModifier::to(torch::Device dev){
}

void StateModifier::set_xml(tinyxml2::XMLElement *xml){
}

tinyxml2::XMLElement* StateModifier::get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc){
	return doc.NewElement("StateModifier");
}

ClassicModifier::ClassicModifier(int observation_size) : n(0){
	mean = torch::zeros(observation_size);
	std = torch::zeros(observation_size);
}

torch::Tensor ClassicModifier::apply(const torch::Tensor &state){
	mtx.lock();
	n += 1;
	torch::Tensor prev_mean = mean.clone();
	mean = prev_mean + (state - prev_mean) / n;
	std = std + (state - prev_mean) * (state - mean);
	mtx.unlock();
	return modify(state);
}

torch::Tensor ClassicModifier::modify(const torch::Tensor &state){
	torch::Tensor ret;
	mtx.lock();
	if(n <= 1) ret = torch::zeros(state.size(0), state.device());
	else ret = (state - mean) / (std.div(n-1).sqrt().add(1e-8));
	mtx.unlock();
	return ret.clamp(-5, 5);
}

void ClassicModifier::to(torch::Device dev){
	mean = mean.to(dev);
	std = std.to(dev);
}

void ClassicModifier::set_xml(tinyxml2::XMLElement *xml){
	tinyxml2::XMLElement* cur = xml->FirstChildElement("ClassicModifier");
	stringToData(cur->Attribute("n"), n);
	stringToTorch(cur->Attribute("mean"), mean);
	stringToTorch(cur->Attribute("std"), std);
}

tinyxml2::XMLElement* ClassicModifier::get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc){
	tinyxml2::XMLElement* out = doc.NewElement("ClassicModifier");
	out->SetAttribute("n", n);
	out->SetAttribute("mean", torchToString(mean).c_str());
	out->SetAttribute("std", torchToString(std).c_str());
	return out;
}
