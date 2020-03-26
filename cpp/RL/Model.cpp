#include "RL/Model.h"

Actor::Actor(DeepNetworkPtr network, AdamPtr opt, DistributionInterfacePtr dist):
	network(network), opt(opt), dist(dist){
}

torch::Tensor Actor::get_action(const torch::Tensor &state){
	return dist->sample(network->forward(state), state);
}

torch::Tensor Actor::get_action_nodist(const torch::Tensor &state){
	return network->forward(state);
}

torch::Tensor Actor::get_std(const torch::Tensor &state){
	return dist->get_scale(state);
}

std::pair<torch::Tensor, torch::Tensor> Actor::evaluate(const torch::Tensor &state, const torch::Tensor &action){
	torch::Tensor mu = network->forward(state);
	return std::make_pair(dist->log_prob(action, mu, state), dist->entropy(mu, state));
}

void Actor::train(){
	network->train();
	dist->train();
}

void Actor::eval(){
	network->eval();
	dist->eval();
}

void Actor::zero_grad(){
	opt->zero_grad();
	dist->zero_grad();
}

void Actor::step(){
	opt->step();
	dist->step();
}

void Actor::to(torch::Device dev){
	network->to(dev);
	dist->to(dev);
}

void Actor::set_xml(TiXmlElement *xml){
	TiXmlElement* cur = xml->FirstChildElement("Actor");
	std::string net = cur->Attribute("network");
	torch::load(network, net);
	dist->set_xml(cur);
}

TiXmlElement* Actor::get_xml(const std::string &prefix){
	TiXmlElement* out = new TiXmlElement("Actor");
	std::string net = prefix + "_actor_network";
	out->SetAttribute((std::string)"network", net); torch::save(network, net);
	out->LinkEndChild(dist->get_xml(prefix));
	return out;
}

Critic::Critic(DeepNetworkPtr network, AdamPtr opt) : network(network), opt(opt){
}
	
torch::Tensor Critic::get_values(const torch::Tensor &state){
	return network->forward(state);
}

void Critic::train(){
	network->train();
}

void Critic::eval(){
	network->eval();
}

void Critic::zero_grad(){
	opt->zero_grad();
}

void Critic::step(){
	opt->step();
}

void Critic::to(torch::Device dev){
	network->to(dev);
}

void Critic::set_xml(TiXmlElement *xml){
	TiXmlElement* cur = xml->FirstChildElement("Critic");
	std::string net = cur->Attribute("network");
	torch::load(network, net);
}

TiXmlElement* Critic::get_xml(const std::string &prefix){
	TiXmlElement* out = new TiXmlElement("Critic");
	std::string net = prefix + "_critic_network";
	out->SetAttribute((std::string)"network", net); torch::save(network, net);
	return out;
}