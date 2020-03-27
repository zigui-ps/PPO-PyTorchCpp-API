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

void Actor::set_xml(tinyxml2::XMLElement *xml){
	tinyxml2::XMLElement* cur = xml->FirstChildElement("Actor");
	std::string net = cur->Attribute("network");
	torch::load(network, net);
	dist->set_xml(cur);
}

tinyxml2::XMLElement* Actor::get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc){
	tinyxml2::XMLElement* out = doc.NewElement("Actor");
	std::string net = prefix + "_actor_network";
	out->SetAttribute("network", net.c_str()); torch::save(network, net);
	out->LinkEndChild(dist->get_xml(prefix, doc));
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

void Critic::set_xml(tinyxml2::XMLElement *xml){
	tinyxml2::XMLElement* cur = xml->FirstChildElement("Critic");
	std::string net = cur->Attribute("network");
	torch::load(network, net);
}

tinyxml2::XMLElement* Critic::get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc){
	tinyxml2::XMLElement* out = doc.NewElement("Critic");
	std::string net = prefix + "_critic_network";
	out->SetAttribute("network", net.c_str()); torch::save(network, net);
	return out;
}