#ifndef MODEL
#define MODEL

#include <tinyxml2.h>
#include "RL/Deepnetwork.h"
#include "RL/Distribution.h"

class Actor{
public:
	Actor(DeepNetworkPtr network, AdamPtr opt, DistributionInterfacePtr dist);
	DeepNetworkPtr network;
	AdamPtr opt;
	DistributionInterfacePtr dist;

	torch::Tensor get_action(const torch::Tensor &state);	
	torch::Tensor get_action_nodist(const torch::Tensor &state);
	torch::Tensor get_std(const torch::Tensor &state);
	std::pair<torch::Tensor, torch::Tensor> evaluate(const torch::Tensor &state, const torch::Tensor &action);
	void train();
	void eval();
	void zero_grad();
	void step();

	void to(torch::Device device);
	virtual void set_xml(tinyxml2::XMLElement *xml);
	virtual tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc);
};

using ActorPtr = std::shared_ptr<Actor>;

class Critic{
public:
	Critic(DeepNetworkPtr network, AdamPtr opt);
	DeepNetworkPtr network;
	AdamPtr opt;

	torch::Tensor get_values(const torch::Tensor &state);	
	void train();
	void eval();
	void zero_grad();
	void step();

	void to(torch::Device device);
	virtual void set_xml(tinyxml2::XMLElement *xml);
	virtual tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc);
};

using CriticPtr = std::shared_ptr<Critic>;

#endif
