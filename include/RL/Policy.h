#ifndef POLICY
#define POLICY

#include "RL/ReplayBuffer.h"
#include "RL/Model.h"
#include "Environment/GymEnvironment.h"
#include "RL/StateModifier.h"

class AgentInterface : public CheckPoint{
	public:
	AgentInterface(GymEnvironmentPtr env, ActorPtr actor, CriticPtr critic, StateModifierPtr state_modifier, \
		double gamma, int steps, int batch_size);

	GymEnvironmentPtr env;
	ActorPtr actor;
	CriticPtr critic;
	StateModifierPtr state_modifier;
	double gamma;
	int steps, batch_size;

	torch::Tensor zero_state;
	double max_score;
	int episodes, total_tuple;

	ReplayBuffer get_replay_buffer(bool render = false);
	torch::Tensor next_action(const torch::Tensor &state);
	torch::Tensor next_action_nodist(const torch::Tensor &state);
	virtual void train(int train_step, torch::Device device, bool render = false) = 0;

	void to(torch::Device dev);
	virtual void set_xml(tinyxml2::XMLElement *xml);
	virtual tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc);
};

class VanilaAgent : public AgentInterface{
	public:
	VanilaAgent(GymEnvironmentPtr env, ActorPtr actor, CriticPtr critic, StateModifierPtr state_modifier, \
		double gamma, int steps, int batch_size);

	virtual void train(int train_step, torch::Device device, bool render = false);
};

class PPOAgent : public AgentInterface{
	public:
	PPOAgent(GymEnvironmentPtr env, ActorPtr actor, CriticPtr critic, StateModifierPtr state_modifier, \
		double gamma, double lamda, int steps, int batch_size);
	double lamda;

	virtual void train(int train_step, torch::Device device, bool render = false);
	
	virtual void set_xml(tinyxml2::XMLElement *xml);
	virtual tinyxml2::XMLElement* get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc);
};

using AgentInterfacePtr = std::shared_ptr<AgentInterface>;

#endif