#include "RL/Policy.h"
#include <mutex>

static std::mutex mtx;

AgentInterface::AgentInterface(std::vector<PytorchEnvironmentPtr> env, ActorPtr actor, CriticPtr critic, StateModifierPtr state_modifier, double gamma, int steps, int batch_size)
	:env(env), actor(actor), critic(critic), state_modifier(state_modifier), gamma(gamma), steps(steps), batch_size(batch_size),
	zero_state(env[0]->reset()), max_score(0), episodes(0), total_tuple(0){
}

ReplayBuffer AgentInterface::get_replay_buffer(bool render){
	std::vector<ReplayBuffer> replay_buffer(env.size());
	std::vector<int> tuple(env.size(), 0), n(env.size(), 0);
	std::vector<double> total_score(env.size(), 0.);

#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < env.size(); i++){
		mtx.lock();
		torch::Tensor state = state_modifier->apply(env[i]->reset());
		mtx.unlock();
		while(tuple[i] * env.size() < steps){
			if(++n[i] == 1 && i == 0) std::cout << "0 state value : " << critic->get_values(state_modifier->modify(zero_state)).item().toDouble() << "\n";
			double score = 0;
			while(1){
				if(render && i == 0) env[i]->render();
				torch::Tensor next_state; double reward; int done, tl;
				torch::Tensor action = actor->get_action(state);
				env[i]->step(action, next_state, reward, done, tl);
				
				mtx.lock();
				next_state = state_modifier->apply(next_state); //TODO
				mtx.unlock();

				if(tl) reward += critic->get_values(next_state).item().toDouble() * gamma;
				replay_buffer[i].append(state.clone(), action.clone(), reward, done);
				state = next_state;

				score += reward; tuple[i] += 1;
				if(done) break;
			}
			total_score[i] += score;
		}
	}
	int tn = 0, tp = 0; double tc = 0;
	for(int i = 0; i < env.size(); i++) tn += n[i], tp += tuple[i], tc += total_score[i];
	total_tuple += tp; episodes += tn;
	std::cout << "episodes : " << episodes << "(" << total_tuple << "), score : " << tc / tn << ", avg steps : " << tp / (double)tn \
		<< ", avg reward : " << tc / tp << "\n";
	for(int i = 1; i < env.size(); i++) replay_buffer[0].merge(replay_buffer[i]);
	return replay_buffer[0];
}

torch::Tensor AgentInterface::next_action(const torch::Tensor &state){
	return actor->get_action(state);
}

torch::Tensor AgentInterface::next_action_nodist(const torch::Tensor &state){
	return actor->get_action_nodist(state_modifier->modify(state));
}

void AgentInterface::to(torch::Device dev){
	for(auto e : env) e->to(dev);
	actor->to(dev);
	critic->to(dev);
	state_modifier->to(dev);
	zero_state.to(dev);
}

// Every tensor object must be in CPU when set_xml called.
void AgentInterface::set_xml(tinyxml2::XMLElement *xml){
	tinyxml2::XMLElement* cur = xml->FirstChildElement("Policy");
	actor->set_xml(cur);
	critic->set_xml(cur);
	state_modifier->set_xml(cur);
	stringToData(cur->Attribute("gamma"), gamma);
	stringToData(cur->Attribute("steps"), steps);
	stringToData(cur->Attribute("batch_size"), batch_size);
	stringToData(cur->Attribute("episodes"), episodes);
	stringToData(cur->Attribute("total_tuple"), total_tuple);
}

tinyxml2::XMLElement* AgentInterface::get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc){
	tinyxml2::XMLElement* out = doc.NewElement("Policy");
	out->LinkEndChild(actor->get_xml(prefix, doc));
	out->LinkEndChild(critic->get_xml(prefix, doc));
	out->LinkEndChild(state_modifier->get_xml(prefix, doc));
	out->SetAttribute("gamma", gamma);
	out->SetAttribute("steps", steps);
	out->SetAttribute("batch_size", batch_size);
	out->SetAttribute("episodes", episodes);
	out->SetAttribute("total_tuple", total_tuple);
	return out;
}

VanilaAgent::VanilaAgent(std::vector<PytorchEnvironmentPtr> env, ActorPtr actor, CriticPtr critic, StateModifierPtr state_modifier, double gamma, int steps, int batch_size)
	: AgentInterface(env, actor, critic, state_modifier, gamma, steps, batch_size){
}

void VanilaAgent::train(int train_step, torch::Device device, bool render){
	for(int _ = 0; _ < train_step; _++){
		torch::Tensor states, actions, returns; 
		// no gradient
		{
			torch::NoGradGuard guard;
			actor->eval(); critic->eval();
			ReplayBuffer replay_buffer = get_replay_buffer(render);
			std::tie(states, actions) = replay_buffer.get_tensor();
			returns = replay_buffer.get_returns(gamma).unsqueeze(1);
		}
		// with gradient
		{
			actor->train(); critic->train();
			torch::Tensor log_policy = actor->evaluate(states, actions).first;
			torch::Tensor loss = -(returns * log_policy).mean();
			
			actor->zero_grad(); critic->zero_grad();
			loss.backward();
			actor->step(); critic->step();
		}
	}
}

PPOAgent::PPOAgent(std::vector<PytorchEnvironmentPtr> env, ActorPtr actor, CriticPtr critic, StateModifierPtr state_modifier, double gamma, double lamda, int steps, int batch_size)
	: AgentInterface(env, actor, critic, state_modifier, gamma, steps, batch_size), lamda(lamda){
}

void PPOAgent::train(int train_step, torch::Device device, bool render){
	to(device);
	for(int i = 0; i < train_step; i++){
		episodes += 1;
		torch::Tensor states, actions, entropy, returns, advants;
		torch::Tensor old_values, old_policy;
		torch::nn::MSELoss criterion;
		// no gradient
		{
			torch::NoGradGuard guard;
			actor->eval(); critic->eval();
			ReplayBuffer replay_buffer = get_replay_buffer(gamma);
			std::tie(states, actions) = replay_buffer.get_tensor();

			std::tie(old_policy, entropy) = actor->evaluate(states, actions);
			old_values = critic->get_values(states).squeeze(1);

			std::tie(returns, advants) = replay_buffer.get_gae(old_values.to(torch::kCPU), gamma, lamda);
			returns = returns.to(device);
			advants = advants.to(device);
		}

		// with gradient
		{
			int n = states.size(0);
			actor->train();
			critic->train();
	
			torch::Tensor arr = torch::randperm(n, torch::TensorOptions(torch::kLong).device(device));
			for(int i = 0; i < n/batch_size; i++){
				torch::Tensor batch_index = arr.slice(0, batch_size * i, batch_size * (i+1));
				torch::Tensor states_samples = states.index(batch_index);
				torch::Tensor returns_samples = returns.index(batch_index);
				torch::Tensor advants_samples = advants.index(batch_index);
				torch::Tensor actions_samples = actions.index(batch_index);
				torch::Tensor oldvalues_samples = old_values.index(batch_index);
				torch::Tensor oldpolicy_samples = old_policy.index(batch_index);

				torch::Tensor new_policy, entropy, ratio, clipped_ratio, actor_loss;
				//std::cout << "states_samples : " << states_samples << "\n";
				//std::cout << "actions_samples : " << actions_samples << "\n";
				std::tie(new_policy, entropy) = actor->evaluate(states_samples, actions_samples);
				//std::cout << "new_policy : " << new_policy << "\n";
				//std::cout << "entropy : " << entropy << "\n";
				ratio = (new_policy - oldpolicy_samples).exp();
				//std::cout << "ratio : " << ratio << "\n";
				clipped_ratio = ratio.clamp(1.0 - 0.2, 1.0 + 0.2);
				//std::cout << "clipped_ratio : " << clipped_ratio << "\n";
				//std::cout << "advants_samples : " << advants_samples << "\n";
				actor_loss = -torch::min(ratio * advants_samples, clipped_ratio * advants_samples).mean();

				torch::Tensor values, critic_loss;
				values = critic->get_values(states_samples);
				//clipped_values = oldvalues_samples + torch.clamp(values - oldvalues_samples, -0.2, 0.2) # clip param
				//critic_loss1 = criterion(clipped_values, returns_samples)
				critic_loss/*2*/ = criterion(values, returns_samples) * 0.5;
				//critic_loss = torch.max(critic_loss1, critic_loss2).mean()
				//std::cout << "actor_loss : " << actor_loss << "\n" << "critic_loss : " << critic_loss << "\n";

				torch::Tensor loss = actor_loss + critic_loss; // - 0.01*entropy.mean()

				actor->zero_grad(); critic->zero_grad();
				loss.backward();
				actor->step(); critic->step();
			}
			actor->eval(); critic->eval();
		}
	}
}

void PPOAgent::set_xml(tinyxml2::XMLElement *xml){
	AgentInterface::set_xml(xml);
	tinyxml2::XMLElement* cur = xml->FirstChildElement("Policy");
	stringToData(cur->Attribute("lamda"), lamda);
}

tinyxml2::XMLElement* PPOAgent::get_xml(const std::string &prefix, tinyxml2::XMLDocument &doc){
	tinyxml2::XMLElement* out = AgentInterface::get_xml(prefix, doc);
	out->SetAttribute("lamda", lamda);
	out->SetAttribute("Type", "PPO");
	return out;
}
