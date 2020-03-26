#include "RL/ReplayBuffer.h"
	
void ReplayBuffer::append(const torch::Tensor &s, const torch::Tensor &a, double r, int d){
	state.push_back(s);
	action.push_back(a);
	reward.push_back(r);
	done.push_back(d);
}

void ReplayBuffer::merge(const ReplayBuffer &replay_buffer){
	state.insert(state.end(), replay_buffer.state.begin(), replay_buffer.state.end());
	action.insert(action.end(), replay_buffer.action.begin(), replay_buffer.action.end());
	reward.insert(reward.end(), replay_buffer.reward.begin(), replay_buffer.reward.end());
	done.insert(done.end(), replay_buffer.done.begin(), replay_buffer.done.end());
}

std::pair<torch::Tensor, torch::Tensor> ReplayBuffer::get_tensor(){
	at::TensorList s(state);
	at::TensorList a(action);
	return std::make_pair(torch::stack(s), torch::stack(a));
}

torch::Tensor ReplayBuffer::get_returns(double gamma){
	int size = state.size();
	torch::Tensor returns_tensor = torch::empty(size);
	auto returns = returns_tensor.accessor<float,1>();

	double current_value = 0;
	for(int i = size-1; i >= 0; i--){
		double prev_value = done[i]? current_value : 0.;
		current_value = reward[i] + gamma * prev_value;
		returns[i] = current_value;
	}
	return returns_tensor;
}

std::pair<torch::Tensor, torch::Tensor> ReplayBuffer::get_gae(const torch::Tensor &values_tensor, double gamma, double lamda){
	double current_return = 0., current_value = 0., current_advant = 0.;
	int size = state.size();

	torch::Tensor returns_tensor = torch::empty(size);
	torch::Tensor advants_tensor = torch::empty(size);
	auto returns = returns_tensor.accessor<float,1>();
	auto advants = advants_tensor.accessor<float,1>();
	auto values = values_tensor.accessor<float,1>();
	double mse = 0;

	for (int i = size - 1; i >= 0; i--){
		double prev_return = done[i] ? 0. : current_return;
		double prev_advant = done[i] ? 0. : current_advant;
		double prev_value = done[i] ? 0. : current_value;

		current_return = reward[i] + gamma * prev_return;
		current_advant = prev_advant * gamma * lamda + (reward[i] - values[i] + gamma * prev_value);
		current_value = values[i];

		returns[i] = current_return;
		advants[i] = current_advant;
		mse += current_advant * current_advant;
	}
	advants_tensor = advants_tensor.div(sqrt(mse / size));
	return std::make_pair(returns_tensor.unsqueeze(1), advants_tensor.unsqueeze(1));
}