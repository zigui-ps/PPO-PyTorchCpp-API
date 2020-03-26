#ifndef REPLAYBUFFER
#define REPLAYBUFFER

#include <torch/torch.h>
#include<vector>

class ReplayBuffer{
	public:
	std::vector<torch::Tensor> state, action;
	std::vector<double> reward;
	std::vector<int> done;

	void append(const torch::Tensor &s, const torch::Tensor &a, double r, int d);
	void merge(const ReplayBuffer &replay_buffer);
	std::pair<torch::Tensor, torch::Tensor> get_tensor();
	torch::Tensor get_returns(double gamma);
	std::pair<torch::Tensor, torch::Tensor> get_gae(const torch::Tensor &values_tensor, double gamma, double lamda);
};

#endif