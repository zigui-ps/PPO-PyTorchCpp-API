#ifndef DEEPNETWORK
#define DEEPNETWORK
#include <torch/torch.h>
#include <vector>
#include <memory>

struct DeepNetwork : torch::nn::Module{
	DeepNetwork(const std::vector<int> &size, bool zeroInit = false);
	torch::Tensor forward(torch::Tensor x);
	std::vector<torch::nn::Linear> fc;
};

using DeepNetworkPtr = std::shared_ptr<DeepNetwork>;
using AdamPtr = std::shared_ptr<torch::optim::Adam>;

#endif
