#include "RL/Deepnetwork.h"
	
DeepNetwork::DeepNetwork(const std::vector<int> &size, bool zeroInit){
	char name[4] = "fc1";
	for(int i = 0; i+1 < size.size(); i++, name[2]++){
		torch::nn::Linear f = register_module(name, torch::nn::Linear(size[i], size[i+1]));
		if(!zeroInit) torch::nn::init::xavier_uniform_(f.get()->weight);
		else torch::nn::init::constant_(f.get()->weight, 0.0);
		fc.push_back(f);
	}
}

torch::Tensor DeepNetwork::forward(torch::Tensor x){
	for(int i = 0; i+1 < fc.size(); i++) x = torch::relu(fc[i](x));
	return fc.back()(x);
}
