#ifndef TORCHENVIRONMENT
#define TORCHENVIRONMENT

#include <memory>
#include <torch/torch.h>

class PytorchEnvironment{
public:
	PytorchEnvironment(torch::Device device);
	torch::Device device;
	int observationSize, actionSize;

	virtual torch::Tensor reset() = 0;
	virtual void step(const torch::Tensor &action, torch::Tensor &next_state, double &reward, int &done, int &tl) = 0;
	virtual void render() = 0;
	virtual void to(torch::Device device);

	virtual int getObservationSize();
	virtual int getActionSize();
};

using PytorchEnvironmentPtr = std::shared_ptr<PytorchEnvironment>;

#endif
