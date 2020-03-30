#ifndef GYMENVIRONMENT
#define GYMENVIRONMENT

#include <memory>
#include <torch/torch.h>
#include "Environment/PyWrapper.h"
#include "Environment/PytorchEnvironment.h"

class GymEnvironment : public PyWrapper, public PytorchEnvironment{
public:
	GymEnvironment(const char* name, torch::Device device);
	PyObject* pyReset, *pyStep, *pyRender;
	int steps;

	virtual torch::Tensor reset();
	virtual void step(const torch::Tensor &action, torch::Tensor &next_state, double &reward, int &done, int &tl);
	virtual void render();

	static PyObject* pModule, *pMake;
private:
	PyObject* init(const char* name);
};

using GymEnvironmentPtr = std::shared_ptr<GymEnvironment>;

#endif
