#include <Eigen/Dense>
#include "api.h"
#include "Environment/PytorchEnvironment.h"

template<typename T>
class BipedEnvironment : public PytorchEnvironment{
	public:
	BipedEnvironment(torch::Device device, bool render);
	int env;

	torch::Tensor reset();
	void step(const torch::Tensor &action, torch::Tensor &next_state, double &reward, int &done, int &tl);
	void render();
};
