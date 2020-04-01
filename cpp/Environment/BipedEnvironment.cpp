#include "Environment/BipedEnvironment.h"

template<typename T, typename U>
torch::Tensor eigenToTensor(const Eigen::Matrix<U, -1, 1> &vec, c10::ScalarType type){
	torch::Tensor tensor = torch::zeros(vec.size(), type);
	auto tensor_it = tensor.accessor<T, 1>();
	for(int i = 0; i < vec.size(); i++) tensor_it[i] = vec[i];
	return tensor;
}

template<typename T, typename U>
Eigen::Matrix<T, -1, 1> tensorToEigen(const torch::Tensor &tensor){
	auto tensor_it = tensor.accessor<U, 1>();
	Eigen::Matrix<T, -1, 1> vec = Eigen::Matrix<T, -1, 1>::Zero(tensor.size(0));
	for(int i = 0; i < tensor.size(0); i++) vec[i] = tensor_it[i];
	return vec;
}

template<typename T>
BipedEnvironment<T>::BipedEnvironment(torch::Device device, bool render):
	PytorchEnvironment(device){
	env = BipedEnv::GeneralEnvironmentInit<T>(render);
	actionSize = BipedEnv::getActionSize(env);
	observationSize = BipedEnv::getObservationSize(env);
}

template<typename T>
torch::Tensor BipedEnvironment<T>::reset(){
	auto res = eigenToTensor<float, double>(BipedEnv::reset(env), torch::kFloat32).to(device);
	return res;
}

template<typename T>
void BipedEnvironment<T>::step(const torch::Tensor &action, torch::Tensor &next_state, double &reward, int &done, int &tl){
	Eigen::VectorXd state;
	BipedEnv::step(env, tensorToEigen<double, float>(action.to(torch::kCPU)), state, reward, done, tl);
	next_state = eigenToTensor<float, double>(state, torch::kFloat32).to(device);
}

template<typename T>
void BipedEnvironment<T>::render(){
	BipedEnv::globalRender();
}

template class BipedEnvironment<PhaseShiftDeepMimic>;
