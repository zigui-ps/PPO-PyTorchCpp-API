#include "Environment/PytorchEnvironment.h"

PytorchEnvironment::PytorchEnvironment(torch::Device device):device(device){
}

void PytorchEnvironment::to(torch::Device dev){
	device = dev;
}

int PytorchEnvironment::getObservationSize(){
	return observationSize;
}
	
int PytorchEnvironment::getActionSize(){
	return actionSize;
}
	