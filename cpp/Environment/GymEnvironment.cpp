#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "Environment/GymEnvironment.h"

PyObject* GymEnvironment::pModule;
PyObject* GymEnvironment::pMake;

static std::mutex mtx;

GymEnvironment::GymEnvironment(const char* name, torch::Device device) : 
		PytorchEnvironment(device), PyWrapper(pModule == NULL? init(name) : PyObject_CallFunctionObjArgs(pMake, PyString(name).obj, NULL)){
	pyReset = PyObject_GetAttrString(obj, "reset");
	pyStep = PyObject_GetAttrString(obj, "step");
	pyRender = PyObject_GetAttrString(obj, "render");
	// TODO
	auto ob_s = PyObject_GetAttrString(obj, "observation_space"), ac_s = PyObject_GetAttrString(obj, "action_space");
	PyErr_Print();
	observationSize = PyLong_AsLong(PyTuple_GetItem(PyObject_GetAttrString(ob_s, "shape"), 0));
	PyErr_Print();
	actionSize = PyLong_AsLong(PyTuple_GetItem(PyObject_GetAttrString(ac_s, "shape"), 0));
	PyErr_Print();
}

PyObject* GymEnvironment::init(const char* name){
	pModule = PyImport_Import(PyString("gym").obj);
	pMake = PyObject_GetAttrString(GymEnvironment::pModule, "make");
	return PyObject_CallFunctionObjArgs(pMake, PyString(name).obj, NULL);
}

torch::Tensor GymEnvironment::reset(){
	steps = 0;
	mtx.lock();
	torch::Tensor tmp = PySequenceToTensor(PyObject_CallFunctionObjArgs(pyReset, NULL), true).to(device);
	mtx.unlock();
	return tmp;
}

void GymEnvironment::step(const torch::Tensor &action, torch::Tensor &next_state, double &reward, int &done, int &tl){
	mtx.lock();
	PyObject* tuple = PyObject_CallFunctionObjArgs(pyStep, PyArray(action.to(torch::kCPU)).obj, NULL);
	if(tuple == NULL) PyErr_Print();
	next_state = PySequenceToTensor(PyTuple_GetItem(tuple, 0)).to(device);
	reward = PyFloat_AsDouble(PyTuple_GetItem(tuple, 1));
	done = (PyTuple_GetItem(tuple, 2) == Py_True);
	tl = steps >= 2000;
	Py_DECREF(tuple);
	mtx.unlock();
	if(done) next_state = reset();
}

void GymEnvironment::render(){
	mtx.lock();
	if(PyObject_CallFunctionObjArgs(pyRender, NULL) == NULL) PyErr_Print();
	mtx.unlock();
}
