#ifndef PYWRAPPER
#define PYWRAPPER

#include<torch/torch.h>
#include <Python.h>
#include<string>

torch::Tensor PySequenceToTensor(PyObject* obj, bool decref = false);

class PyWrapper{
	public:
	PyObject* obj;
	PyWrapper(PyObject* obj);
	~PyWrapper();
};

class PyArray : public PyWrapper{
public:
	PyArray(torch::Tensor tensor);
};

class PyString : public PyWrapper{
public:
	PyString(const char* str);
};

#endif
