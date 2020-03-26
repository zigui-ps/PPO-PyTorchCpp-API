#include "Environment/PyWrapper.h"

torch::Tensor PySequenceToTensor(PyObject* obj, bool decref){
	if(!obj) PyErr_Print();
	int size = PySequence_Size(obj);
	torch::Tensor tensor = torch::zeros(size);
	auto tensor_it = tensor.accessor<float, 1>();
	for(int i = 0; i < size; i++) tensor_it[i] = PyFloat_AsDouble(PySequence_GetItem(obj, i));
	if(decref) Py_DECREF(obj);
	return tensor;
}

PyWrapper::PyWrapper(PyObject* obj):obj(obj){ if(!obj) PyErr_Print(); }
PyWrapper::~PyWrapper(){ Py_DECREF(obj); }

PyArray::PyArray(torch::Tensor tensor) : PyWrapper(PyList_New(tensor.size(0))){
	auto tensor_it = tensor.accessor<float, 1>();
	for(int i = 0; i < tensor.size(0); i++) PyList_SET_ITEM(obj, i, PyFloat_FromDouble(tensor_it[i]));
}

PyString::PyString(const char* str) : PyWrapper(PyUnicode_DecodeASCII(str, strlen(str), NULL)){}
