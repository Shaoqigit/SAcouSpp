#include <Python.h>
#include <numpy/arrayobject.h>

#include "Mesh.h"

bool PyArray_To_VectorArray(PyArrayObject* input, std::vector<std::array<double, 3>>& output) {
    // Check if the input is a floating point array with 2 dimensions and the second dimension is 3
    if (!PyArray_Check(input) ||
        PyArray_NDIM(input) != 2 ||
        PyArray_DIM(input, 1) != 3 ||
        !PyArray_ISFLOAT(input) ||
        !PyArray_ISCARRAY(input)) {
        PyErr_SetString(PyExc_ValueError, "Input should be a 2D float array with shape (-1, 3) and C-contiguous.");
        return false;
    }
    // Get the number of rows in the input array
    npy_intp num_rows = PyArray_DIM(input, 0);

    // Access the array data
    double* data = static_cast<double*>(PyArray_DATA(input));

    // Resize the output vector to hold all the rows
    output.resize(num_rows);

    // Copy data from the PyArrayObject to the std::vector<std::array<double, 3>>
    for (npy_intp i = 0; i < num_rows; ++i) {
        std::array<double, 3>& row = output[i];
        for (int j = 0; j < 3; ++j) {
            row[j] = data[i * 3 + j];  // Assuming row-major order
        }
    }

    return true;
}
typedef struct {
    PyObject_HEAD
    SAcouS::acxfem::MeshTriangle* mesh;
} PyMeshTriangleObject;


static int PyMeshTriangle_init(PyMeshTriangleObject* self, PyObject* args, PyObject* kwds) {
    // Parse Python arguments to C++ types
    // Create a MeshTriangle object and assign to self->mesh
    PyArrayObject *nodes_coordinate, *element_connectivity;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &nodes_coordinate, &PyArray_Type, &element_connectivity)) {
        return NULL;
    }
    self->mesh = new SAcouS::acxfem::MeshTriangle(reinterpret_cast<double*>(PyArray_DATA(nodes_coordinate)), PyArray_DIM(nodes_coordinate, 0), reinterpret_cast<int*>(PyArray_DATA(element_connectivity)), PyArray_DIM(element_connectivity, 0));
    return 0;
}

static void PyMeshTriangleObject_dealloc(PyMeshTriangleObject* self) {
    // Destructor logic here. Clean up the C++ object.
    delete self->mesh;
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyMeshTriangle_get_nodes(PyMeshTriangleObject* self, PyObject* args) {
    // Convert args to C++ types and call self->cpp_obj->set_nodes(...)
    // Return None or an appropriate value
    
    self->mesh->get_nodes();

}
// Function to create a Mesh1D object from Python
static PyObject* createMeshTrigular(PyObject* self, PyObject* args) {
    // Parse Python arguments to C++ types
    // Create a Mesh1D object and return as PyObject*
}


// Define methods of the module
static PyMethodDef MeshMethods[] = {
    {"createMesh1D", createMeshTrigular, METH_VARARGS, "Create a element mesh"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Define the module
static struct PyModuleDef meshmodule = {
    PyModuleDef_HEAD_INIT,
    "meshmodule",
    "Python interface for the Mesh C++ classes",
    -1,
    MeshMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_meshmodule(void) {
    return PyModule_Create(&meshmodule);
}