/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of DistNumPy <https://github.com/distnumpy>.
 *
 * DistNumPy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DistNumPy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DistNumPy. If not, see <http://www.gnu.org/licenses/>.
 */

#include <Python.h>
#define CPHVBNUMPY_MODULE
#include "cphvbnumpy.h"
//Tells numpy that this file initiate the module.
#define PY_ARRAY_UNIQUE_SYMBOL CPHVB_ARRAY_API
#include "numpy/arrayobject.h"

//We include all .h and .c files.
//NumPy distutil complains when having multiple module files.
#include "arrayobject.h"
#include "vem_interface.h"
#include "arraydata.h"
#include "batch.h"
#include "arrayobject.c"
#include "vem_interface.c"
#include "arraydata.c"
#include "batch.c"

/*
 * ===================================================================
 * Initialization of distnumpy.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_Init(void)
{
    //Initiate the VEM interface.
    vem_if_init();

    //Init the Array Data Protection.
    arydat_init();

    return 0;
} /* PyDistArray_Init */

/*
 * ===================================================================
 * De-initialization of distnumpy.
 */
static void
PyDistArray_Exit(void)
{
    batch_flush();

    //De-allocate the memory pool.
//    mem_pool_finalize();

    //Finalize the Array Data Protection.
    arydat_finalize();

    //Finalize the VEM interface.
    vem_if_finalize();

} /* PyDistArray_Exit */

/*
 * ===================================================================
 * Python wrapper for batch_flush.
 */
static PyObject *
_batch_flush(PyObject *m, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    batch_flush();
    Py_RETURN_NONE;
} /* _batch_flush */

/*
 * ===================================================================
 * Python wrapper for PyDistArray_HandleArray.
 */
static PyObject *
_handle_array(PyObject *m, PyObject *args)
{
    PyObject *obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if(!PyArray_Check(obj))
    {
        PyErr_SetString(PyExc_RuntimeError, "Must be a NumPy array.");
        return NULL;
    }

    if(PyDistArray_HandleArray((PyArrayObject *) obj, 1) != 0)
        return NULL;

    Py_RETURN_NONE;
} /* _handle_array */


static PyMethodDef cphVBMethods[] = {
    {"flush", _batch_flush, METH_VARARGS,
     "Executes all appending operations."},
    {"handle_array", _handle_array, METH_VARARGS,
     "Indicate that cphVB should handle the array."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initcphvbnumpy(void)
{
    PyObject *m;
    static void *cphVB_API[cphVB_API_pointers];
    PyObject *c_api_object;

    m = Py_InitModule("cphvbnumpy", cphVBMethods);
    if (m == NULL)
        return;

    /* Initialize the C API pointer array */
    cphVB_API[PyDistArray_Init_NUM] = (void *)PyDistArray_Init;
    cphVB_API[PyDistArray_Exit_NUM] = (void *)PyDistArray_Exit;
    cphVB_API[PyDistArray_NewBaseArray_NUM] = (void *)PyDistArray_NewBaseArray;
    cphVB_API[PyDistArray_DelViewArray_NUM] = (void *)PyDistArray_DelViewArray;
    cphVB_API[PyDistArray_HandleArray_NUM] = (void *)PyDistArray_HandleArray;
    cphVB_API[PyDistArray_MallocArray_NUM] = (void *)PyDistArray_MallocArray;
    cphVB_API[PyDistArray_MfreeArray_NUM] = (void *)PyDistArray_MfreeArray;
    cphVB_API[PyDistArray_NewViewArray_NUM] = (void *)PyDistArray_NewViewArray;

    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)cphVB_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

    // Import NumPy
    import_array();

    // Init DistNumPy
    PyDistArray_Init();
}
