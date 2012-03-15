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
#include "arraycollection.h"
#include "vem_interface.h"
#include "arraydata.h"
#include "batch.h"
#include "ufunc.h"
#include "reduce.h"
#include "copyinto.h"
#include "random.h"
#include "arrayobject.c"
#include "arraycollection.c"
#include "vem_interface.c"
#include "arraydata.c"
#include "batch.c"
#include "ufunc.c"
#include "reduce.c"
#include "copyinto.c"
#include "random.c"

/*
 * ===================================================================
 * Initialization of distnumpy.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_Init(void)
{
    //Initiate the VEM interface.
    vem_if_init();

    //Init the Array Data Protection.
    arydat_init();

    //Initiate the reduce function.
    reduce_init();

    //Initiate the random function.
    random_init();

    return 0;
} /* PyCphVB_Init */

/*
 * ===================================================================
 * De-initialization of distnumpy.
 */
static void
PyCphVB_Exit(void)
{
    batch_flush();

    //Finalize the Array Data Protection.
    arydat_finalize();

    //Finalize the VEM interface.
    vem_if_finalize();

} /* PyCphVB_Exit */

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
 * Python wrapper for PyCphVB_HandleArray.
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

    if(PyCphVB_HandleArray((PyArrayObject *) obj, 1) != 0)
        return NULL;

    Py_RETURN_NONE;
} /* _handle_array */

/*
 * ===================================================================
 * Python wrapper for PyCphVB_UnHandleArray.
 */
static PyObject *
_unhandle_array(PyObject *m, PyObject *args)
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

    if(PyCphVB_UnHandleArray((PyArrayObject *) obj) != 0)
        return NULL;

    Py_RETURN_NONE;
} /* _unhandle_array */

/*
 * ===================================================================
 * Python wrapper for PyCphVB_HandleArray.
 */
static PyObject *
_fill_random(PyObject *m, PyObject *args)
{
    PyObject *obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }
    if(!PyArray_CheckExact(obj))
    {
        PyErr_SetString(PyExc_RuntimeError, "Must be a NumPy array.");
        return NULL;
    }

    if(PyCphVB_Random((PyArrayObject *) obj) != 0)
        return NULL;

    Py_RETURN_NONE;
} /* _handle_array */



static PyMethodDef cphVBMethods[] = {
    {"flush", _batch_flush, METH_VARARGS,
     "Executes all appending operations."},
    {"handle_array", _handle_array, METH_VARARGS,
     "Indicate that cphVB should handle the array."},
    {"unhandle_array", _unhandle_array, METH_VARARGS,
     "Indicate that cphVB should NOT handle the array."},
    {"fill_random", _fill_random, METH_VARARGS,
     "Fill the empty array with numpy.random.random()."},
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
    cphVB_API[PyCphVB_Init_NUM] = (void *)PyCphVB_Init;
    cphVB_API[PyCphVB_Exit_NUM] = (void *)PyCphVB_Exit;
    cphVB_API[PyCphVB_NewBaseArray_NUM] = (void *)PyCphVB_NewBaseArray;
    cphVB_API[PyCphVB_DelViewArray_NUM] = (void *)PyCphVB_DelViewArray;
    cphVB_API[PyCphVB_HandleArray_NUM] = (void *)PyCphVB_HandleArray;
    cphVB_API[PyCphVB_UnHandleArray_NUM] = (void *)PyCphVB_UnHandleArray;
    cphVB_API[PyCphVB_MallocArray_NUM] = (void *)PyCphVB_MallocArray;
    cphVB_API[PyCphVB_MfreeArray_NUM] = (void *)PyCphVB_MfreeArray;
    cphVB_API[PyCphVB_NewViewArray_NUM] = (void *)PyCphVB_NewViewArray;
    cphVB_API[PyCphVB_Ufunc_NUM] = (void *)PyCphVB_Ufunc;
    cphVB_API[PyCphVB_BaseArray_NUM] = (void *)PyCphVB_BaseArray;
    cphVB_API[PyCphVB_Reduce_NUM] = (void *)PyCphVB_Reduce;
    cphVB_API[PyCphVB_CopyInto_NUM] = (void *)PyCphVB_CopyInto;
    cphVB_API[PyCphVB_Random_NUM] = (void *)PyCphVB_Random;

    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)cphVB_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

    // Import NumPy
    import_array();

    // Init DistNumPy
    PyCphVB_Init();

    // Run PyCphVB_Exit() on Python exit.
    Py_AtExit(PyCphVB_Exit);
}
