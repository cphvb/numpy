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
/*

#include "dependency_system.h"
#include "process_grid.h"
#include "arraydata.h"
*/

#include "arrayobject.c"
#include "vem_interface.c"
/*

#include "dependency_system.c"
#include "process_grid.c"
#include "arraydata.c"
*/

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
    //arydat_init();

    return 0;
} /* PyDistArray_Init */

/*
 * ===================================================================
 * De-initialization of distnumpy.
 */
static void
PyDistArray_Exit(void)
{
    //De-allocate the memory pool.
//    mem_pool_finalize();

    //Finalize the Array Data Protection.
//    arydat_finalize();

    //Finalize the VEM interface.
    vem_if_finalize();

} /* PyDistArray_Exit */


/*
 * ===================================================================
 * Executes all appending operations.
 * If the optional option barrier is true, a MPI barrier is included
 * in SPMD mode.
 */
/*
static PyObject *
evalflush(PyObject *m, PyObject *args, PyObject *kws)
{
    static char *kwd[]= {"barrier", NULL};
    PyObject *barrier = Py_False;

    if (!PyArg_ParseTupleAndKeywords(args, kws, "|O:bool", kwd, &barrier))
        return NULL;

    //The master should also do the operation.
    //dep_flush(1);

    Py_RETURN_NONE;
}*/ /* evalflush */


static PyMethodDef cphVBMethods[] = {/*
    {"evalflush", evalflush, METH_VARARGS|METH_KEYWORDS,
     "Executes all appending operations. If the optional option "\
     "barrier is true, a MPI barrier is included in SPMD mode"},*/
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
/*
    cphVB_API[PyDistArray_GetItem_NUM] = (void *)PyDistArray_GetItem;
    cphVB_API[PyDistArray_PutItem_NUM] = (void *)PyDistArray_PutItem;
    cphVB_API[PyDistArray_ProcGridSet_NUM] = (void *)PyDistArray_ProcGridSet;
    cphVB_API[PyDistArray_IsDist_NUM] = (void *)PyDistArray_IsDist;
    cphVB_API[PyDistArray_NewViewArray_NUM] = (void *)PyDistArray_NewViewArray;
*/
    /* Create a CObject containing the API pointer array's address */
    c_api_object = PyCObject_FromVoidPtr((void *)cphVB_API, NULL);

    if (c_api_object != NULL)
        PyModule_AddObject(m, "_C_API", c_api_object);

    // Import NumPy
    import_array();

    // Init DistNumPy
    PyDistArray_Init();
}
