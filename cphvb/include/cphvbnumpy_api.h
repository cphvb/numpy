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

#ifndef CPHVBNUMPY_API_H
#define CPHVBNUMPY_API_H
#ifdef __cplusplus
extern "C" {
#endif

/* C API functions */

#define PyCphVB_Init_NUM 0
#define PyCphVB_Init_RETURN int
#define PyCphVB_Init_PROTO (void)

#define PyCphVB_Exit_NUM 1
#define PyCphVB_Exit_RETURN void
#define PyCphVB_Exit_PROTO (void)

#define PyCphVB_NewBaseArray_NUM 2
#define PyCphVB_NewBaseArray_RETURN int
#define PyCphVB_NewBaseArray_PROTO (PyArrayObject *ary)

#define PyCphVB_DelViewArray_NUM 3
#define PyCphVB_DelViewArray_RETURN int
#define PyCphVB_DelViewArray_PROTO (PyArrayObject *array)

#define PyCphVB_HandleArray_NUM 4
#define PyCphVB_HandleArray_RETURN int
#define PyCphVB_HandleArray_PROTO (PyArrayObject *array, int transfer_data)

#define PyCphVB_MallocArray_NUM 5
#define PyCphVB_MallocArray_RETURN int
#define PyCphVB_MallocArray_PROTO (PyArrayObject *ary, cphvb_intp size)

#define PyCphVB_MfreeArray_NUM 6
#define PyCphVB_MfreeArray_RETURN int
#define PyCphVB_MfreeArray_PROTO (PyArrayObject *ary)

#define PyCphVB_NewViewArray_NUM 7
#define PyCphVB_NewViewArray_RETURN int
#define PyCphVB_NewViewArray_PROTO (PyArrayObject *ary)

#define PyCphVB_Ufunc_NUM 8
#define PyCphVB_Ufunc_RETURN int
#define PyCphVB_Ufunc_PROTO (PyUFuncObject *ufunc, PyArrayObject **op)

#define PyCphVB_BaseArray_NUM 9
#define PyCphVB_BaseArray_RETURN PyArrayObject *
#define PyCphVB_BaseArray_PROTO (PyArrayObject *array)

#define PyCphVB_Reduce_NUM 10
#define PyCphVB_Reduce_RETURN int
#define PyCphVB_Reduce_PROTO (PyUFuncObject *ufunc, PyArrayObject *in, PyArrayObject *out, int axis)

/* Total number of C API pointers */
#define cphVB_API_pointers 11


#ifdef CPHVBNUMPY_MODULE
/* This section is used when compiling distnumpymodule.c */

static PyCphVB_Init_RETURN         PyCphVB_Init         PyCphVB_Init_PROTO;
static PyCphVB_Exit_RETURN         PyCphVB_Exit         PyCphVB_Exit_PROTO;
static PyCphVB_NewBaseArray_RETURN PyCphVB_NewBaseArray PyCphVB_NewBaseArray_PROTO;
static PyCphVB_DelViewArray_RETURN PyCphVB_DelViewArray PyCphVB_DelViewArray_PROTO;
static PyCphVB_HandleArray_RETURN  PyCphVB_HandleArray  PyCphVB_HandleArray_PROTO;
static PyCphVB_MallocArray_RETURN  PyCphVB_MallocArray  PyCphVB_MallocArray_PROTO;
static PyCphVB_MfreeArray_RETURN   PyCphVB_MfreeArray   PyCphVB_MfreeArray_PROTO;
static PyCphVB_NewViewArray_RETURN PyCphVB_NewViewArray PyCphVB_NewViewArray_PROTO;
static PyCphVB_Ufunc_RETURN        PyCphVB_Ufunc        PyCphVB_Ufunc_PROTO;
static PyCphVB_BaseArray_RETURN    PyCphVB_BaseArray    PyCphVB_BaseArray_PROTO;
static PyCphVB_Reduce_RETURN       PyCphVB_Reduce   PyCphVB_Reduce_PROTO;


#else
/* This section is used in modules that use distnumpy's API */

static void **cphVB_API;

#define PyCphVB_Init \
 (*(PyCphVB_Init_RETURN (*)PyCphVB_Init_PROTO) cphVB_API[PyCphVB_Init_NUM])

#define PyCphVB_Exit \
 (*(PyCphVB_Exit_RETURN (*)PyCphVB_Exit_PROTO) cphVB_API[PyCphVB_Exit_NUM])

#define PyCphVB_NewBaseArray \
 (*(PyCphVB_NewBaseArray_RETURN (*)PyCphVB_NewBaseArray_PROTO) cphVB_API[PyCphVB_NewBaseArray_NUM])

#define PyCphVB_DelViewArray \
 (*(PyCphVB_DelViewArray_RETURN (*)PyCphVB_DelViewArray_PROTO) cphVB_API[PyCphVB_DelViewArray_NUM])

#define PyCphVB_HandleArray \
 (*(PyCphVB_HandleArray_RETURN (*)PyCphVB_HandleArray_PROTO) cphVB_API[PyCphVB_HandleArray_NUM])

#define PyCphVB_MallocArray \
 (*(PyCphVB_MallocArray_RETURN (*)PyCphVB_MallocArray_PROTO) cphVB_API[PyCphVB_MallocArray_NUM])

#define PyCphVB_MfreeArray \
 (*(PyCphVB_MfreeArray_RETURN (*)PyCphVB_MfreeArray_PROTO) cphVB_API[PyCphVB_MfreeArray_NUM])

#define PyCphVB_NewViewArray \
 (*(PyCphVB_NewViewArray_RETURN (*)PyCphVB_NewViewArray_PROTO) cphVB_API[PyCphVB_NewViewArray_NUM])

#define PyCphVB_Ufunc \
 (*(PyCphVB_Ufunc_RETURN (*)PyCphVB_Ufunc_PROTO) cphVB_API[PyCphVB_Ufunc_NUM])

#define PyCphVB_BaseArray \
 (*(PyCphVB_BaseArray_RETURN (*)PyCphVB_BaseArray_PROTO) cphVB_API[PyCphVB_BaseArray_NUM])

#define PyCphVB_Reduce \
 (*(PyCphVB_Reduce_RETURN (*)PyCphVB_Reduce_PROTO) cphVB_API[PyCphVB_Reduce_NUM])


/* Return -1 and set exception on error, 0 on success. */
static int
import_cphvbnumpy(void)
{
    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule("cphvbnumpy");
    if (module == NULL)
        return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");
    if (c_api_object == NULL) {
        Py_DECREF(module);
        return -1;
    }
    if (PyCObject_Check(c_api_object))
        cphVB_API = (void **)PyCObject_AsVoidPtr(c_api_object);

    Py_DECREF(c_api_object);
    Py_DECREF(module);
    return 0;
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* !defined(CPHVBNUMPY_API_H) */
