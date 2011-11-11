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

#define PyDistArray_Init_NUM 0
#define PyDistArray_Init_RETURN int
#define PyDistArray_Init_PROTO (void)

#define PyDistArray_Exit_NUM 1
#define PyDistArray_Exit_RETURN void
#define PyDistArray_Exit_PROTO (void)

#define PyDistArray_NewBaseArray_NUM 2
#define PyDistArray_NewBaseArray_RETURN int
#define PyDistArray_NewBaseArray_PROTO (PyArrayObject *ary)

#define PyDistArray_DelViewArray_NUM 3
#define PyDistArray_DelViewArray_RETURN int
#define PyDistArray_DelViewArray_PROTO (PyArrayObject *array)

#define PyDistArray_HandleArray_NUM 4
#define PyDistArray_HandleArray_RETURN int
#define PyDistArray_HandleArray_PROTO (PyArrayObject *array, int transfer_data)

#define PyDistArray_MallocArray_NUM 5
#define PyDistArray_MallocArray_RETURN int
#define PyDistArray_MallocArray_PROTO (PyArrayObject *ary)

/*
#define PyDistArray_NewViewArray_NUM 10
#define PyDistArray_NewViewArray_RETURN int
#define PyDistArray_NewViewArray_PROTO (PyArrayObject *orig_ary, PyArrayObject *new_ary, int nslice, dndslice slice[NPY_MAXDIMS])
*/

/* Total number of C API pointers */
#define cphVB_API_pointers 6


#ifdef CPHVBNUMPY_MODULE
/* This section is used when compiling distnumpymodule.c */

static PyDistArray_Init_RETURN PyDistArray_Init PyDistArray_Init_PROTO;
static PyDistArray_Exit_RETURN PyDistArray_Exit PyDistArray_Exit_PROTO;
static PyDistArray_NewBaseArray_RETURN PyDistArray_NewBaseArray PyDistArray_NewBaseArray_PROTO;
static PyDistArray_DelViewArray_RETURN PyDistArray_DelViewArray PyDistArray_DelViewArray_PROTO;
static PyDistArray_HandleArray_RETURN PyDistArray_HandleArray PyDistArray_HandleArray_PROTO;
static PyDistArray_MallocArray_RETURN PyDistArray_MallocArray PyDistArray_MallocArray_PROTO;

/*
static PyDistArray_NewViewArray_RETURN PyDistArray_NewViewArray PyDistArray_NewViewArray_PROTO;
*/


#else
/* This section is used in modules that use distnumpy's API */

static void **cphVB_API;

#define PyDistArray_Init \
 (*(PyDistArray_Init_RETURN (*)PyDistArray_Init_PROTO) cphVB_API[PyDistArray_Init_NUM])

#define PyDistArray_Exit \
 (*(PyDistArray_Exit_RETURN (*)PyDistArray_Exit_PROTO) cphVB_API[PyDistArray_Exit_NUM])

#define PyDistArray_NewBaseArray \
 (*(PyDistArray_NewBaseArray_RETURN (*)PyDistArray_NewBaseArray_PROTO) cphVB_API[PyDistArray_NewBaseArray_NUM])

#define PyDistArray_DelViewArray \
 (*(PyDistArray_DelViewArray_RETURN (*)PyDistArray_DelViewArray_PROTO) cphVB_API[PyDistArray_DelViewArray_NUM])

#define PyDistArray_HandleArray \
 (*(PyDistArray_HandleArray_RETURN (*)PyDistArray_HandleArray_PROTO) cphVB_API[PyDistArray_HandleArray_NUM])

#define PyDistArray_MallocArray \
 (*(PyDistArray_MallocArray_RETURN (*)PyDistArray_MallocArray_PROTO) cphVB_API[PyDistArray_MallocArray_NUM])

/*
#define PyDistArray_NewViewArray \
 (*(PyDistArray_NewViewArray_RETURN (*)PyDistArray_NewViewArray_PROTO) cphVB_API[PyDistArray_NewViewArray_NUM])
*/

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
