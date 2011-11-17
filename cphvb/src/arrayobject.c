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

#include <errno.h>
#include <sys/mman.h>
#include <signal.h>
#include "types.h"

/*
 *===================================================================
 * Create a new base array and updates the PyArrayObject.
 * NB: The PyArrayObject must be behaved.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_NewBaseArray(PyArrayObject *ary)
{
    int i;
    cphvb_index s;
    cphvb_index stride[CPHVB_MAXDIM];
    cphvb_index shape[CPHVB_MAXDIM];
    cphvb_error err;
    cphvb_type dtype = type_py2cph[PyArray_TYPE(ary)];

    printf("PyCphVB_NewBaseArray - type: %s\n",cphvb_type_text(dtype));

    if(dtype == CPHVB_UNKNOWN)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "cphVB does not support the datatype\n");
        return -1;
    }

    //Append the array to the base array collection.
    ary->prev = NULL;
    ary->next = ary_root;
    ary_root = ary;
    if(ary->next != NULL)
    {
        assert(ary->next->prev == NULL);
        ary->next->prev = ary_root;
    }

    //cphVB is handling the array.
    ary->cphvb_handled = 1;

    //We handle scalars as 1-dim array with size 1.
    cphvb_intp ndims = PyArray_NDIM(ary);
    if(PyArray_IsZeroDim(ary))
    {
        ndims = 1;
        stride[0] = 1;
        shape[0] = 1;
    }
    else
    {
        //Compute the stride. Row-Major (C-style)
        s=1;
        for(i=PyArray_NDIM(ary)-1; i>=0; --i)
        {
            stride[i] = s;
            shape[i] = (cphvb_index) PyArray_DIM(ary,i);
            s *= shape[i];
        }
    }
    err = vem_create_array(NULL, dtype, ndims, 0, shape,
                           stride, 0, (cphvb_constant)0L,
                           &PyCphVB_ARRAY(ary));
    assert(PyCphVB_ARRAY(ary) != NULL);
    return err;
} /* PyCphVB_NewBaseArray */

/*
 *===================================================================
 * Create a new of a base array and updates the PyArrayObject.
 * NB: The PyArrayObject must be behaved.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_NewViewArray(PyArrayObject *ary)
{
    int i;
    cphvb_error err;
    cphvb_type dtype = type_py2cph[PyArray_TYPE(ary)];
    cphvb_intp offset;
    cphvb_intp strides[CPHVB_MAX_NO_OPERANDS];
    char *data = PyArray_BYTES(ary);
    PyArrayObject *base = (PyArrayObject *) ary->base;
    printf("PyCphVB_NewViewArray\n");

    if(base == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "PyCphVB_NewViewArray - the PyArrayObject "
                        "must have a base.\n");
        return -1;
    }
    if(PyCphVB_ARRAY(base) == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "PyCphVB_NewViewArray - the base must "
                        "have an associated cphvb_array.\n");
        return -1;
    }
    if(dtype == CPHVB_UNKNOWN)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "cphVB does not support the datatype\n");
        return -1;
    }
    if(PyArray_TYPE(ary) != PyArray_TYPE(base))
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "PyCphVB_NewViewArray - the type of the "
                        "view and base must be identical.\n");
        return -1;
    }
    if(base->mprotected_start > ((cphvb_intp) data) ||
       base->mprotected_end <= ((cphvb_intp) data))
    {
        PyErr_Format(PyExc_RuntimeError, "PyCphVB_NewViewArray - the "
                     "view data (%p) is not inside the interval of "
                     "its base array (%p to %p).\n", data, base->data,
                     (char *) base->mprotected_end);
        return -1;
    }
    //Compute offset in elements from the start of the base array.
    offset = ((cphvb_intp) data) - base->mprotected_start;

    //Convert bytes to element size.
    //This works because the array is behaved.
    assert(offset % PyArray_ITEMSIZE(ary) == 0);
    offset /= PyArray_ITEMSIZE(ary);
    for(i=0; i<PyArray_NDIM(ary); ++i)
    {
        strides[i] = PyArray_STRIDE(ary,i) / PyArray_ITEMSIZE(ary);
        assert(PyArray_STRIDE(ary,i) % PyArray_ITEMSIZE(ary) == 0);
    }

    //Tell the VEM.
    err = vem_create_array(PyCphVB_ARRAY(base),
                           dtype, PyArray_NDIM(ary), offset,
                           PyArray_DIMS(ary), strides, 0,
                           (cphvb_constant)0L, &PyCphVB_ARRAY(ary));
    return err;
} /* PyCphVB_NewViewArray */

/*
 *===================================================================
 * Delete array.
 * It is up to the caller to make sure that no view is deleted
 * before its base.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_DelViewArray(PyArrayObject *array)
{
    //Free the data pointer in the NumPy address space.
    if(PyCphVB_MfreeArray(array) == -1)
        return 0;

    if(PyCphVB_ARRAY(array) != NULL)
    {
        printf("PyCphVB_DelViewArray - deleting cphVB handled array\n");
        cphvb_instruction inst;
        inst.opcode = CPHVB_DESTROY;
        inst.operand[0] = PyCphVB_ARRAY(array);
        batch_schedule(&inst);

        if(array->base == NULL)//It it a base array.
        {
            //Remove the array from the base array collection.
            if(array->next != NULL)
                array->next->prev = array->prev;
            if(array->prev != NULL)
                array->prev->next = array->next;
            else
                ary_root = array->next;
        }
    }
    return 0;
} /* PyCphVB_DelViewArray */


/*
 *===================================================================
 * Indicate that cphVB should handle the array and all associated
 * array views and the array base.
 *
 * @array         The array cphVB should handle.
 * @transfer_data Whether data should be transferred from NumPy to
 *                cphVB address space.
 * @return        -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_HandleArray(PyArrayObject *array, int transfer_data)
{
    cphvb_error err;
    cphvb_instruction inst;
    cphvb_intp size = PyArray_NBYTES(array);
    cphvb_array *a = PyCphVB_ARRAY(array);
    PyArrayObject *base = NULL;

    //Check if the array is a view because in that case we also have to
    //handle the base array.
    if(array->base != NULL && PyArray_CheckExact(array->base) &&
       !PyArray_CHKFLAGS(array, NPY_UPDATEIFCOPY))
    {
        base = (PyArrayObject *) array->base;
        if(PyCphVB_HandleArray(base, transfer_data) == -1)
            return -1;
    }

    if(!PyArray_ISBEHAVED(array))//The array must be behaved.
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "PyCphVB_HandleArray - the view must be "
                        "behaved.\n");
        return -1;
    }

    if(base != NULL)//It's a view.
    {
        if(a == NULL)//The view has never been handled by cphVB before.
            return PyCphVB_NewViewArray(array);
        return 0;
    }

    //It's a base array.
    if(a != NULL && array->cphvb_handled)
        return 0;//And it is already being handled by cphVB.

    if(a == NULL)//The base array has never been handled by cphVB before.
    {
        PyCphVB_NewBaseArray(array);
        a = PyCphVB_ARRAY(array);
    }
    else if(transfer_data)
    {
        //Tell the VEM to syncronize the data.
        inst.opcode = CPHVB_SYNC;
        inst.operand[0] = a;
        batch_schedule(&inst);
        batch_flush();
    }

    if(transfer_data)
    {
        //Make sure that the memory is allocated.
        err = cphvb_malloc_array_data(a);
        if(err != CPHVB_SUCCESS)
        {
            PyErr_Format(PyExc_RuntimeError,"Error when allocating "
                         "array (%p): %s\n", a, cphvb_error_text(err));
            return -1;
        }
        //We need to move data from NumPy to cphVB address space.
        memcpy(a->data, array->data, size);
        memset(array->data, 0, size); //DEBUG;
    }

    //Proctect the NumPy array data.
    if(mprotect(array->data, size, PROT_NONE) == -1)
    {
        int errsv = errno;//mprotect() sets the errno.
        PyErr_Format(PyExc_RuntimeError,"Error - could not protect a data"
                     "data region. Returned error code by mprotect: %s.\n",
                     strerror(errsv));
        return -1;
    }

    //The array is now handled by cphVB.
    array->cphvb_handled = 1;
    return 0;
} /* PyCphVB_HandleArray */


/*
 *===================================================================
 * Easy retrieval of array's base.
 *
 * @array         The array view (or base).
 * @return        The base or NULL on error.
 */
static PyArrayObject *
PyCphVB_BaseArray(PyArrayObject *array)
{
    if(array->base == NULL)
    {
        assert(array->data != NULL);
        return array;
    }

    //To be a view the array must be a PyArrayObject and not have the
    //flag NPY_UPDATEIFCOPY set.
    if(PyArray_CheckExact(array->base) &&
       !PyArray_CHKFLAGS(array, NPY_UPDATEIFCOPY))
    {
        return (PyArrayObject *) array->base;
    }

    PyErr_SetString(PyExc_RuntimeError, "PyCphVB_BaseArray - the "
                    "array is not a base or a cphVB compatible view.\n");
    return NULL;
}/* PyCphVB_BaseArray */
