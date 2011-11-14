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
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_NewBaseArray(PyArrayObject *ary)
{
    int i;
    cphvb_index s;
    cphvb_index stride[CPHVB_MAXDIM];
    cphvb_index shape[CPHVB_MAXDIM];
    cphvb_error err;
    cphvb_type dtype = type_py2cph[PyArray_TYPE(ary)];

    printf("PyDistArray_NewBaseArray - type: %s\n",cphvb_type_text(dtype));

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

    //Compute the stride. Row-Major (C-style)
    s=1;
    for(i=PyArray_NDIM(ary)-1; i>=0; --i)
    {
        stride[i] = s;
        shape[i] = (cphvb_index) PyArray_DIM(ary,i);
        s *= shape[i];
    }

    err = vem_create_array(NULL, dtype, PyArray_NDIM(ary), 0, shape,
                           stride, 0, (cphvb_constant)0L,
                           &PyDistArray_ARRAY(ary));
    return err;
} /* PyDistArray_NewBaseArray */

/*
 *===================================================================
 * Delete array view.
 * When it is the last view of the base array, the base array is de-
 * allocated.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_DelViewArray(PyArrayObject *array)
{
    //Free the data pointer in the NumPy address space.
    if(PyDistArray_MfreeArray(array) == -1)
        return 0;

    if(PyDistArray_ARRAY(array) != NULL)
    {
        printf("PyDistArray_DelViewArray - deleting cphVB handled array\n");
        cphvb_instruction inst;
        inst.opcode = CPHVB_DESTROY;
        inst.operand[0] = PyDistArray_ARRAY(array);
        batch_schedule(&inst);
    }

    return 0;

} /* PyDistArray_DelViewArray */


/*
 *===================================================================
 * Indicate that cphVB should handle the array.
 * @array The array cphVB should handle.
 * @transfer_data Whether data should be transferred from NumPy to
 *                cphVB address space.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_HandleArray(PyArrayObject *array, int transfer_data)
{
    printf("PyDistArray_HandleArray, transfer_data: %d\n", transfer_data);
    cphvb_error err;
    cphvb_instruction inst;
    cphvb_intp size = PyArray_NBYTES(array);
    cphvb_array *a = PyDistArray_ARRAY(array);

    if(a == NULL)//Array has never been handled by cphVB before.
    {
        PyDistArray_NewBaseArray(array);
        a = PyDistArray_ARRAY(array);
    }
    else if(transfer_data)
    {
        //Tell the VEM to syncronize the data.
        inst.opcode = CPHVB_SYNC;
        inst.operand[0] = a;
        batch_schedule(&inst);
        batch_flush();
    }

    assert(a->base == NULL);//Base Array for now.

    if(transfer_data)
    {
        //Make sure that the memory is allocated.
        err = cphvb_malloc_array_data(a);
        if(err != CPHVB_SUCCESS)
        {
            fprintf(stderr, "Error when allocating array (%p): %s\n",
                            a, cphvb_error_text(err));
            exit(err);
        }

        //We need to move data from NumPy to cphVB address space.
        memcpy(a->data, array->data, size);
    }

    //Proctect the NumPy array data.
    if(mprotect(array->data, size, PROT_NONE) == -1)
    {
        int errsv = errno;//mprotect() sets the errno.
        fprintf(stderr,"Error - could not protect a data region."
                       " Returned error code by mprotect: %s.\n",
                       strerror(errsv));
        exit(errno);
    }
    return 0;

} /* PyDistArray_HandleArray */

