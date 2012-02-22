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

/*
 *===================================================================
 * Execute an ufunc. The function do nothing when the opcode and/or
 * type is not supported.
 * @ufunc  The ufunc object.
 * @op     List of operands (inputs before outputs).
 * @return -1 and set exception on error, 0 on success, 1 when doing
 *         nothing.
 */
static int
PyCphVB_Ufunc(PyUFuncObject *ufunc, PyArrayObject **op)
{
    int i;
    cphvb_instruction inst;
    cphvb_error err;
    int data_overlap = 0;

    if(ufunc->opcode == CPHVB_NONE)
        return 1;//opcode not supported.

    if(ufunc->nargs > CPHVB_MAX_NO_OPERANDS)
        return 1;//To many arguments.

    //Make sure cphVB handles all operands.
    for(i=0; i<ufunc->nargs; ++i)
    {
        if(PyCphVB_HandleArray(op[i], 1) != 0)
        {
            PyErr_Clear();
            return 1;//This in not a fatal error.
        }
    }

    //Create an instruction that represent the ufunc operation.
    //Note that the outputs are the first operands.
    inst.opcode = ufunc->opcode;
    for(i=0; i<ufunc->nin; ++i)
    {
        cphvb_intp j = i + ufunc->nout;
        inst.operand[j] = PyCphVB_ARRAY(op[i]);
    }
    for(i=0; i<ufunc->nout; ++i)
    {
        cphvb_intp j = i + ufunc->nin;
        inst.operand[i] = PyCphVB_ARRAY(op[j]);
    }

    //Broadcast to match the number of dimensions of the output by
    //appending and extending 1-length dimensions.
    for(i=ufunc->nout; i<ufunc->nargs; ++i)
    {
        cphvb_intp j;
        cphvb_intp shape[CPHVB_MAXDIM];
        cphvb_intp stride[CPHVB_MAXDIM];
        cphvb_array *tary;
        cphvb_intp nd_diff = inst.operand[0]->ndim -
                             inst.operand[i]->ndim;
        int broadcast = 0; // false
        for(j=inst.operand[0]->ndim-1; j>=nd_diff; --j)
        {
            shape[j] = inst.operand[i]->shape[j-nd_diff];
            if(shape[j] < inst.operand[0]->shape[j])
            {
                assert(shape[j] == 1);
                shape[j] = inst.operand[0]->shape[j];
                stride[j] = 0;
                broadcast = 1; // true
            }
            else
            {
                stride[j] = inst.operand[i]->stride[j-nd_diff];
            }
        }
        for(j=nd_diff-1; j>=0; --j)
        {
            shape[j] = 1;
            stride[j] = 0;
            broadcast = 1; // true
        }
        if (broadcast)
        {
            err = vem_create_array(cphvb_base_array(inst.operand[i]),
                                   inst.operand[i]->type,
                                   inst.operand[0]->ndim,
                                   inst.operand[i]->start,
                                   shape,
                                   stride,
                                   NPY_FALSE,
                                   (cphvb_constant)0L,
                                   &tary);
            if(err)
            {
                PyErr_Format(PyExc_RuntimeError, "Ufunc - internal "
                             "error when calling vem_create_array(): "
                             "%s.\n", cphvb_error_text(err));
                return -1;
            }
            inst.operand[i] = tary;
        }
    }
    //Only execute the ufunc if the output is greater than zero.
    if(cphvb_nelements(inst.operand[0]->ndim,
                       inst.operand[0]->shape) <= 0)
        return 0;


    //Check for data overlap.
    {
        cphvb_array *out = cphvb_base_array(inst.operand[0]);
        for(i=ufunc->nout; i<ufunc->nargs; ++i)
            if(out == cphvb_base_array(inst.operand[i]))
            {
                data_overlap = 1;
                break;
            }
    }
    err = 0;
    if(data_overlap)
    {
        /* We will NOT use a temporary array.
         * It would make sense but NumPy simply overwrite the previous
         * element when computing from left to right.
         * Therefore, we will let NumPy compute the Ufunc.

        PyArrayObject *out = op[ufunc->nin];

        //We need to use a temporary array
        PyArrayObject *tmp = (PyArrayObject *) PyArray_NewLikeArray(out, NPY_KEEPORDER, NULL, 0);
        if(tmp == NULL || PyCphVB_HandleArray(tmp, 0) != 0)
            return -1;

        //Lets schedule an instruction that writes to the tmp array.
        inst.operand[0] = PyCphVB_ARRAY(tmp);
        batch_schedule(&inst);

        //Lets copy the output data back into the original output array.
        err = PyCphVB_CopyInto(out,tmp);
        Py_DECREF(tmp);
        */
        return 1;
    }
    else
    {
        batch_schedule(&inst);
    }

    return err;
}/* PyCphVB_Ufunc */

