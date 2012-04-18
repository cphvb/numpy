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
 * Copy data from src into dst. The function do nothing when the type
 * is not supported.
 * @dst    Destination array.
 * @src     Source array.
 * @return -1 and set exception on error, 0 on success, 1 when doing
 *         nothing.
 */
static int
PyCphVB_CopyInto(PyArrayObject *dst, PyArrayObject *src)
{
    int i;
    cphvb_instruction inst;
    cphvb_error err;

    //Make sure cphVB handles all operands.
    if(PyCphVB_HandleArray(dst, 1) != 0)
    {
        PyErr_Clear();
        return 1;//This in not a fatal error.
    }
    if(PyCphVB_HandleArray(src, 1) != 0)
    {
        PyErr_Clear();
        return 1;//This in not a fatal error.
    }

    //Create an instruction that represent the ufunc operation.
    //Note that the outputs are the first operands.
    inst.opcode = CPHVB_IDENTITY;
    inst.operand[0] = PyCphVB_ARRAY(dst);
    inst.operand[1] = PyCphVB_ARRAY(src);

    //Broadcast to match the number of dimensions of the output by
    //appending and extending 1-length dimensions.
    for(i=1; i<2; ++i)
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
                       inst.operand[0]->shape) > 0)
        batch_schedule(&inst);

    return 0;
}/* PyCphVB_CopyInto */

