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
PyDistArray_Ufunc(PyUFuncObject *ufunc, PyArrayObject **op)
{
    int i;
    cphvb_instruction inst;
    cphvb_error err;
    printf("PyDistArray_Ufunc: %s\n", cphvb_opcode_text(ufunc->opcode));

    if(ufunc->opcode == CPHVB_NONE)
        return 1;//opcode not supported.

    if(ufunc->nargs > CPHVB_MAX_NO_OPERANDS)
        return 1;//To many arguments.

    //Make sure cphVB handles all operands.
    for(i=0; i<ufunc->nargs; ++i)
        PyDistArray_HandleArray(op[i], 1);

    //Create an instruction that represent the ufunc operation.
    //Note that the outputs are the first operands.
    inst.opcode = ufunc->opcode;
    for(i=0; i<ufunc->nin; ++i)
    {
        cphvb_intp j = i + ufunc->nout;
        inst.operand[j] = PyDistArray_ARRAY(op[i]);
    }
    for(i=0; i<ufunc->nout; ++i)
    {
        cphvb_intp j = i + ufunc->nin;
        inst.operand[i] = PyDistArray_ARRAY(op[j]);
    }

    //Broadcast to match the number of dimensions of the output by
    //appending and extending 1-length dimensions.
    for(i=ufunc->nout; i<ufunc->nargs; ++i)
    {
        if(inst.operand[i] != CPHVB_CONSTANT )//&&
           //inst.operand[i]->ndim < inst.operand[0]->ndim)
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
                    char errtxt[256];
                    sprintf(errtxt, "Ufunc - internal error when "
                            "calling vem_create_array(): %s.\n",
                            cphvb_error_text(err));
                    PyErr_SetString(PyExc_RuntimeError, errtxt);
                    return -1;
                }
                inst.operand[i] = tary;
            }
        }
    }
    //Only execute the ufunc if the output is greater than zero.
    if(cphvb_nelements(inst.operand[0]->ndim,
                       inst.operand[0]->shape) > 0)
        batch_schedule(&inst);

    return 0;
}/* PyDistArray_Ufunc */

