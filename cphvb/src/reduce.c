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

static cphvb_intp reduce_id;

//The type of the user-defined reduce function
typedef struct
{
    //User-defined function header with two operands.
    CPHVB_USER_FUNC_HEADER(2)
    //The Axis to reduce
    cphvb_error   axis;
    //The opcode to reduce with
    cphvb_opcode  opcode;
} reduce_type;

/*
 *===================================================================
 * Initiate PyCphVB_Reduce().
 * @return -1 and set exception on error, 0 on success.
 */
static int
reduce_init(void)
{
    reduce_id = 0;
    cphvb_error err = vem_reg_func("reduce", "reduce", &reduce_id);
    if(err != CPHVB_SUCCESS)
    {
        fprintf(stderr, "Fatal error in the initialization of the user"
            "-defined reduce operation: %s.\n", cphvb_error_text(err));
        exit(err);
    }
    if(reduce_id <= 0)
    {
        fprintf(stderr, "Fatal error in the initialization of the user"
                        "-defined reduce operation: invalid ID returned"
                        " (%ld).\n", reduce_id);
        exit(err);
    }

    return 0;
}

/*
 *===================================================================
 * Execute an reduce. The function do nothing when the opcode and/or
 * type is not supported.
 * @ufunc  The ufunc object to reduce with.
 * @in     Input Array.
 * @put    Output Array.
 * @axis   The Axis to reduce.
 * @return -1 and set exception on error, 0 on success, 1 when doing
 *         nothing.
 */
static int
PyCphVB_Reduce(PyUFuncObject *ufunc, PyArrayObject *in,
               PyArrayObject *out, int axis)
{
    cphvb_instruction inst;
    reduce_type *rinst;

    if(ufunc->opcode == CPHVB_NONE)
        return 1;//opcode not supported.

    //Make sure cphVB handles all operands.
    if(PyCphVB_HandleArray(in, 1) != 0 ||
       PyCphVB_HandleArray(out,1) != 0)
    {
        PyErr_Clear();
        return 1;//This in not a fatal error.
    }

    //Allocate the user-defined function.
    rinst = malloc(sizeof(reduce_type));
    if(rinst == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }

    //Set the instruction
    rinst->id          = reduce_id;
    rinst->nout        = 1;
    rinst->nin         = 1;
    rinst->struct_size = sizeof(reduce_type);
    rinst->operand[0]  = PyCphVB_ARRAY(out);
    rinst->operand[1]  = PyCphVB_ARRAY(in);
    rinst->opcode      = ufunc->opcode;
    rinst->axis        = axis;
    inst.opcode        = CPHVB_USERFUNC;
    inst.userfunc      = (cphvb_userfunc *) rinst;

    batch_schedule(&inst);

    return 1;
}


