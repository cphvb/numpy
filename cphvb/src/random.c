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

#include <cphvb.h>

static cphvb_intp random_id;

/*
 *===================================================================
 * Initiate PyCphVB_Random().
 * @return -1 and set exception on error, 0 on success.
 */
static int
random_init(void)
{
/*
    reduce_id = 0;
    cphvb_error err = vem_reg_func(NULL, "cphvb_random", &random_id);
    if(err != CPHVB_SUCCESS)
    {
        fprintf(stderr, "Fatal error in the initialization of the user"
            "-defined reduce operation: %s.\n", cphvb_error_text(err));
        exit(err);
    }
    if(random_id <= 0)
    {
        fprintf(stderr, "Fatal error in the initialization of the user"
                        "-defined random operation: invalid ID returned"
                        " (%ld).\n", random_id);
        exit(err);
    }
*/
    return 0;
}

/*
 *===================================================================
 * Execute an random.
 * @ary    Uninitialized Base Array
 * @return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_Random(PyArrayObject *ary)
{
    cphvb_instruction inst;
    cphvb_random_type *rinst;

    if(PyCphVB_HandleArray(ary, 0) == -1)
        return -1;

    //Allocate the user-defined function.
    rinst = malloc(sizeof(cphvb_random_type));
    if(rinst == NULL)
    {
        PyErr_NoMemory();
        return -1;
    }

    //Set the instruction
    rinst->id          = random_id;
    rinst->nout        = 1;
    rinst->nin         = 0;
    rinst->struct_size = sizeof(cphvb_random_type);
    rinst->operand[0]  = PyCphVB_ARRAY(ary);
    inst.opcode        = CPHVB_USERFUNC;
    inst.userfunc      = (cphvb_userfunc *) rinst;

    batch_schedule(&inst);

    return 0;
}


