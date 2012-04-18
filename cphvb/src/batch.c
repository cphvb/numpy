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


//Instructions currently scheduled.
static cphvb_instruction inst_scheduled[CPHVB_MAX_NO_INST];
//Number of instructions currently scheduled.
static cphvb_intp ninst_scheduled = 0;

/*
 *===================================================================
 * Flush scheduled instruction to the VEM.
 */
void batch_flush(void)
{
    cphvb_error error;
    if(ninst_scheduled > 0)
    {
        error = vem_execute(ninst_scheduled, inst_scheduled);
        if(error == CPHVB_PARTIAL_SUCCESS)//Only partial success
        {
            int i,j;
            fprintf(stderr, "Error in scheduled batch of "
                    "instructions: %s\n", cphvb_error_text(error));
            for(i=0; i<ninst_scheduled; ++i)
            {
                cphvb_instruction *ist = &inst_scheduled[i];
                fprintf(stderr,"\tOpcode: %s, Operand types:",
                        cphvb_opcode_text(ist->opcode));
                for(j=0; j<cphvb_operands(ist->opcode); ++j)
                    fprintf(stderr," %s", cphvb_type_text(
                                          cphvb_type_operand(ist,j)));
                fprintf(stderr,", Status: %s\n",
                        cphvb_error_text(ist->status));
            }
            exit(error);
        }
        if(error != CPHVB_SUCCESS)
        {
            fprintf(stderr, "Unhandled error returned by "
                            "vem_execute() in util_schedule(): %s\n",
                            cphvb_error_text(error));
            exit(error);
        }
        ninst_scheduled = 0;
    }
} /* batch_flush */


/*
 *===================================================================
 * Schedule one instruction.
 */
void batch_schedule(cphvb_instruction *inst)
{
    if(inst != NULL)
    {
        inst->status = CPHVB_INST_UNDONE;
        inst_scheduled[ninst_scheduled++] = *inst;
    }
    if(ninst_scheduled >= CPHVB_MAX_NO_INST)
        batch_flush();

} /* batch_schedule */

