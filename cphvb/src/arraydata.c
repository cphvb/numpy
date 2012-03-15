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
 * The Array Data Protection handles the event when NumPy or external
 * libraries access the array data directly. Since DistNumPy distribute
 * this data, the result of such direct array data access is a
 * segmentation fault. The handle this access we allocate protected
 * memory and makes the local array data pointer points to this memory.
 */

#include <errno.h>
#include <sys/mman.h>
#include <signal.h>

/*
 *===================================================================
 * Allocate cphVB-compatible memory.
 * @array  The array that should own the memory.
 * @size   The size of the memory allocation (in bytes).
 * @return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_MallocArray(PyArrayObject *ary, cphvb_intp size)
{
    //Allocate page-size aligned memory.
    //The MAP_PRIVATE and MAP_ANONYMOUS flags is not 100% portable. See:
    //<http://stackoverflow.com/questions/4779188/how-to-use-mmap-to-allocate-a-memory-in-heap>
    void *addr = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(addr == MAP_FAILED)
    {
        int errsv = errno;//mmap() sets the errno.
        PyErr_Format(PyExc_RuntimeError, "The Array Data Protection "
                     "could not mmap a data region. "
                     "Returned error code by mmap: %s.", strerror(errsv));
        return -1;
    }
    //Update the ary data pointer.
    PyArray_BYTES(ary) = addr;
    //We also need to save the start and end address.
    ary->mprotected_start = addr;
    ary->mprotected_end = ary->mprotected_start + size;

    //Save the number of bytes allocated.
    ary->data_allocated = size;

    return 0;
}/* PyCphVB_MallocArray */

/*
 *===================================================================
 * Free cphVB-compatible memory.
 * @array The array that should own the memory.
 * @return -1 and set exception on error, 0 on success.
 */
static int
PyCphVB_MfreeArray(PyArrayObject *ary)
{
    void *addr = PyArray_DATA(ary);

    if(munmap(addr, ary->data_allocated) == -1)
    {
        int errsv = errno;//munmmap() sets the errno.
        PyErr_Format(PyExc_RuntimeError, "The Array Data Protection "
                     "could not mummap a data region. "
                     "Returned error code by mmap: %s.", strerror(errsv));
        return -1;
    }
    return 0;
} /* PyCphVB_MfreeArray */

/*
 *===================================================================
 * Signal handler for SIGSEGV.
 * Private.
 */
static void
sighandler(int signal_number, siginfo_t *info, void *context)
{
    cphvb_error err;
    //Get array base
    PyArrayObject *ary = arraycollection_get(info->si_addr);

    if(ary == NULL)//Normal segfault.
    {
        signal(signal_number, SIG_DFL);
    }
    else//Segfault triggered by accessing the protected data pointer.
    {
        cphvb_instruction inst;
        cphvb_intp size = PyArray_NBYTES(ary);
        cphvb_array *a = PyCphVB_ARRAY(ary);
        /*
        printf("Warning - un-handle array(%p) because of "
               "direct data access(%p). size: %ld\n", a, info->si_addr, size);
        */

        //Tell the VEM to syncronize the data.
        inst.opcode = CPHVB_SYNC;
        inst.operand[0] = a;
        batch_schedule(&inst);
        batch_flush();

        //Make sure that the memory is allocated.
        err = cphvb_data_malloc(a);
        if(err != CPHVB_SUCCESS)
        {
            fprintf(stderr, "Error when allocating array (%p): %s\n",
                            a, cphvb_error_text(err));
            exit(err);
        }

        //mremap does not work since a->data is not guaranteed to be
        //page aligned.
        if(mremap(a->data, size, size, MREMAP_FIXED|MREMAP_MAYMOVE,
                  ary->data) == MAP_FAILED)
        {
            int errsv = errno;//mremap() sets the errno.
            fprintf(stderr,"Error - could not mremap a data region."
                           " Returned error code by mremap: %s.\n",
                           strerror(errsv));
            exit(errno);
        }
        //The cphvb data will have to be allocated again when
        //PyCphVB_HandleArray() moves the data back again to the
        //cphVB address space.
        a->data = NULL;

        //The array is not handled by cphVB anymore.
        ary->cphvb_handled = 0;

    }
}

/*
 *===================================================================
 * Initialization of the Array Data Protection.
 */
int arydat_init(void)
{
   // Install Signal handler
   struct sigaction sact;

   sigfillset(&(sact.sa_mask));
   sact.sa_flags = SA_SIGINFO | SA_ONSTACK;
   sact.sa_sigaction = sighandler;
   sigaction (SIGSEGV, &sact, &sact);


    return 0;
} /* arydat_init */

/*
 *===================================================================
 * Finalization of the Array Data Protection.
 */
int arydat_finalize(void)
{

    return 0;
} /* arydat_finalize */

