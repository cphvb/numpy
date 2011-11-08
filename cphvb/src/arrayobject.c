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

/*
 *===================================================================
 * Create a new base array and updates the PyArrayObject.
 * If 'one_node_dist_rank' is positive it specifies the rank of an
 * one-node-distribution.
 * Return -1 and set exception on error, 0 on success.
 */
static int
PyDistArray_NewBaseArray(PyArrayObject *ary)
{
    /*
    int i;

    //Create dndarray.
    dndarray newarray;
    newarray.dtype = PyArray_TYPE(ary);
    newarray.elsize = PyArray_ITEMSIZE(ary);
    newarray.ndims = PyArray_NDIM(ary);
    newarray.nelem = PyArray_SIZE(ary);
    newarray.isdist = 1;
    newarray.refcount = 1;
    newarray.onerank = one_node_dist_rank;
    for(i=0; i<PyArray_NDIM(ary); i++)
        newarray.dims[i] = PyArray_DIM(ary, i);

    //Create dndview. NB: the base will have to be set when 'newarray'
    //has found its final resting place. (Done by put_dndarray).
    dndview newview;
    newview.uid = ++uid_count;
    newview.nslice = PyArray_NDIM(ary);
    newview.ndims = PyArray_NDIM(ary);
    newview.alterations = 0;
    for(i=0; i<PyArray_NDIM(ary); i++)
    {
        //Default the view will span over the whole array.
        newview.slice[i].start = 0;
        newview.slice[i].step = 1;
        newview.slice[i].nsteps = PyArray_DIM(ary, i);
    }

    dndview *ret = handle_NewBaseArray(&newarray, &newview);

    if(ret == NULL)
        return -1;

    PyDistArray_ARRAY(ary) = ret;
    ret->base->pyary = ary;

    //Protect the original NumPy data pointer.
    //This is only done by the Master MPI Process.
    return arydat_malloc(ary);
    */
    return 0;
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
    /*
    //We have to free the protected data pointer when the NumPy array
    //is not a view.
    if((array->flags & NPY_OWNDATA) && array->data != NULL)
        return arydat_free(array);
    */
    return 0;

} /* PyDistArray_DelViewArray */
