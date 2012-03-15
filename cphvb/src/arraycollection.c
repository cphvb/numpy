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

#include <assert.h>
#include <pthread.h>

//The root of the base array collection.
static PyArrayObject *arraycollection_root = NULL;

//The mutex used when accessing the array collection.
static pthread_mutex_t arraycollection_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 *===================================================================
 * Add an base array to the array collection.
 */
static void arraycollection_add(PyArrayObject *ary)
{
    pthread_mutex_lock(&arraycollection_mutex);
    ary->prev = NULL;
    ary->next = arraycollection_root;
    arraycollection_root = ary;
    if(ary->next != NULL)
    {
        assert(ary->next->prev == NULL);
        ary->next->prev = arraycollection_root;
    }
    pthread_mutex_unlock(&arraycollection_mutex);
}

/*
 *===================================================================
 * Get base array that owns the memory address.
 * Return NULL when nobody owns the address.
 */
static PyArrayObject* arraycollection_get(char* addr)
{

    //Iterate through all base arrays.
    PyArrayObject *ary = arraycollection_root;
    while(ary != NULL)
    {
        char *start = PyArray_BYTES(ary);
        char *end = start + PyArray_NBYTES(ary);
        if(start <= addr && addr < end)
           return ary;

        //Go to the next ary.
        ary = ary->next;
    }
    return NULL;
}

/*
 *===================================================================
 * Remove an base array from the array collection.
 */
static void arraycollection_rm(PyArrayObject *ary)
{
    assert(ary->base == NULL);

    pthread_mutex_lock(&arraycollection_mutex);
    if(ary->next != NULL)
        ary->next->prev = ary->prev;
    if(ary->prev != NULL)
        ary->prev->next = ary->next;
    else
        arraycollection_root = ary->next;
    pthread_mutex_unlock(&arraycollection_mutex);

    ary->next = NULL;//To easy debugging
    ary->prev = NULL;
}





