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

#ifndef VEM_INTERFACE_H
#define VEM_INTERFACE_H
#ifdef __cplusplus
extern "C" {
#endif

#include <cphvb.h>

//Function pointers that makes up the VEM interface.
cphvb_init vem_init;
cphvb_execute vem_execute;
cphvb_shutdown vem_shutdown;
cphvb_create_array vem_create_array;
cphvb_com *self_component, *vem_component;

/*
 *===================================================================
 * Initiate the VEM interface.
 */
void vem_if_init(void);

/*
 *===================================================================
 * Finalize the VEM interface.
 */
void vem_if_finalize(void);



#ifdef __cplusplus
}
#endif

#endif /* !defined(VEM_INTERFACE_H) */
