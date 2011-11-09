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
 * Initiate the VEM interface.
 */
void vem_if_init(void)
{
    cphvb_error err;
    cphvb_com **coms;
    cphvb_intp children_count;

    //We are the root in the configuration.
    self_component = cphvb_com_setup();

    cphvb_com_children(self_component, &children_count, &coms);

    if(children_count != 1 || coms[0]->type != CPHVB_VEM)
    {
        fprintf(stderr, "Error in the configuration: the bridge must "
                        "have exactly one child of type VEM\n");
        exit(-1);
    }
    vem_component = coms[0];
    free(coms);
    vem_init = vem_component->init;
    vem_execute = vem_component->execute;
    vem_shutdown = vem_component->shutdown;
    vem_create_array = vem_component->create_array;

    err = vem_init(NULL, NULL, NULL, NULL, vem_component);
    if(err)
    {
        fprintf(stderr, "Error in vem_init()\n");
        exit(-1);
    }

} /* vem_if_init */


/*
 *===================================================================
 * Finalize the VEM interface.
 */
void vem_if_finalize(void)
{
    vem_shutdown();
    cphvb_com_free(self_component);
    cphvb_com_free(vem_component);
} /* vem_if_finalize */
