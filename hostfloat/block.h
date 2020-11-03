// ************************************************************************
//
// miniAMR: stencil computations with boundary exchange and AMR.
//
// Copyright (2014) Sandia Corporation. Under the terms of Contract
// DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government 
// retains certain rights in this software.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
// Questions? Contact Courtenay T. Vaughan (ctvaugh@sandia.gov)
//                    Richard F. Barrett (rfbarre@sandia.gov)
//
// ************************************************************************

#ifndef BLOCK_H
#define BLOCK_H

//typedef int num_sz;
typedef long long num_sz;

typedef struct {
   num_sz number;
   num_sz num_prime;
   int level;
   int refine;
   int new_proc;
   num_sz parent;       // if original block -1,
                     // else if on node, number in structure
                     // else (-2 - parent->number)
   int parent_node;
   int child_number;
   int nei_refine[6];
   int nei_level[6];  /* 0 to 5 = W, E, S, N, D, U; use -2 for boundary */
   int nei[6][2][2];  /* negative if off processor (-1 - proc) */
   int cen[3];
   float ****array;
} block;
block *blocks __attribute__((weak));

typedef struct {
   num_sz number;
   num_sz num_prime;
   int level;
   num_sz parent;      // -1 if original block
   int parent_node;
   int child_number;
   int refine;
   num_sz child[8];    // n if on node, number if not
                    // if negative, then onnode child is a parent (-1 - n)
   int child_node[8];
   int cen[3];
} parent;
parent *parents __attribute__((weak));

typedef struct {
   num_sz number;     // number of block
   int n;          // position in block array
} sorted_block;
sorted_block *sorted_list __attribute__((weak));
int *sorted_index __attribute__((weak));

int my_pe __attribute__((weak));
int num_pes __attribute__((weak));

int max_num_blocks __attribute__((weak));
int num_refine __attribute__((weak));
int uniform_refine __attribute__((weak));
int x_block_size __attribute__((weak)), y_block_size __attribute__((weak)), z_block_size __attribute__((weak));
int num_vars __attribute__((weak));
int comm_vars __attribute__((weak));
int init_block_x __attribute__((weak)), init_block_y __attribute__((weak)), init_block_z __attribute__((weak));
int reorder __attribute__((weak));
int npx __attribute__((weak)), npy __attribute__((weak)), npz __attribute__((weak));
int inbalance __attribute__((weak));
int refine_freq __attribute__((weak));
int report_diffusion __attribute__((weak));
int error_tol __attribute__((weak));
int num_tsteps __attribute__((weak));
int use_time __attribute__((weak));
double end_time __attribute__((weak));
int stages_per_ts __attribute__((weak));
int checksum_freq __attribute__((weak));
int stencil __attribute__((weak));
int report_perf __attribute__((weak));
int plot_freq __attribute__((weak));
int num_objects __attribute__((weak));
int lb_opt __attribute__((weak));
int block_change __attribute__((weak));
int code __attribute__((weak));
int permute __attribute__((weak));
int nonblocking __attribute__((weak));
int refine_ghost __attribute__((weak));
int change_dir __attribute__((weak));
int group_blocks __attribute__((weak));
int limit_move __attribute__((weak));
int send_faces __attribute__((weak));
int use_rcb __attribute__((weak));

int no_validate __attribute__((weak));

int first __attribute__((weak));
int *dirs __attribute__((weak));
int num_cells __attribute__((weak));
int mat __attribute__((weak));
int max_num_parents __attribute__((weak));
int num_parents __attribute__((weak));
int max_active_parent __attribute__((weak));
int cur_max_level __attribute__((weak));
num_sz *num_blocks __attribute__((weak));
num_sz *local_num_blocks __attribute__((weak));
num_sz *block_start __attribute__((weak));
int num_active __attribute__((weak));
int max_active_block __attribute__((weak));
num_sz global_active __attribute__((weak));
int x_block_half __attribute__((weak)), y_block_half __attribute__((weak)), z_block_half __attribute__((weak));
double tol __attribute__((weak));
double *grid_sum __attribute__((weak));
int *p8 __attribute__((weak)), *p2 __attribute__((weak));
int mesh_size[3] __attribute__((weak));
int max_mesh_size __attribute__((weak));
int *from __attribute__((weak)), *to __attribute__((weak));
int msg_len[3][4] __attribute__((weak));
int local_max_b __attribute__((weak));
int global_max_b __attribute__((weak));
double *alpha __attribute__((weak)), beta_arg __attribute__((weak));
double total_fp_divs __attribute__((weak));
double total_fp_adds __attribute__((weak));
double total_fp_muls __attribute__((weak));

typedef struct {
   int type;
   int bounce;
   double cen[3];
   double orig_cen[3];
   double move[3];
   double orig_move[3];
   double size[3];
   double orig_size[3];
   double inc[3];
} object;
object *objects __attribute__((weak));

int num_dots __attribute__((weak));
int max_num_dots __attribute__((weak));
int max_active_dot __attribute__((weak));
typedef struct {
   num_sz number;
   int n;
   int proc;
   int new_proc;
   int cen[3];
} dot;
dot *dots __attribute__((weak));
typedef struct {
   num_sz number;
   num_sz num_prime;
   int n;
   int proc;
   int new_proc;
} spot;
spot *spots __attribute__((weak));

#endif