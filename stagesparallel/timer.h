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

#ifndef TIMER_H
#define TIMER_H

double average[135] __attribute__((weak));
double stddev[132] __attribute__((weak));
double minimum[132] __attribute__((weak));
double maximum[132] __attribute__((weak));

double timer_all __attribute__((weak));

double timer_comm_all __attribute__((weak));
double timer_comm_dir[3] __attribute__((weak));
double timer_comm_recv[3] __attribute__((weak));
double timer_comm_pack[3] __attribute__((weak));
double timer_comm_send[3] __attribute__((weak));
double timer_comm_same[3] __attribute__((weak));
double timer_comm_diff[3] __attribute__((weak));
double timer_comm_bc[3] __attribute__((weak));
double timer_comm_wait[3] __attribute__((weak));
double timer_comm_unpack[3] __attribute__((weak));

double timer_calc_all __attribute__((weak));

double timer_cs_all __attribute__((weak));
double timer_cs_red __attribute__((weak));
double timer_cs_calc __attribute__((weak));

double timer_refine_all __attribute__((weak));
double timer_refine_co __attribute__((weak));
double timer_refine_mr __attribute__((weak));
double timer_refine_cc __attribute__((weak));
double timer_refine_sb __attribute__((weak));
double timer_refine_c1 __attribute__((weak));
double timer_refine_c2 __attribute__((weak));
double timer_refine_sy __attribute__((weak));
double timer_cb_all __attribute__((weak));
double timer_cb_cb __attribute__((weak));
double timer_cb_pa __attribute__((weak));
double timer_cb_mv __attribute__((weak));
double timer_cb_un __attribute__((weak));
double timer_lb_all __attribute__((weak));
double timer_lb_sort __attribute__((weak));
double timer_lb_pa __attribute__((weak));
double timer_lb_mv __attribute__((weak));
double timer_lb_un __attribute__((weak));
double timer_lb_misc __attribute__((weak));
double timer_lb_mb __attribute__((weak));
double timer_lb_ma __attribute__((weak));
double timer_rs_all __attribute__((weak));
double timer_rs_ca __attribute__((weak));
double timer_rs_pa __attribute__((weak));
double timer_rs_mv __attribute__((weak));
double timer_rs_un __attribute__((weak));

double timer_plot __attribute__((weak));

long long total_blocks __attribute__((weak));
num_sz nb_min __attribute__((weak));
num_sz nb_max __attribute__((weak));
int nrrs __attribute__((weak));
int nrs __attribute__((weak));
int nps __attribute__((weak));
int nlbs __attribute__((weak));
int num_refined __attribute__((weak));
int num_reformed __attribute__((weak));
int num_moved_all __attribute__((weak));
int num_moved_lb __attribute__((weak));
int num_moved_rs __attribute__((weak));
int num_moved_coarsen __attribute__((weak));
int num_comm_x __attribute__((weak));
int num_comm_y __attribute__((weak));
int num_comm_z __attribute__((weak));
int num_comm_tot __attribute__((weak));
int num_comm_uniq __attribute__((weak));
int num_comm_x_min __attribute__((weak));
int num_comm_y_min __attribute__((weak));
int num_comm_z_min __attribute__((weak));
int num_comm_t_min __attribute__((weak));
int num_comm_u_min __attribute__((weak));
int num_comm_x_max __attribute__((weak));
int num_comm_y_max __attribute__((weak));
int num_comm_z_max __attribute__((weak));
int num_comm_t_max __attribute__((weak));
int num_comm_u_max __attribute__((weak));
int counter_halo_recv[3] __attribute__((weak));
int counter_halo_send[3] __attribute__((weak));
double size_mesg_recv[3] __attribute__((weak));
double size_mesg_send[3] __attribute__((weak));
int counter_face_recv[3] __attribute__((weak));
int counter_face_send[3] __attribute__((weak));
int counter_bc[3] __attribute__((weak));
int counter_same[3] __attribute__((weak));
int counter_diff[3] __attribute__((weak));
int counter_malloc __attribute__((weak));
double size_malloc __attribute__((weak));
int counter_malloc_init __attribute__((weak));
double size_malloc_init __attribute__((weak));
int total_red __attribute__((weak));

double timer_stencil_total __attribute__((weak));
double timer_stencil_1 __attribute__((weak));
double timer_stencil_2 __attribute__((weak));
double timer_stencil_3 __attribute__((weak)); 
double timer_stencil_4 __attribute__((weak));
double timer_stencil_5 __attribute__((weak)); 
double timer_stencil_6 __attribute__((weak));

#endif