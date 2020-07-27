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

#include <mpi.h>
#include <math.h>
#include <CL/sycl.hpp>
#include <CL/sycl/intel/fpga_extensions.hpp>

#include "block.h"
#include "comm.h"
#include "proto.h"
#include "timer.h"

class Stencil_kernel;
using namespace cl;

//Device selection
//We will explicitly compile for the FPGA_EMULATOR, CPU_HOST, or FPGA
#if defined(FPGA_EMULATOR)
sycl::intel::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
sycl::host_selector device_selector;
#else
sycl::intel::fpga_selector device_selector;
#endif

//Create queue
auto property_list = sycl::property_list{ sycl::property::queue::enable_profiling() };
sycl::queue device_queue(device_selector, NULL, property_list);
sycl::event queue_event;


void fpga_kernel(sycl::buffer<double, 1>& input_buffer, sycl::buffer<double, 1>& output_buffer)
{
   //Device queue submit
   queue_event = device_queue.submit([&](sycl::handler& cgh) {

      //sycl::stream os(1024, 1024, cgh);

      //Create accessors
      auto accessor_in = input_buffer.get_access<sycl::access::mode::read_write>(cgh);
      auto accessor_out = output_buffer.get_access<sycl::access::mode::discard_write>(cgh);

      cgh.single_task<class Stencil_kernel>([=]() {

         double local_array[12][12][12];
         //double sb, sm, sf;

         //sycl::stream os(1024,128,cgh);
         for (int var = 0; var < 40; var++) {

            //create a local copy of the array data for increased performance
            for (int i = 0; i <= 11; i++)
               for (int j = 0; j <= 11; j++)
                  for (int k = 0; k <= 11; k++)
                     local_array[i][j][k] = accessor_in[(var * (12) * (12) * (12)) + (k + (12) * (j + (12) * i))];

            for (int i = 1; i <= 10; i++)
               for (int j = 1; j <= 10; j++)
                  for (int k = 1; k <= 10; k++) {
                     accessor_out[(var * (12) * (12) * (12)) + (k + (12) * (j + (12) * i))] = (
                     local_array[i - 1][j - 1][k - 1] +
                     local_array[i - 1][j - 1][k] +
                     local_array[i - 1][j - 1][k + 1] +
                     local_array[i - 1][j][k - 1] +
                     local_array[i - 1][j][k] +
                     local_array[i - 1][j][k + 1] +
                     local_array[i - 1][j + 1][k - 1] +
                     local_array[i - 1][j + 1][k] +
                     local_array[i - 1][j + 1][k + 1] +
                     local_array[i][j - 1][k - 1] +
                     local_array[i][j - 1][k] +
                     local_array[i][j - 1][k + 1] +
                     local_array[i][j][k - 1] +
                     local_array[i][j][k] +
                     local_array[i][j][k + 1] +
                     local_array[i][j + 1][k - 1] +
                     local_array[i][j + 1][k] +
                     local_array[i][j + 1][k + 1] +
                     local_array[i + 1][j - 1][k - 1] +
                     local_array[i + 1][j - 1][k] +
                     local_array[i + 1][j - 1][k + 1] +
                     local_array[i + 1][j][k - 1] +
                     local_array[i + 1][j][k] +
                     local_array[i + 1][j][k + 1] +
                     local_array[i + 1][j + 1][k - 1] +
                     local_array[i + 1][j + 1][k] +
                     local_array[i + 1][j + 1][k + 1]) / 27;
                  }
         }



         //for (int i = 1; i <= 10; i++)
         //   for (int j = 1; j <= 10; j++)
         //      for (int k = 1; k <= 10; k++) {
         //         //os<< "i: " << i << "--" << "j: " << j << "--" << "k: " << k << sycl::endl;
         //         //os<< "Work: " << work[i][j][k] << "--" << "Array: " << _array[i][j][k] << sycl::endl;
         //         _array[i][j][k] = work[i][j][k];
         //      }
         });

      });

}