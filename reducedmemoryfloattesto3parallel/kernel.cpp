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
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "block.h"
#include "comm.h"
#include "proto.h"
#include "timer.h"

class Stencil_kernel;
using namespace cl;

//Device selection
//We will explicitly compile for the FPGA_EMULATOR, CPU_HOST, or FPGA
#if defined(FPGA_EMULATOR)
sycl::INTEL::fpga_emulator_selector device_selector;
#elif defined(CPU_HOST)
sycl::host_selector device_selector;
#else
sycl::INTEL::fpga_selector device_selector;
#endif

//Create queue
auto property_list = sycl::property_list{ sycl::property::queue::enable_profiling() };
sycl::queue device_queue(device_selector, NULL, property_list);
sycl::event queue_event;

//create a buffer that goes to the fpga
//double* arrayForBuffer = new double[(x_block_size + 2) * (y_block_size + 2) * (z_block_size + 2)];

//Buffer setup
//Define the sizes of the buffers
//The sycl buffer creation expects a type of sycl:: range for the size
//sycl::range<3> num_array{ static_cast<size_t>(x_block_size + 2) , static_cast<size_t>(y_block_size + 2) , static_cast<size_t>(z_block_size + 2) };

//sycl::buffer<double, 3> array_buf(arrayForBuffer, num_array);

//auto buf_array = array_buf.get_access<sycl::access::mode::read_write>();


void fpga_kernel(sycl::buffer<float, 1>& input_buffer, sycl::buffer<float, 1>& output_buffer)
{
   //Device queue submit
   queue_event = device_queue.submit([&](sycl::handler& cgh) {

      //sycl::stream os(1024, 1024, cgh);

      //Create accessors
      auto accessor_in = input_buffer.get_access<sycl::access::mode::read>(cgh);
      auto accessor_out = output_buffer.get_access<sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for(sycl::range<1> (40), [=](sycl::id<1> idx) {

        for (int i = 1; i <= 10; i++)
           for (int j = 1; j <= 10; j++)
              for (int k = 1; k <= 10; k++)
                 accessor_out[(idx * (12) * (12) * (12)) + (k + (12) * (j + (12) * i))] = (
                    accessor_in[(idx * (12) * (12) * (12)) + (k + (12) * (j + (12) * (i - 1)))] +
                    accessor_in[(idx * (12) * (12) * (12)) + (k + (12) * ((j-1) + (12) * i))] +
                    accessor_in[(idx * (12) * (12) * (12)) + ((k-1) + (12) * (j + (12) * i))] +
                    accessor_in[(idx * (12) * (12) * (12)) + (k + (12) * (j + (12) * i))] +
                    accessor_in[(idx * (12) * (12) * (12)) + ((k+1) + (12) * (j + (12) * i))] +
                    accessor_in[(idx * (12) * (12) * (12)) + (k + (12) * ((j+1) + (12) * i))] +
                    accessor_in[(idx * (12) * (12) * (12)) + (k + (12) * (j + (12) * (i + 1)))]) / 7.0;

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