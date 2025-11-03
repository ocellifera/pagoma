#pragma once

#include <deal.II/base/memory_space.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/read_write_vector.h>

#include <mpi.h>

#include "constraint_manager.h"
#include "dof_manager.h"

namespace pagoma {

template<unsigned int dim, typename VectorType>
class VectorOutput
{
public:
  using RealType = typename VectorType::value_type;
  using MemorySpace = typename VectorType::memory_space;

  VectorOutput(const VectorType& vec,
               const DoFManager<dim>& dof_manager,
               unsigned int degree,
               const std::string& filename_without_extension,
               unsigned int increment,
               MPI_Comm mpi_communicator)
  {
    // Update ghosts
    vec.update_ghost_values();

    // Init data out & add vectors
    // TODO: Make this more general to handle block vectors.
    dealii::DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_manager.get_dof_handler());
    data_out.add_data_vector(vec, filename_without_extension);
    data_out.build_patches(degree);

    dealii::DataOutBase::VtkFlags flags;
    flags.compression_level = dealii::DataOutBase::CompressionLevel::best_speed;
    data_out.set_flags(flags);
    data_out.write_vtu_with_pvtu_record(
      "./", filename_without_extension, increment, mpi_communicator, 6);
  };
};

}
