#pragma once

#include <deal.II/base/memory_space.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "dof_manager.h"

namespace pagoma {

template<typename number, typename memory_space>
class SolutionManager
{
public:
  /**
   * @brief Constructor.
   */
  SolutionManager(unsigned int n_blocks)
    : solutions(n_blocks) {};

  /**
   * @brief Reinitialize a solution with ghost indices of a given block index.
   *
   * Note that this involves global communication, so when possible call this
   * once, and copy the parallel layout to other vectors with
   * `reinit(Vector<number, memory_space>)`. For now, this ought to only be
   * called a single time for scalar and vector fields.
   */
  template<unsigned int dim, unsigned int spacedim>
  void reinit(const pagoma::DoFManager<dim, spacedim>& dof_manager,
              MPI_Comm mpi_communicator,
              unsigned int index = 0)
  {
    Assert(index < solutions.n_blocks(),
           dealii::ExcMessage("Index out of range for block vector."));
    solutions.block(index).reinit(dof_manager.get_locally_owned_dofs(),
                                  dof_manager.get_locally_relevant_dofs(),
                                  mpi_communicator);
  }

  /**
   * @brief Reinitialize a solution of a given block index with the same
   * parallel layout as the one given.
   */
  void reinit(
    const dealii::LinearAlgebra::distributed::Vector<number, memory_space>& vec,
    unsigned int index = 0)
  {
    Assert(index < solutions.n_blocks(),
           dealii::ExcMessage("Index out of range for block vector."));
    solutions.block(index).reinit(vec, false);
  }

  /**
   * @brief Reinitialize a solution of a given block index with the same
   * parallel layout as the other given block index.
   *
   * This first index is the solution we are copying from and the second is the
   * one we are copying to.
   */
  void reinit(unsigned int copy_index, unsigned int index = 0)
  {
    Assert(copy_index < solutions.n_blocks(),
           dealii::ExcMessage("Index out of range for block vector."));
    Assert(index < solutions.n_blocks(),
           dealii::ExcMessage("Index out of range for block vector."));
    solutions.block(index).reinit(solutions.block(copy_index), false);
  }

  /**
   * @brief Reinitialze a solution of a given block index with the same parallel
   * layout as the given matrix-free data.
   */
  template<typename matrix_free>
  void reinit(const matrix_free& data, unsigned int index = 0)
  {
    Assert(index < solutions.n_blocks(),
           dealii::ExcMessage("Index out of range for block vector."));
    data.initialize_dof_vector(solutions.block(index));
  }

  /**
   * @brief Get a solution vector of a given block index.
   */
  dealii::LinearAlgebra::distributed::Vector<number, memory_space>&
  get_solution(unsigned int index = 0)
  {
    Assert(index < solutions.n_blocks(),
           dealii::ExcMessage("Index out of range for block vector."));
    return solutions.block(index);
  }

private:
  /**
   * @brief Solution block.
   */
  dealii::LinearAlgebra::distributed::BlockVector<number, memory_space>
    solutions;
};

}
