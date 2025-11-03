#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>

#include <functional>

namespace pagoma {

template<unsigned int dim>
class MeshManager
{
public:
  using Triangulation = dealii::parallel::distributed::Triangulation<dim, dim>;

  MeshManager(const MPI_Comm mpi_communicator)
    : triangulation(mpi_communicator) {};

  ~MeshManager() = default;

  const Triangulation* get_triangulation() const { return &triangulation; }

  void generate_triangulation(std::function<void(Triangulation&)> generator)
  {
    generator(triangulation);
  }

  void refine(unsigned int refinements)
  {
    triangulation.refine_global(refinements);
  }

  void clear() { triangulation.clear(); }

  void print(std::string& filename) const
  {
    Assert(triangulation.n_cells() > 0,
           dealii::ExcMessage("The triangulation has 0 active cells. Make sure "
                              "it is generated before trying to print."));
    dealii::DataOut<dim, dim> data_out;
    data_out.attach_triangulation(triangulation);
    data_out.build_patches();
    data_out.write_vtu_in_parallel(filename,
                                   triangulation.get_mpi_communicator());
  }

private:
  Triangulation triangulation;
};

template<unsigned int dim>
class MeshBase
{
public:
  virtual ~MeshBase() = default;

  virtual void generate(
    typename MeshManager<dim>::Triangulation& triangulation) = 0;

  template<typename... Args>
  void initialize(Args&&... args)
  {
    do_initialize(std::forward<Args>(args)...);
  }

  virtual void clear() {}

  virtual void print_summary() const = 0;

protected:
  virtual void do_initialize()
  {
    Assert(false, dealii::ExcMessage("Not implemented"));
  }
};

template<unsigned int dim>
class Cube : public MeshBase<dim>
{
public:
  Cube(unsigned int _repetitions = 0,
       double _lower_bound = 0.0,
       double _upper_bound = 0.0)
    : repetitions(_repetitions)
    , lower_bound(_lower_bound)
    , upper_bound(_upper_bound) {};

  void generate(
    typename MeshManager<dim>::Triangulation& triangulation) override
  {
    Assert(triangulation.n_cells() == 0,
           dealii::ExcMessage(
             "The triangulation must have 0 active cells before generating. "
             "Please make sure to call clear beforehand"));
    dealii::GridGenerator::subdivided_hyper_cube(
      triangulation, repetitions, lower_bound, upper_bound);
  }

  void clear() override
  {
    repetitions = 0;
    lower_bound = 0.0;
    upper_bound = 0.0;
  }

  void print_summary() const override
  {
    // TODO:Implement this
  }

private:
  unsigned int repetitions = 0;
  double lower_bound = 0.0;
  double upper_bound = 0.0;
};

template<unsigned int dim>
class Torus : public MeshBase<dim>
{
public:
  Torus(double _center_radius = 0.0,
        double _inner_radius = 0.0,
        unsigned int _n_cells = 6,
        double _phi = 2.0 * dealii::numbers::PI)
    : center_radius(_center_radius)
    , inner_radius(_inner_radius)
    , n_cells(_n_cells)
    , phi(_phi) {};

  void generate(
    typename MeshManager<dim>::Triangulation& triangulation) override
  {
    Assert(triangulation.n_cells() == 0,
           dealii::ExcMessage(
             "The triangulation must have 0 active cells before generating. "
             "Please make sure to call clear beforehand"));
    dealii::GridGenerator::torus(
      triangulation, center_radius, inner_radius, n_cells, phi);
  }

  void clear() override
  {
    center_radius = 0.0;
    inner_radius = 0.0;
    n_cells = 6;
    phi = 2.0 * dealii::numbers::PI;
  }

  void print_summary() const override
  {
    // TODO:Implement this
  }

private:
  double center_radius = 0.0;
  double inner_radius = 0.0;
  unsigned int n_cells = 6;
  double phi = 2.0 * dealii::numbers::PI;
};

}
