#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <memory>

namespace pagoma {
class Timer
{
public:
  Timer(const MPI_Comm mpi_communicator)
  {
    scoped_timer = std::make_unique<dealii::TimerOutput>(
      mpi_communicator,
      std::cout,
      dealii::TimerOutput::OutputFrequency::summary,
      dealii::TimerOutput::OutputType::cpu_and_wall_times_grouped);
  }

  void start_section(const std::string& section)
  {
    scoped_timer->enter_subsection(section);
  }

  void end_section(const std::string& section)
  {
    scoped_timer->leave_subsection(section);
  }

private:
  std::unique_ptr<dealii::TimerOutput> scoped_timer;
};
}
