"""
ASE compatible reaction related classes.

Written by Tamas K. Stenczel in 2021,

This file is distributed under the same BSD 3-clause licence
as the rest of the project, attributed to T. K. Stenczel.




Notes on how Refine.py was doing Pathways, which I have tweaked a little here:

Trajectory() creates:
- 208 joined.xyz : opt START - MD steps - opt END
- 165 spaced.xyz : joined.xyz re-spaced in Cartesian Coords

Pathway() is called:
# - opt-init/, opt-final/ : optimisations of the initial and final frame
# - 167 rejoined.xyz : re-joins the path with new optimisations
# - 134 respaced.xyz : re-spaces the path in Cartesians
# <launches FS, not for us though>
- 134 interpolate.in.xyz : same trajectory as before, used as input for Nebterpolate.py
- 134 interpolated.xyz : Nebterpolate.py ran on the trajectory
- command.sh, interpolate.log : Nebterpolate command and log
- 21 interspaced.xyz : re-spaced version of the output of nebterpolation

"""
import os

from .rxndb import Calculation, Optimization, Pathway, Trajectory, make_task


class OptimzationGeomeTRIC(Optimization):
    initial_filename = 'initial.xyz'
    optimized_filename = 'optimize_optim.xyz'
    log_filename = 'optimize.log'

    def launch_task(self):
        # take the optins passed from cli to cli
        opts = self.kwargs.get("geometric_opts")

        make_task(
            f"geometric-optimize {self.initial_filename} --engine ase --prefix=optimize {opts}",
            self.home, inputs=[self.initial_filename], outputs=[self.optimized_filename, self.log_filename],
            tag=self.name, calc=self, verbose=self.verbose)


class PreparePathway(Pathway):
    # skip the optimisation of the ends, we have those anyways
    respaced_filename = "spaced.xyz"

    def launch_fs(self):
        # we are skipping this for this implementation
        pass

    def _create_optimization_object(self, **kwargs):
        return OptimzationGeomeTRIC(**kwargs)

    def launch_gs(self, inter_spaced):
        if not hasattr(self, 'GS'):
            self.GS = GrowingString(inter_spaced, home=os.path.join(self.home, 'GS'),
                                    parent=self, charge=self.charge, mult=self.mult,
                                    stability_analysis=True, priority=self.priority, **self.kwargs)
            self.GS.launch()


class ProcessTrajectory(Trajectory):
    """
    Reactive dynamics trajectory for creating Pathways that can be
    optimised and analysed later. The goal of this subclass is to
    do the bare minimum needed for ML models rather than everything.
    """

    optimize_filename = 'optimize_optim.xyz'

    def _create_optimization_object(self, **kwargs):
        return OptimzationGeomeTRIC(**kwargs)

    def _create_pathway_object(self, *args, **kwargs):
        return PreparePathway(*args, **kwargs)


class GrowingString(Calculation):

    def launch_(self):
        pass
