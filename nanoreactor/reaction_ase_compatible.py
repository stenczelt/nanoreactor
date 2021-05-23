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
from copy import deepcopy

import os

from .molecule import Molecule
from .rxndb import Calculation, Optimization, Pathway, Trajectory, make_task


class OptimizationGeomeTRIC(Optimization):
    initial_filename = 'initial.xyz'
    optimized_filename = 'optimize_optim.xyz'
    log_filename = 'optimize.log'

    def launch_task(self):
        # take the options passed from cli to cli
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
        return OptimizationGeomeTRIC(**kwargs)

    def launch_gs(self, inter_spaced):
        if not hasattr(self, 'GS'):
            self.GS = GrowingString(inter_spaced, home=os.path.join(self.home, 'GS'),
                                    parent=self, charge=self.charge, mult=self.mult,
                                    stability_analysis=True, priority=self.priority,
                                    use_path_guess=True,
                                    **self.kwargs)
            self.GS.launch()

        if not hasattr(self, 'GS-ends-only'):
            self.GS_ends_only = GrowingString(inter_spaced, home=os.path.join(self.home, 'GS-ends-only'),
                                              parent=self, charge=self.charge, mult=self.mult,
                                              stability_analysis=True, priority=self.priority,
                                              use_path_guess=False,
                                              **self.kwargs)
            self.GS_ends_only.launch()


class ProcessTrajectory(Trajectory):
    """
    Reactive dynamics trajectory for creating Pathways that can be
    optimised and analysed later. The goal of this subclass is to
    do the bare minimum needed for ML models rather than everything.
    """

    optimize_filename = 'optimize_optim.xyz'

    def _create_optimization_object(self, **kwargs):
        return OptimizationGeomeTRIC(**kwargs)

    def _create_pathway_object(self, *args, **kwargs):
        return PreparePathway(*args, **kwargs)


class GrowingString(Calculation):
    """Growing string, can represent both one with a guessed path and one with no guess of the path"""

    calctype = "GrowingString"

    filename_log = "gsm.log"
    filename_initial = "initial.xyz"

    def launch_(self):

        if os.path.exists(os.path.join(self.home, self.filename_log)) and os.path.exists(
                os.path.join(self.home, "opt_converged_000.xyz")):
            # calculation is done
            self.saveStatus("complete")

        else:
            # set up the calculations
            if not isinstance(self.initial, Molecule):
                mol = Molecule(self.initial)
            else:
                mol = deepcopy(self.initial)

            # write the starting configs
            restart_str = ""
            if self.kwargs.get("use_path_guess", True):
                # use the guess of path given
                mol.write(os.path.join(self.home, self.filename_initial))
                restart_str = f"-restart_file {self.filename_initial} -reparametrize"
            else:
                # only write start and end
                mol_ends = mol[0] + mol[-1]
                mol_ends.write(os.path.join(self.home, self.filename_initial))

            # the ASE calculator options are the same for this as for geomeTRIC
            opts = self.kwargs.get("geometric_opts")

            # launch the task
            make_task(
                f"gsm -xyzfile {self.filename_initial} {restart_str} -mode DE_GSM -num_nodes {self.images} {opts}"
                f" -ID 0 -coordinate_type TRIC -package ase -max_gsm_iters {self.gsmax} > {self.filename_log}",
                self.home, inputs=["initial.xyz"],
                outputs=["0000_string.png", "IC_data_0000.txt", "opt_converged_000.xyz", "TSnode_0.xyz",
                         self.filename_log],
                tag=self.name, calc=self, verbose=self.verbose)
