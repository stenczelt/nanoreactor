"""
ASE compatible reaction related classes.

Written by Tamas K. Stenczel in 2021,

This file is distributed under the same BSD 3-clause licence
as the rest of the project, attributed to T. K. Stenczel.
"""

from .rxndb import Pathway, Trajectory


class PreparePathway(Pathway):
    def launch_fs(self):
        # we are skipping this for this implementation
        pass


class ProcessTrajectory(Trajectory):
    """
    Reactive dynamics trajectory for creating Pathways that can be
    optimised and analysed later. The goal of this subclass is to
    do the bare minimum needed for ML models rather than everything.
    """

    def _create_pathway_object(self, *args, **kwargs):
        return PreparePathway(*args, **kwargs)
