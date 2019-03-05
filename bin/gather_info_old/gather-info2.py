#!/usr/bin/env python

import os
import subprocess
import sys
from subprocess import PIPE
import re
import pickle
from numpy import abs, array, argsort, linalg, sqrt
import numpy as np
import glob

""" This script loops through the directories of saved runs
and then extracts the energies and coordinates.

LPW note 2019-03-04: This script appears to have been hacked
to read TeraChem outputs that have changed. I don't recommend
using it, but keeping it in case it's needed.
"""

def num_frames(fnm):
    fileName, fileExtension = os.path.splitext(fnm)
    i = -1
    lpf = 1
    nf = 0
    with open(fnm) as f:
        for i, line in enumerate(f):
            if fileExtension == ".xyz" :
                if i == 0:
                    lpf = int(line.strip()) + 2
                pass
            else:
                if 'MD Iteration' in line:
                    nf = int(line.split('MD Iteration')[1].split()[0])+1
    if fileExtension == ".xyz" :
        nl = int(i+1)
        nf = nl / lpf
    return int(nf)

def first_frame(fnm):
    fileName, fileExtension = os.path.splitext(fnm)
    first_frame = -1

    with open(fnm) as f:
        for i, line in enumerate(f):
            if fileExtension == ".xyz" :
                if i == 1:
                    if "generated by terachem" in line:
                        # The frame in the XYZ file starts from zero,
                        # and is one less than the frame in the output file.
                        try:
                            first_frame = int(line.split()[1]) + 1
                            return first_frame
                        except:
                            print "Cannot determine frame number for %s" % fnm
            else:
                if 'MD Iteration' in line:
                    first_frame = int(line.split('MD Iteration')[1].split()[0]) + 1
                    return first_frame
    #print "The words 'MD STEP' were not found in %s" % fnm
    return -1

def is_terachem(fnm):
    for line in open(fnm):
        if "Chemistry at the Speed of Graphics!" in line:
            return 1
    return 0

# Find all of the output files we may have.
cmd = 'find . -maxdepth 2 -name \*.o[0-9ut]* | xargs ls -v'
O = subprocess.Popen(cmd,stdout=PIPE,stderr=PIPE,shell=True)
stepn = 0
stepminus = 1
stepnx = 0

if os.path.exists('.firststep'):
    stepminus = int(open('.firststep').readlines()[0].strip())
    stepminus -= 1
    print "The file .firststep exists, and the first step is %i" % stepminus

Storage = 0

if os.path.exists('info.chk') and Storage:
    print "Pickle exists, loading..."
    f = open('info.chk','r')
    chk = pickle.load(f)
    f.close()
    Energies = chk['E'][:]
    Temperatures = chk['T'][:]
    Coordinates = chk['X'][:]
    BondOrders = chk['BO'][:]
    Charges = chk['Q'][:]
    Spins = chk['S'][:]
    print "%i frames loaded!" % len(Energies)
else:
    Energies = []
    Temperatures = [0.0]
    Coordinates = []
    BondOrders = []
    Charges = []
    Spins = []

def get_crd(onexyz):
    return array([array([float(i) for i in line.split()[1:]]) for line in onexyz[2:]])

def get_dx(xyzi,xyzj):
    na = xyzi.shape[0]
    dx = xyzj-xyzi
    return linalg.norm(dx)/sqrt(na)

def check_coors(fchk, flist):
    """ 
    Perform a comprehensive check to see whether the FIRST frame
    of fchk matches ANY frame of any file in flist. 
    """
    from nanoreactor.molecule import Molecule
    M = Molecule(fchk)[0]
    for fnm in flist:
        M1 = Molecule(fnm)
        print "Checking %s" % fnm
        minmaxdev = 1.0
        minmaxi = 0
        for i in range(len(M1)):
            maxdev = np.abs(np.max(M.xyzs[0] - M1.xyzs[i]))
            print "\rframe %i MaxDev %f" % (i, maxdev),
            if maxdev < 0.1:
                print
                if maxdev < minmaxdev:
                    minmaxdev = maxdev
                    minmaxi = i
        if minmaxdev < 0.1:
            tcframe = int(M1.comms[minmaxi].split()[1])
            print "%s first frame matches %s frame %i" % (fchk, fnm, tcframe)
            return tcframe
        print
    else: # No match
        return 0

fnms = []
xyzfnms = []
fstarts = []
flens = []
fadd = 0
fadds = []

print "Figuring out which .out and .xyz files to concatenate..."
for fnm in O.stdout:
    fnm = fnm.strip()
    if is_terachem(fnm):
        dnm = os.path.split(fnm)[0]
        dframe = first_frame(fnm)
        if dframe == -1: continue
        print "%s is a Terachem output file with at least 1 frame" % fnm
        xyzcand = []
        # Determine whether the .xyz file in this directory matches the output file.
        # This is done by matching the "frame number" in the .xyz file to the "MD STEP" in the output file.
        # Anything that isn't generated by TeraChem is automatically thrown out.
        for xyzfnm in glob.glob(os.path.join(dnm,"*.xyz")):
            if first_frame(xyzfnm) >= 0:
                print xyzfnm, ": output starts at", dframe, "; traj starts at", first_frame(xyzfnm),
            if fadd > 0:
                print "; Offset by %i" % fadd,
            if first_frame(xyzfnm) == dframe:
                print "; Valid"
                xyzcand.append(xyzfnm)
            elif first_frame(xyzfnm) >= 0:
                print "; Invalid"
        for xyzfnm in glob.glob(os.path.join(dnm,"scr","*.xyz")):
            if first_frame(xyzfnm) >= 0:
                print xyzfnm, ": output starts at", dframe, "; traj starts at", first_frame(xyzfnm),
            if fadd > 0:
                print "; Offset by %i" % fadd,
            if first_frame(xyzfnm) == dframe:
                print "; Valid"
                xyzcand.append(xyzfnm)
            elif first_frame(xyzfnm) >= 0:
                print "; Invalid"
        if len(xyzcand) != 1:
            print "Warning: Length of xyzcand for %s is %i (not 1)" % (fnm, len(xyzcand))
            print "The candidates are:", xyzcand
            continue
        xyzfnm = xyzcand[0]
        nfx = num_frames(xyzfnm)
        nfo = num_frames(fnm)
        if dframe == 1 and len(fstarts) > 0:
            print "The frame number went back to 1, checking for coordinate-based restart ...",
            fadd += check_coors(xyzfnm, xyzfnms[::-1])
        if abs(nfo - nfx) >= 5:
            print "Warning: The difference in the number of frames for %s, %s is %i" % (fnm, xyzfnm, abs(nfo - nfx))
            # raw_input()
        flens.append(min(nfo,nfx))
        fnms.append(fnm)
        xyzfnms.append(xyzfnm)
        fstarts.append(first_frame(xyzfnm)+fadd)
        fadds.append(fadd)
    else:
        continue

ordidx = argsort(array(fstarts))
stepminus = fstarts[ordidx[0]]
print "We're using these output file / trajectory pairs:"
print fnms
print xyzfnms
print "The output file / trajectory pairs in this directory have the following start indices and lengths:"
print list(array(fstarts)[ordidx])
print list(array(flens)[ordidx])
print "Subtracting %i from each frame" % stepminus

for fnm, xyzfnm, fadd in [zip(fnms,xyzfnms,fadds)[i] for i in ordidx]:
    print "\nNow working on %s and %s" % (fnm, xyzfnm)
    phile = open(fnm.strip())
    stepf = 0
    for line in phile:
        if 'MD ITERATION' in line:
        # if re.match('\*+ MD STEP',line):
            # sline = line.split()
            # stepf = int(sline[-2])
            stepf = int(line.split('MD Iteration')[1].split()[0]) + 1
            stepn = fadd + stepf - stepminus
        elif re.match('FINAL ENERGY',line):
            sline = line.split()
            while stepn > len(Energies):
                print "warning: stepn = %i (%i in file), len(Energies) = %i" % (stepn, stepf, len(Energies))
                # print "Hmm, we're exceeding the energy array size by %i, appending it (This could be Bad)" % (stepn - len(Energies))
                Energies.append(Energies[-1])
            if stepn == len(Energies):
                Energies.append(float(sline[-2]))
            else:
                if stepn < 0:
                    print "fnm", fnm, "stepn", stepn, "len(Energies)", len(Energies), "line", line
                    print "stepn is negative!"
                    sys.exit(1)
                else:
                    Energies[stepn] = float(sline[-2])
        elif re.search('ESCF.*EKIN.*TEMP.*ETOT',line):
            sline = line.split()
            while stepn > len(Temperatures):
                print "warning: stepn = %i (%i in file), len(Temperatures) = %i" % (stepn, stepf, len(Temperatures))
                # print "Hmm, we're exceeding the temperature array size by %i, appending it (This could be Bad)" % (stepn - len(Temperatures))
                Temperatures.append(Temperatures[-1])
            if stepn == len(Temperatures):
                Temperatures.append(float(sline[14]))
            else:
                if stepn < 0:
                    print "Wtf?"
                    sys.exit(1)
                else:
                    Temperatures[stepn] = float(sline[14])
                    
    coors = open(xyzfnm) # Open the coors.xyz file
    chgfnm = os.path.join(os.path.split(xyzfnm)[0],'charge.xls')
    spnfnm = os.path.join(os.path.split(xyzfnm)[0],'spin.xls')
    bofnm = os.path.join(os.path.split(xyzfnm)[0],'bond_order.list')
    chgfile = open(chgfnm) if os.path.exists(chgfnm) else None
    spnfile = open(spnfnm) if os.path.exists(spnfnm) else None
    bofile = open(bofnm) if os.path.exists(bofnm) else None
    if chgfile != None:
        chgfile.readline()
    if spnfile != None:
        spnfile.readline()
    
    mode = 0
    inserts = []
    for line in coors:
        if mode == 0:
            try:
                natoms = int(line.strip())
            except:
                print xyzfnm, line, line.strip()
            mode = 1
            xyz = [line.strip()]
            atomn = 0
            if chgfile != None:
                thischg = [float(i) for i in chgfile.readline().split()]
            else:
                thischg = [0.0 for i in range(natoms)]
            if spnfile != None:
                thisspn = [float(i) for i in spnfile.readline().split()]
            else:
                thisspn = [0.0 for i in range(natoms)]
            # TeraChem output bond order list
            if bofile != None:
                thisbo = []
                # Read in the number which tells us the number of bond orders
                boline = bofile.readline()
                thisbo.append(boline)
                numbo = int(boline.strip())
                # Read in the comment line and then the bond orders themselves
                for i in range(numbo+1):
                    boline = bofile.readline()
                    thisbo.append(boline)
            else:
                thisbo = ['0', 'No bond orders in this frame']
        elif mode == 1:
            comment = line.strip()
            stepnx = int(line.split()[1]) - (stepminus-1) + fadd
            mode = 2
            xyz.append(comment)
        elif mode == 2:
            xyz.append(line.strip())
            atomn += 1
            if atomn == natoms:
                if stepnx % 100 == 0:
                    print "\rRead in xyz frame %i" % stepnx + " "*54,
                while stepnx > len(Coordinates):
                    print "\rHmm, we're exceeding the energy array size by %i, appending it (This could be Bad)" % (stepnx - len(Coordinates))
                    Coordinates.append(Coordinates[-1])
                    Charges.append(Charges[-1])
                    Spins.append(Spins[-1])
                    BondOrders.append(BondOrders[-1])
                if stepnx == len(Coordinates):
                    Coordinates.append(xyz[:])
                    Charges.append(thischg)
                    Spins.append(thisspn)
                    BondOrders.append(thisbo[:])
                    #print "\r" + " "*54,
                    #if len(Coordinates) >= 2:
                    #print get_dx(get_crd(Coordinates[-1]), get_crd(Coordinates[-2]))
                else:
                    Coordinates[stepnx] = xyz[:]
                    Charges[stepnx] = thischg
                    Spins[stepnx] = thisspn
                    BondOrders[stepnx] = thisbo[:]
                    if stepnx > 0:
                        dx = get_dx(get_crd(Coordinates[stepnx]), get_crd(Coordinates[stepnx-1]))
                    else:
                        dx = 0.0
                    print "\rInserting coordinate into step %i; RMSD = % .4f" % (stepnx,dx),
                    if dx > 0.1:
                        print ".. AHOOGA, RMSD of >1 Angstrom!"
                mode = 0


print len(Coordinates)
print len(Charges)
print len(Spins)
print len(BondOrders)

Eout = open("energies.txt",'w')
for e in Energies:
    print >> Eout, e
Eout.close()
Tout = open("temperatures.txt",'w')
for e in Temperatures:
    print >> Tout, e
Tout.close()
Interval = 1
Xout = open("all-coors.xyz",'w')
QSout = open("charge-spin.txt",'w')
BOout = open("bond-orders.txt",'w')
print
for cn,crd in enumerate(Coordinates):
    if (cn%Interval == 0 and cn < len(Charges) and cn < len(Spins)):
        if len(Charges[cn]) < natoms: continue
        if len(Spins[cn]) < natoms: continue
        if len(crd) < (natoms+2): continue
        if cn%100 == 0:
            print "Writing frame %i\r" % cn,
        for line in crd:
            print >> Xout, line
        print >> QSout, crd[0]
        print >> QSout, "%s : Mulliken Charge, Spin, 0" % crd[1].strip()
        for an in range(natoms):
            print >> QSout, "%2s    % 15.10f %15.10f    0" % (crd[an+2].split()[0], Charges[cn][an], Spins[cn][an])
        if cn < len(BondOrders):
            for line in BondOrders[cn]:
                print >> BOout, line,
Xout.close()
QSout.close()

if Storage:
    print "Writing pickle file, containing %i frames" % len(Energies)
    f=open('info.chk','w')
    pickle.dump({'E':Energies,'T':Temperatures,'X':Coordinates,'Q':Charges,'S':Spins,'B':BondOrders}, f)
    f.close()
                                    
