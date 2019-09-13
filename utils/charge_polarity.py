#!/usr/bin/python
#Script started on: Tuesday 03 October 2017 02:56:58 AM IST 

import os
import sys
import numpy


#standard usage/error message:
if len(sys.argv) != 2:
    print 'usage: python '+sys.argv[0]+' <sequence>'
    sys.exit()

#definitions:
sequence = sys.argv[1]

#sequence checks (minimum of 3 residues needed to partition into a helix):
if len(sequence) <3:
    print 'error: sequence must possess at least 3 residues'
    sys.exit()

#minimum of 1 polar residue needed to prevent division by zero:
polar = 'STNQDEKRH'
total_polar = 0
for i in sequence:
    if i in polar:
        total_polar += 1

if total_polar == 0:
    print 'error: sequence must possess at least 1 polar residue'
    sys.exit()

#count positive charge (K/R/H residues):
charge_counter = 0
for i in sequence:
    if i in 'KRH':
        charge_counter += 1

#the alpha helix has 3.6 residues per turn, or 1 residue per 100-degrees:
#create an alpha helix projected on an R-theta plane (only theta matters):
helix = []
angle = 0
for i in sequence:
    if i in polar:
        residue = 'polar'
    else:
        residue = 'apolar'
    helix.append([residue, angle%360])
    angle += 100
#partition helix such that epsilon is maximized:
#epsilon = polar residues on face_A/total polar residues
epsilon_max = 0
for angle in range(0,180):
    face_A = float(0)
    limits = list(numpy.sort([angle, (angle+180)%360]))
    for residue in helix:
        if residue[1] >= limits[0] and residue[1] < limits[1]:
            if residue[0] == 'polar':
                face_A += 1
    
    epsilon = max(face_A/total_polar, 1-face_A/total_polar) #maximize epsilon (1)
    epsilon = (epsilon-0.5)*2 #re-scale from 0.5-1 to 0-1 range
    if epsilon > epsilon_max:
        epsilon_max = epsilon #maximize epsilon (2)


print "sequence: "+sequence+", +ve charge: "+str(charge_counter)+", epsilon: "+str(epsilon_max)




























