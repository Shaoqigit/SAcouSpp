// Welcome to Simple Acoustic Simulator
// SAcouS input file

//  comment begin with // should be skip when parsing file
// level 1 begin with #
// level 2 begin with ##
// level 3 begin with ###

# BEGIN ANALYSIS
// analysis type, frequency range
//REDUCTION,10,1000,100
//MODAL,10,10000,100
DIRECT,10,1000,100
# END ANALYSIS

# BEGIN TOPOLOGY
DIMENSION,1
## BEGIN MESH
// could be tensor mesh defined with range*range or explicit mesh with node ID and coordinate definition
// NODES,RANGE
// NODES,LIST
// ELEMENT,RANGE
// ELEMENT,LIST
// ELEMENT,AUTO  (the connectivity of element is computed automatic according to nodes )
// start point coordinate, end_point coordinate, number of node

### BEGIN NODE
RANGE,-1.0,1.0,100
### END NODES
// node ID, coordinate_x
//0,1,-1.0
//1,1,0.0
//1,2,1.0
### BEGIN ELEMENT
// ELEMENT ID, order, connectivity
// if order==NONE, the order will be computed as global interpolation order defined with ORDER parameter
ORDER,2
LIST,
0,NONE,0,1
1,NONE,1,2
2,NONE,2,3
3,NONE,3,4
4,NONE,4,5
5,NONE,5,6
6,NONE,6,7
7,NONE,7,8
8,NONE,8,9
9,NONE,9,10
10,NONE,10,11
11,NONE,11,12
12,NONE,12,13
13,NONE,13,14
14,NONE,14,15
15,NONE,15,16
16,NONE,16,17
17,NONE,17,18
18,NONE,18,19
19,NONE,19,20
20,NONE,20,21
21,NONE,21,22
22,NONE,22,23
23,NONE,23,24
24,NONE,24,25
25,NONE,25,26
26,NONE,26,27
27,NONE,27,28
28,NONE,28,29
29,NONE,29,30
30,NONE,30,31
31,NONE,31,32
32,NONE,32,33
33,NONE,33,34
34,NONE,34,35
35,NONE,35,36
36,NONE,36,37
37,NONE,37,38
38,NONE,38,39
39,NONE,39,40
40,NONE,40,41
41,NONE,41,42
42,NONE,42,43
43,NONE,43,44
44,NONE,44,45
45,NONE,45,46
46,NONE,46,47
47,NONE,47,48
48,NONE,48,49
49,NONE,49,50
50,NONE,50,51
51,NONE,51,52
52,NONE,52,53
53,NONE,53,54
54,NONE,54,55
55,NONE,55,56
56,NONE,56,57
57,NONE,57,58
58,NONE,58,59
59,NONE,59,60
60,NONE,60,61
61,NONE,61,62
62,NONE,62,63
63,NONE,63,64
64,NONE,64,65
65,NONE,65,66
66,NONE,66,67
67,NONE,67,68
68,NONE,68,69
69,NONE,69,70
70,NONE,70,71
71,NONE,71,72
72,NONE,72,73
73,NONE,73,74
74,NONE,74,75
75,NONE,75,76
76,NONE,76,77
77,NONE,77,78
78,NONE,78,79
79,NONE,79,80
80,NONE,80,81
81,NONE,81,82
82,NONE,82,83
83,NONE,83,84
84,NONE,84,85
85,NONE,85,86
86,NONE,86,87
87,NONE,87,88
88,NONE,88,89
89,NONE,89,90
90,NONE,90,91
91,NONE,91,92
92,NONE,92,93
93,NONE,93,94
94,NONE,94,95
95,NONE,95,96
96,NONE,96,97
97,NONE,97,98
98,NONE,98,99
99,NONE,99,100
### END ELEMENT
## END MESH

## BEGIN DOMAIN
// DOMAIN_NAME, DIMENSION, ELEMENT IDs
1,1D,AIR,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
2,1D,RIGID_POROUS,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99
3,0D,INPUT,0
4,0D,MICRO_1,0
## END DOMAIN
# END TOPOLOGY

// material definition
# BEGIN MATERIAL 
// ID,TYPE,NAME,
// PROPERTIES
1,AIR,Classical air,AUTO
// ID,TYPE,NAME,xfm,porosity,flow_resistivity,viscosity,thermal_conductivity
2,RIGID_POROUS,xfm,0.98,3.75E3,1.17,742E-6,110E-6
# END MATERIAL


# BEGIN PHYSIC_DOMAIN
// PHYSIC,DOMAIN_ID,MATERIAL_ID
FLUID,1,1
FLUID,2,2
# END PHYSIC_DOMAIN

# BEGIN BOUNDARY_CONDITION
// BC_ID,TYPE,DOMAIN_ID, VALUE
// fluid velocity on "INPUT" 0D node domain
1,FLUID_VELOCITY,3,-1.0
# END BOUNDARY_CONDITION

# BEGIN SOLVER
// type (NONE, chosen by default)
1,NONE
# END SOLVER

# BEGIN POST_PRO
PRESSURE 
//FLUID_VELOCITY
## END REQUEST_RESULT

## BEGIN FRF
// output node id
// FILE_NAME,TARGET_DOMAIN_IDs
// AUTO,0,50
1,AUTO,4
## END FRF

## BEGIN MAP
//map iD output frequency, if AUTO, all the frequency in analysis will be output
//1,AUTO,500
1,AUTO,500  
## END MAP

# END POST_PRO
