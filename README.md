# ArcGIS Dijkstra
Dijkstra shortest path python toolbox for ArcGIS

## Generate index of a line feature class
This will output a point feature class with the ID and a json file to be used in the network search

## Run Search Tool
Use index json file as input and node IDs from the point feature class as start/end nodes


## Network Errors
problem with spatial join tool that connects points in very close proximity even when search radius is 0. Likely this indicates a network problem that should be corrected anyways.
