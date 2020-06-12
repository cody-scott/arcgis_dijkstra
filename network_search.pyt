import arcpy
import os
import json
from collections import deque, namedtuple


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [IndexNetwork, SPDijkstra]


class IndexNetwork(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "IndexNetwork"
        self.description = ""
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""

        param0 = arcpy.Parameter(
            displayName="Line Feature",
            name="line_feature",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )
        param0.filter.list = ["Polyline"]

        param1 = arcpy.Parameter(
            displayName="Output Folder",
            name="output_fld",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        param1.filter.list = ["File System"]

        param2 = arcpy.Parameter(
            displayName="Output Geodatabase",
            name="output_gdb",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input",
        )
        param2.filter.list = ["Local Database"]

        params = [param0, param1, param2]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        arcpy.env.overwriteOutput = True

        feature = parameters[0].valueAsText
        output_folder = parameters[1].valueAsText
        output_gdb = parameters[2].valueAsText

        sr = self.get_sr(feature)
        verts = self.get_verticies(feature, sr)
        vert_feature = self.vertices_to_feature(verts, "in_memory", sr)
        event_points = self.collect_events(vert_feature, output_gdb)
        network_index = self.create_index(feature, event_points, "in_memory")
        self.save_index(network_index, output_folder)
        arcpy.AddMessage("Complete")
        return

    def get_sr(self, feature):
        d_obj = arcpy.Describe(feature)
        return d_obj.spatialReference

    def get_verticies(self, feature, sr):
        out_feats = []
        with arcpy.da.SearchCursor(feature, ["SHAPE@", "OID@"]) as _sc:
            for row in _sc:
                cent = row[0]
                fp = arcpy.PointGeometry(cent.firstPoint, sr)
                lp = arcpy.PointGeometry(cent.lastPoint, sr)
                _id = row[1]
                out_feats.append([fp, _id])
                out_feats.append([lp, _id])
        return out_feats

    def vertices_to_feature(self, vertexes, temp_folder, sr):
        """Generate a feature class of supplied vertices"""
        ft = arcpy.CreateFeatureclass_management(
            temp_folder, "verts", "POINT", spatial_reference=sr)[0]

        arcpy.AddField_management(ft, "BASE_ID", "LONG")
        with arcpy.da.InsertCursor(ft, ["SHAPE@", "BASE_ID"]) as ic:
            for item in vertexes:
                ic.insertRow((item[0],item[1],))
        return ft

    def collect_events(self, feature, out_folder):
        ev = arcpy.CollectEvents_stats(
            feature,
            os.path.join(out_folder, "network_points")
        )[0]
        arcpy.AddField_management(ev, "NET_ID", "LONG")
        with arcpy.da.UpdateCursor(ev, ["NET_ID"]) as uc:
            for ix, row in enumerate(uc):
                row[0] = ix
                uc.updateRow(row)
        return ev

    def create_index(self, feature, collect_points, temp_folder):
        arcpy.AddMessage("Creating Index")
        join_feat = arcpy.SpatialJoin_analysis(
            feature, collect_points, "{}\\join_feats".format(temp_folder), 
            "JOIN_ONE_TO_MANY", "KEEP_ALL",
            search_radius="0 Meters"
        )[0]

        arcpy.AddMessage("Joining")
        index_dct = {}
        with arcpy.da.SearchCursor(join_feat, ["TARGET_FID", "NET_ID", "SHAPE@LENGTH"]) as sc:
            for row in sc:
                c_val = index_dct.get(row[0], {"length": None, "nodes": []})
                c_val['nodes'].append(row[1])
                if c_val['length'] is None:
                    c_val['length'] = row[2]
                index_dct[row[0]] = c_val
        
        arcpy.AddMessage("Building Index")
        net_list = []
        for key in index_dct:
            vals = index_dct[key]

            net_val = []
            if len(vals['nodes']) == 1:
                continue

            net_val = [n for n in vals['nodes'] + [vals['length']]]
            net_list.append(net_val)
        return net_list

    def save_index(self, index_list, output_folder):
        out_data = json.dumps(index_list)
        with open(os.path.join(output_folder, "network_index.json"), 'w') as fl:
            fl.write(out_data)


class SPDijkstra(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Shortest Path Dijkstra"
        self.description = "spDijkstra"
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define parameter definitions"""
        param0 = arcpy.Parameter(
            displayName="Network Index",
            name="network_index",
            datatype="DEFile",
            parameterType="Required",
            direction="Input"
        )
        param1 = arcpy.Parameter(
            displayName="Start Node",
            name="start_node",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )
        param2 = arcpy.Parameter(
            displayName="End Node",
            name="end_node",
            datatype="GPLong",
            parameterType="Required",
            direction="Input"
        )

        param0.filter.list = ["json"]

        params = [param0, param1, param2]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        network_data = parameters[0].valueAsText
        start_node = int(parameters[1].valueAsText)
        end_node = int(parameters[2].valueAsText)

        net = None
        with open(network_data, 'r') as fl:
            net = json.loads(fl.read())

        graph = Graph(net)

        shortest_path = graph.dijkstra(start_node, end_node)
        out_path = "->".join(["{}".format(i) for i in shortest_path])
        arcpy.AddMessage(out_path)
        print(out_path)

        return


class Graph:
    """
    source: https://dev.to/mxl/dijkstras-algorithm-in-python-algorithms-for-beginners-dkc
    """
    def __init__(self, edges):
        self.inf = float('inf')
        self.Edge = namedtuple('Edge', 'start, end, cost')

        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [self.make_edge(*edge) for edge in edges]

        rev_nodes = [[e[1], e[0], e[2]] for e in edges]
        self.edges += [self.make_edge(*edge) for edge in rev_nodes]

    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(self.Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(self.Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def make_edge(self, start, end, cost=1):
        return self.Edge(start, end, cost)

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'
        distances = {vertex: self.inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])
            vertices.remove(current_vertex)
            if distances[current_vertex] == self.inf:
                break
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path
