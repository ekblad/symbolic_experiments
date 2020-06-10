from pygraphviz import AGraph

nodes1 = ['IN1','IN2','IN3']
nodes2 = ['MUL','ADD']#,'Sum']
colors = ['midnightblue','goldenrod']
# nodes.reverse()
edges_gp_inputs = [('IN1','MUL'),('IN2','MUL'),('IN3','ADD')]
edges_gp_funcs = [('MUL','ADD'),]
# edges_gp.reverse()
# edges_nn = [('Input 1','ADD'),('Input 2','ADD'),('Input 3','MUL'),('ADD','Sum'),('MUL','Sum')]

labels = {'IN1':'IN1','IN2':'IN2','IN3':'IN3','MUL':'MUL','ADD':'ADD','Sum':'Sum'}

g = AGraph(strict=False,directed=True)
g.graph_attr['overlap'] = False
g.graph_attr['size'] = '3,3'
g.graph_attr['dpi'] = 100
g.graph_attr['fontname'] = 'serif'
g.node_attr['fontsize'] = 30
g.node_attr['fontname'] = 'helvetica'
g.node_attr['penwidth'] = 3
g.edge_attr['penwidth'] = 5
# g.node_attr['pencolor'] = colors[1]
# g.edge_attr['pencolor'] = colors[0]
g.add_nodes_from(nodes1,color=colors[0])
g.add_edges_from(edges_gp_inputs,color=colors[0],arrowsize=0.1)
g.add_nodes_from(nodes2,color=colors[1])
g.add_edges_from(edges_gp_funcs,color=colors[0],arrowsize=0.1)
# g.add_edges_from(edges_nn,color='blue')
# g.size(5)
g.layout(prog="dot")
# g.graph_attr.update(name=settings)

for i in nodes1+nodes2:
	if i in nodes1:
		n = g.get_node(i)
		n.attr["label"] = labels[i]
		n.attr["fontcolor"] = colors[0]
		n.attr["fillcolor"] = colors[1]
		n.attr["color"] = colors[1]
	if i in nodes2:
		n = g.get_node(i)
		n.attr["label"] = labels[i]
		n.attr["fontcolor"] = colors[1]
		n.attr["fillcolor"] = colors[0]
		n.attr["color"] = colors[0]

g.draw('gp_tree_final.pdf',format='pdf')