from graphviz import Digraph
import os
import shutil

from yateto.ast.node import IndexedTensor


class GraphvisHelper():
  def __init__(self, output_dir):
    self.output_dir = output_dir

    # creat a new empty output directory
    if os.path.isdir(self.output_dir):
      shutil.rmtree(self.output_dir)
    os.mkdir(self.output_dir)



  def visit(self, node, graph, counter=0):

    parent_id = str(counter)

    for child in node:
      counter += 1
      children_id = str(counter)


      if isinstance(child, IndexedTensor):
        indices = child.indices.size().keys()
        shape = child.indices.size().values()
        child_name = "{0}[{1}] shape=({2})".format(child.tensor.name(),
                                                   ','.join(map(str, indices)),
                                                   ','.join(map(str, shape)))

      else:
        child_name = "{}; idx={}".format(child.__class__.__name__, children_id)

      graph.node(children_id, child_name)
      graph.edge(parent_id, children_id, constraint='false')

      counter = self.visit(child, graph, counter)
    return counter


  def visualize(self, tree_name, tree_root, is_display=False):

      global_node_counter = 0


      # adjust graphviz
      graph = Digraph(tree_name,
                      filename='{}/{}'.format(self.output_dir, tree_name),
                      engine='circo',
                      format='png')
      graph.graph_attr['rankdir'] = 'LR'

      # generate the root node
      parent_name = "{}; idx={}".format(tree_root.__class__.__name__, str(global_node_counter))
      graph.node(str(global_node_counter), parent_name)

      # traverse a tree
      self.visit(tree_root, graph, global_node_counter)


      # display a result
      graph.render(view=is_display)