import itertools
import numpy as np

class TreeNode:
  name = "node"
  children = []
  isLeaf = True
  labelId = 0  # label for training (might differ to datalabel when a node has both
               #  subnodes and leafs as children)
  dataLabelId = 0 # actual label in the dataset
  levelLabelId = 0 # index of node in all nodes of same hierarchy level (depth) in tree

  depth = 0 # hierarchy level in 3, root has 0 depth

  labels = []  # used later as label
  walkerLabels = [] # all levelLabels until here
  activeGroups = []

  def __init__(self, name, children=None, leafs=None, labelId=0):
    self.name = name

    if children is not None:
      self.children = children
      start = 0

      # assign labels to all children
      for child in self.children:
        child.labelId = start
        start = start + 1


    if leafs is not None:
      for leaf in leafs:
        self.addChild(TreeNode("leaf"+str(leaf), labelId=leaf))

    self.isLeaf = (len(self.children) == 0)
    self.labelId = labelId
    self.dataLabelId = labelId

  def addChild(self, node):
    node.labelId = len(self.children)
    self.children = self.children + [node]
    self.isLeaf = False

  def print(self):
    result = str(self.labelId) + ","\
             + str(self.dataLabelId) + ","\
             + str(self.levelLabelId) + ","\
             + str(len(self.children)) + ","
    if not self.isLeaf:
      result = "\n" + result + "\n"

    for child in self.children:
      result = result + child.print()

    return result

  # create indices for all nodes on a 'level' of the tree
  # assumes node is root
  def computeLevelIndices(self):
    nodes = [self]
    current_depth = 1

    while len(nodes) > 0:
      children = []
      for node in nodes:
        children = children + node.children

      child_index = 0
      for child in children:
        child.depth = current_depth
        child.levelLabelId = child_index
        child_index = child_index + 1

      nodes = children

      current_depth = current_depth + 1


  # pass through the tree and assign labels to all leafs
  def assignLabels(self):
    nodes = [self]

    while len(nodes) > 0:
      children = []

      labels = [0] * len(nodes)
      groupFlags = [0] * len(nodes)
      nodeIndex = 0

      for node in nodes:
        children = children + node.children
        for child in node.children:
          childLabels = labels.copy()
          childLabels[nodeIndex] = child.labelId

          childGroupFlags = groupFlags.copy()
          childGroupFlags[nodeIndex] = 1

          child.labels = node.labels + childLabels
          child.walkerLabels = node.walkerLabels + [child.levelLabelId]
          child.activeGroups = node.activeGroups + childGroupFlags

        nodeIndex = nodeIndex + 1

      nodes = children


  def getLeafs(self):
    if self.isLeaf:
      return [self]
    else:
      leafs = [c.getLeafs() for c in self.children]
      return [item for l in leafs for item in l] #flatten list

  def getNodes(self):
    if self.isLeaf:
      return None

    nodes = [self]
    for child in self.children:
      if not child.isLeaf:
        nodes = nodes + child.getNodes()
    return nodes

  def getDepth(self):
    maxd = 0
    for leaf in self.getLeafs():
      if leaf.depth > maxd:
        maxd = leaf.depth
    return maxd


def createLabelNodeMap(tree, num_labels):
  """ creates a hash map from labels to tree nodes, so that lookup is cheap"""
  map = [0] * num_labels
  for leaf in tree.getLeafs():
    map[leaf.dataLabelId] = leaf

  return map


class TreeStructure:
  def __init__(self, tree):
    tree.computeLevelIndices()
    tree.assignLabels()

    a = tree.getNodes()
    self.num_nodes = len(tree.getNodes())
    self.node_sizes = [len(n.children) for n in tree.getNodes()]

    self.offsets = [0]+list(itertools.accumulate(self.node_sizes))
    self.num_labels = self.offsets[-1] # number of all labels (NUM_LABELS + NUM_NODES -1)
    self.lookupMap = createLabelNodeMap(tree, self.num_labels)
    self.depth = tree.getDepth()
    self.tree = tree


def findLabelFromTree(tree, pred):
  node = tree.tree

  while not node.isLeaf:
    gi = len(node.labels)+node.levelLabelId
    children_preds = pred[tree.offsets[gi]:(tree.node_sizes[gi]+tree.offsets[gi])]
    next_child_ind = np.argmax(children_preds)

    node = node.children[next_child_ind]

  return node.dataLabelId
