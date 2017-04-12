import unittest
from tools.tree import *


class TreeTestCase(unittest.TestCase):
  """Tests for `tree.py`."""

  # use tree from synth chars for testing
  digits = TreeNode("digits", leafs=range(2))
  uppercase = TreeNode("uppercase", leafs=range(2, 4))
  lowercase = TreeNode("lowercase", leafs=range(4, 5))
  root = TreeNode("root", children=[digits, uppercase, lowercase])

  tree = TreeStructure(root)

  def test_structure(self):
    self.assertEqual(self.tree.depth, 2)
    self.assertEqual(self.tree.num_labels, 8)
    self.assertEqual(self.tree.num_nodes, 4)
    self.assertListEqual(self.tree.level_sizes, [1,3,5])

    lookedUp = self.tree.lookupMap[3]
    self.assertEqual(lookedUp.dataLabel, 3)
    self.assertEqual(lookedUp.depth, 2)
    self.assertEqual(len(self.tree.lookupMap), 5)

    self.assertListEqual(self.tree.node_sizes, [3, 2, 2, 1])
    self.assertListEqual(self.tree.offsets, [0, 3, 5, 7, 8])

  def test_labels(self):
    middleNode = self.tree.tree.children[1].children[0]
    self.assertEqual(middleNode.name, "leaf2")
    # first split is for top node, second for digits (inactive), third for uppercase
    self.assertListEqual(middleNode.activeGroups, [1, 0, 1, 0])
    self.assertListEqual(middleNode.labels, [1, 0, 0, 0])
    self.assertListEqual(middleNode.walkerLabels, [1, 2])

    middleNode2 = self.tree.tree.children[0].children[0]
    self.assertEqual(middleNode2.name, "leaf0")
    self.assertListEqual(middleNode2.activeGroups, [1, 1, 0, 0])
    self.assertListEqual(middleNode2.labels, [0, 0, 0, 0])
    self.assertListEqual(middleNode2.walkerLabels, [0, 0])

    upperNode = self.tree.tree.children[1]
    self.assertEqual(upperNode.name, "uppercase")
    # first split is for top node, no others yet
    self.assertListEqual(upperNode.activeGroups, [1])
    self.assertListEqual(upperNode.labels, [1])
    self.assertListEqual(upperNode.walkerLabels, [1])

    labels = middleNode.getLabels()
    al = findActiveLabel(labels, self.tree.num_nodes)
    self.assertListEqual(al, [1, 0])

    anotherNode = self.tree.tree.children[1].children[1]
    labels = anotherNode.getLabels()
    al = findActiveLabel(labels, self.tree.num_nodes)
    self.assertListEqual(al, [1, 1])


    labels = middleNode.getLabels()
    al = getWalkerLabel(labels, self.tree.depth, self.tree.num_nodes)
    self.assertListEqual(al, [1, 2])

    anotherNode = self.tree.tree.children[1].children[1]
    labels = anotherNode.getLabels()
    al = getWalkerLabel(labels, self.tree.depth, self.tree.num_nodes)
    self.assertListEqual(al, [1, 3])

  def test_inference(self):
    # len of pred = num_labels = 8 (5+3)
    pred = [0,1,2, 0,1, 0,1, 0]
    nodes, dataLabel = findLabelsFromTree(self.tree, pred)
    self.assertEqual(dataLabel, 4)
    self.assertListEqual(nodes, [2,4])

    pred = [0,1,0, 0,1, 0,1, 0]
    nodes, dataLabel = findLabelsFromTree(self.tree, pred)
    self.assertEqual(dataLabel, 3)
    self.assertListEqual(nodes, [1,3])

    pred = [0,1,0, 0,1, 1,0, 0]
    nodes, dataLabel = findLabelsFromTree(self.tree, pred)
    self.assertEqual(dataLabel, 2)
    self.assertListEqual(nodes, [1,2])



if __name__ == '__main__':
  unittest.main()
