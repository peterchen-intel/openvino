# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.RemoveUselessConcatSplit import RemoveUselessConcatSplitPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class RemoveUselessConcatSplitTests(unittest.TestCase):

    def test_useless_concat_split(self):
        graph = build_graph({'br1': {'kind': 'op', 'op': None},
                             'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'br2': {'kind': 'op', 'op': None},
                             'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'br3': {'kind': 'op', 'op': None},
                             'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'concat': {'kind': 'op', 'op': 'Concat'},
                             'concat_data': {'kind': 'data', 'shape': int64_array([1, 108])},
                             'split': {'kind': 'op', 'op': 'Split'},
                             'split_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'split_br1': {'kind': 'op', 'op': None},
                             'split_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'split_br2': {'kind': 'op', 'op': None},
                             'split_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'split_br3': {'kind': 'op', 'op': None},
                             },
                            [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                             ('br_data_1', 'concat', {'in': 0}),
                             ('br_data_2', 'concat', {'in': 1}),
                             ('br_data_3', 'concat', {'in': 2}),
                             ('concat', 'concat_data'),
                             ('concat_data', 'split'),
                             ('split', 'split_data_1', {'out': 0}),
                             ('split', 'split_data_2', {'out': 1}),
                             ('split', 'split_data_3', {'out': 2}),
                             ('split_data_1', 'split_br1'),
                             ('split_data_2', 'split_br2'),
                             ('split_data_3', 'split_br3')])
        RemoveUselessConcatSplitPattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'br1': {'kind': 'op', 'op': None},
                                 'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                                 'br2': {'kind': 'op', 'op': None},
                                 'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                                 'br3': {'kind': 'op', 'op': None},
                                 'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                                 'split_br1': {'kind': 'op', 'op': None},
                                 'split_br2': {'kind': 'op', 'op': None},
                                 'split_br3': {'kind': 'op', 'op': None}},
                                [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                                 ('br_data_1', 'split_br1'),
                                 ('br_data_2', 'split_br2'),
                                 ('br_data_3', 'split_br3'),
                                 ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'split_br3')
        self.assertTrue(flag, resp)

    def test_usefull_concat_split(self):
        graph = build_graph({'br1': {'kind': 'op', 'op': None},
                             'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'br2': {'kind': 'op', 'op': None},
                             'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'br3': {'kind': 'op', 'op': None},
                             'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'concat': {'kind': 'op', 'op': 'Concat'},
                             'concat_data': {'kind': 'data', 'shape': int64_array([1, 108])},
                             'split': {'kind': 'op', 'op': 'Split'},
                             'split_data_1': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'split_br1': {'kind': 'op', 'op': None},
                             'split_data_2': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'split_br2': {'kind': 'op', 'op': None},
                             'split_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'split_br3': {'kind': 'op', 'op': None},
                             },
                            [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                             ('br_data_1', 'concat', {'in': 0}),
                             ('br_data_2', 'concat', {'in': 1}),
                             ('br_data_3', 'concat', {'in': 2}),
                             ('concat', 'concat_data'),
                             ('concat_data', 'split'),
                             ('split', 'split_data_1', {'out': 0}),
                             ('split', 'split_data_2', {'out': 1}),
                             ('split', 'split_data_3', {'out': 2}),
                             ('split_data_1', 'split_br1'),
                             ('split_data_2', 'split_br2'),
                             ('split_data_3', 'split_br3')])
        RemoveUselessConcatSplitPattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'br1': {'kind': 'op', 'op': None},
                                 'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                                 'br2': {'kind': 'op', 'op': None},
                                 'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                                 'br3': {'kind': 'op', 'op': None},
                                 'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data', 'shape': int64_array([1, 108])},
                                 'split': {'kind': 'op', 'op': 'Split'},
                                 'split_data_1': {'kind': 'data', 'shape': int64_array([1, 36])},
                                 'split_br1': {'kind': 'op', 'op': None},
                                 'split_data_2': {'kind': 'data', 'shape': int64_array([1, 26])},
                                 'split_br2': {'kind': 'op', 'op': None},
                                 'split_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                                 'split_br3': {'kind': 'op', 'op': None},
                                 },
                                [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                                 ('br_data_1', 'concat', {'in': 0}),
                                 ('br_data_2', 'concat', {'in': 1}),
                                 ('br_data_3', 'concat', {'in': 2}),
                                 ('concat', 'concat_data'),
                                 ('concat_data', 'split'),
                                 ('split', 'split_data_1', {'out': 0}),
                                 ('split', 'split_data_2', {'out': 1}),
                                 ('split', 'split_data_3', {'out': 2}),
                                 ('split_data_1', 'split_br1'),
                                 ('split_data_2', 'split_br2'),
                                 ('split_data_3', 'split_br3')])

        (flag, resp) = compare_graphs(graph, ref_graph, 'split_br3')
        self.assertTrue(flag, resp)

    def test_useful_concat_2_outputs_split(self):
        graph = build_graph({'br1': {'kind': 'op', 'op': None},
                             'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'br2': {'kind': 'op', 'op': None},
                             'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'br3': {'kind': 'op', 'op': None},
                             'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'concat': {'kind': 'op', 'op': 'Concat'},
                             'concat_data': {'kind': 'data', 'shape': int64_array([1, 108])},
                             'placeholder': {'kind': 'op', 'op': None},
                             'split': {'kind': 'op', 'op': 'Split'},
                             'split_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'split_br1': {'kind': 'op', 'op': None},
                             'split_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'split_br2': {'kind': 'op', 'op': None},
                             'split_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'split_br3': {'kind': 'op', 'op': None},
                             },
                            [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                             ('br_data_1', 'concat', {'in': 0}),
                             ('br_data_2', 'concat', {'in': 1}),
                             ('br_data_3', 'concat', {'in': 2}),
                             ('concat', 'concat_data'),
                             ('concat_data', 'split'),
                             ('concat_data', 'placeholder'),
                             ('split', 'split_data_1', {'out': 0}),
                             ('split', 'split_data_2', {'out': 1}),
                             ('split', 'split_data_3', {'out': 2}),
                             ('split_data_1', 'split_br1'),
                             ('split_data_2', 'split_br2'),
                             ('split_data_3', 'split_br3')])
        RemoveUselessConcatSplitPattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'br1': {'kind': 'op', 'op': None},
                                 'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                                 'br2': {'kind': 'op', 'op': None},
                                 'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                                 'br3': {'kind': 'op', 'op': None},
                                 'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_data': {'kind': 'data', 'shape': int64_array([1, 108])},
                                 'placeholder': {'kind': 'op', 'op': None},
                                 'split': {'kind': 'op', 'op': 'Split'},
                                 'split_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                                 'split_br1': {'kind': 'op', 'op': None},
                                 'split_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                                 'split_br2': {'kind': 'op', 'op': None},
                                 'split_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                                 'split_br3': {'kind': 'op', 'op': None},
                                 },
                                [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                                 ('br_data_1', 'concat', {'in': 0}),
                                 ('br_data_2', 'concat', {'in': 1}),
                                 ('br_data_3', 'concat', {'in': 2}),
                                 ('concat', 'concat_data'),
                                 ('concat_data', 'split'),
                                 ('concat_data', 'placeholder'),
                                 ('split', 'split_data_1', {'out': 0}),
                                 ('split', 'split_data_2', {'out': 1}),
                                 ('split', 'split_data_3', {'out': 2}),
                                 ('split_data_1', 'split_br1'),
                                 ('split_data_2', 'split_br2'),
                                 ('split_data_3', 'split_br3')])

        (flag, resp) = compare_graphs(graph, ref_graph, 'split_br3')
        self.assertTrue(flag, resp)

    def test_useless_concat_split_2_outputs(self):
        graph = build_graph({'br1': {'kind': 'op', 'op': None},
                             'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'br2': {'kind': 'op', 'op': None},
                             'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'br3': {'kind': 'op', 'op': None},
                             'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'concat': {'kind': 'op', 'op': 'Concat'},
                             'concat_data': {'kind': 'data', 'shape': int64_array([1, 108])},
                             'split': {'kind': 'op', 'op': 'Split'},
                             'split_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                             'split_br1': {'kind': 'op', 'op': None},
                             'split_br1_1': {'kind': 'op', 'op': None},
                             'split_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                             'split_br2': {'kind': 'op', 'op': None},
                             'split_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                             'split_br3': {'kind': 'op', 'op': None},
                             },
                            [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                             ('br_data_1', 'concat', {'in': 0}),
                             ('br_data_2', 'concat', {'in': 1}),
                             ('br_data_3', 'concat', {'in': 2}),
                             ('concat', 'concat_data'),
                             ('concat_data', 'split'),
                             ('split', 'split_data_1', {'out': 0}),
                             ('split', 'split_data_2', {'out': 1}),
                             ('split', 'split_data_3', {'out': 2}),
                             ('split_data_1', 'split_br1'),
                             ('split_data_1', 'split_br1_1'),
                             ('split_data_2', 'split_br2'),
                             ('split_data_3', 'split_br3')])
        RemoveUselessConcatSplitPattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'br1': {'kind': 'op', 'op': None},
                                 'br_data_1': {'kind': 'data', 'shape': int64_array([1, 26])},
                                 'br2': {'kind': 'op', 'op': None},
                                 'br_data_2': {'kind': 'data', 'shape': int64_array([1, 36])},
                                 'br3': {'kind': 'op', 'op': None},
                                 'br_data_3': {'kind': 'data', 'shape': int64_array([1, 46])},
                                 'split_br1': {'kind': 'op', 'op': None},
                                 'split_br1_1': {'kind': 'op', 'op': None},
                                 'split_br2': {'kind': 'op', 'op': None},
                                 'split_br3': {'kind': 'op', 'op': None}},
                                [('br1', 'br_data_1'), ('br2', 'br_data_2'), ('br3', 'br_data_3'),
                                 ('br_data_1', 'split_br1'),
                                 ('br_data_1', 'split_br1_1'),
                                 ('br_data_2', 'split_br2'),
                                 ('br_data_3', 'split_br3'),
                                 ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'split_br3')
        self.assertTrue(flag, resp)
