# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 Fujitsu Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import seaborn as sns

from aif360.algorithms.isf_helpers.isf_utils.common import create_multi_group_label


def plot_intersectionalbias_compare(df_bef, df_aft, vmax=0.8, vmin=0.2, center=0,
                                    title={"right": "before", "left": "after"},
                                    filename=None):
    """
    Compare drawing of intersectional bias in heat map

    Parameters
    ----------
    ds_bef : StructuredDataset
        Dataset containing two sensitive attributes (left figure)
    ds_aft : StructuredDataset
        Dataset containing two sensitive attributes (right figure)
    filename : str, optional
        File name(png)
        e.g. "./result/pict.png"
    metric : str
        Fairness metric name
        ["DisparateImpact"]
    title : dictonary, optional
        Graph title (right figure, left figure)
    """

    gs = GridSpec(1, 2)
    ss1 = gs.new_subplotspec((0, 0))
    ss2 = gs.new_subplotspec((0, 1))

    ax1 = plt.subplot(ss1)
    ax2 = plt.subplot(ss2)

    max_val = df_bef.values.max()
    cmap = mcolors.LinearSegmentedColormap.from_list("red_to_green", ["red", "green"])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    ax1.set_title(title['right'])
    sns.heatmap(df_bef, ax=ax1, vmax=max_val, vmin=vmin, center=center, annot=True, cmap=cmap, norm=norm)

    ax2.set_title(title['left'])
    sns.heatmap(df_aft, ax=ax2, vmax=max_val, vmin=vmin, center=center, annot=True, cmap=cmap, norm=norm)

    if filename is not None:
        plt.savefig(filename, format="png", dpi=300)
    plt.show()
