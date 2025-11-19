import numpy as np
import copy
from sklearn.linear_model import LinearRegression

def is_minimal(s,plausibleS):
    """
    Given a set of plausible sets we check if s is minimal within that set.
    :set s: specific set we check for minimality
    :list plausibleS: set of plausible sets.
    """
    
    for t in plausibleS:
        if s==t:
            pass
        else:
            if set(t).issubset(set(s)):
                return False
    return True


def calculate_partial_correlation(covariate,target,lags,subset=[]):
    """
    Calculate the partial correlation coefficients of the lagged covariates against the target, conditioned on 
    subset with the same lags.

    Parameters:
        covariate: A numpy vector with the covariate data
        target: A numpy array containing the tarbet data
        subset: A numpy array containing the subset to condition on
        

    Returns: a partial correlation matrix of, where columns correspond to different lags and rows to different target
    vectors.
    """
    # Initialize the linear regression model
    lr = LinearRegression()
    m=max(lags)
    covariate=copy.copy(np.hstack([covariate[m-i:-i] for i in lags]))
    target=copy.copy(target[m:])
    if len(subset)>0:
        subset=copy.copy(np.hstack([subset[m-i:-i] for i in lags]))
        cov_residual = covariate - lr.fit(subset, covariate).predict(subset)
        target_residual = target - lr.fit(subset, target).predict(subset)
    else:

        cov_residual = covariate
        target_residual = target
    if np.prod(target_residual.shape) == (target_residual).shape[0]:
        target_residual=np.reshape(target_residual,(len(target_residual),1))
    lagleng=len(lags)
    corr=np.corrcoef(np.hstack([cov_residual,target_residual]).T)

    return corr[:lagleng,lagleng:]

def match_time_series(data,source_timestamps,target_timestamps):
    """
    A generic function to up/downsample entries of 'data' with time indication 'source_timestamps' to match
    the time index 'target_timestamps'.

    Parameters:
        data (np.array): The data entries we want to up/downsample.
        source_timestamps (np.array): Timestamps for the input data.
        target_timestamps (np.array): Timestamps we want to match.

    Returns:
        np.array: New data array matched in length to target_timestamps and filled according to the closest times.
    """
    # Ensure the inputs are numpy arrays
    target_timestamps = np.asarray(target_timestamps)
    source_timestamps = np.asarray(source_timestamps)
    data = np.asarray(data)
    
    # Create an output array for y values that matches the length of x
    matched_data = np.zeros_like(target_timestamps)

    # Find the closest source_timestamps for each entry in target_timestamps using np.searchsorted
    idx = np.searchsorted(source_timestamps, target_timestamps, side='left')

    # Adjust indices that are out of bounds (past the last source_timestamps)
    for i in range(len(target_timestamps)):
        if idx[i] == len(source_timestamps):  # If index equals the length of source_timestamps, use the last value in y
            idx[i] -= 1
        elif idx[i] > 0 and (target_timestamps[i] - source_timestamps[idx[i]-1]) <= (source_timestamps[idx[i]] - target_timestamps[i]):
            idx[i] -= 1

    # Assign values to matched_data based on found indices
    matched_data = data[idx]

    return matched_data

def _data_transform(data):
    """
    If the data was not provided in the dictionary structure we preprocess it here

    :param data: Dataset to transform.
    """
    data_transformed=[]
    if isinstance(data[0], np.ndarray):
        for e in range(len(data)):
            n,d=data[e].shape
            data_transformed.append({})
            data_transformed[e].update({ind: np.reshape(data[e][:,ind],(n,1)) for ind in range(d)})

    return data_transformed

def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (defult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items