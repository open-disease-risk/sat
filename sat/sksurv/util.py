# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This program includes a portion of scikit-survival, authored by
# Sebastian Pölsterl and modified on October 19, 2023
import numpy as np
from sklearn.utils import check_array

__all__ = ["check_y_survival"]


def check_y_survival(y_or_event, *args, allow_all_censored=False):
    """Check that array correctly represents an outcome for survival analysis.

    Parameters
    ----------
    y_or_event : structured array with two fields, or boolean array
        If a structured array, it must contain the binary event indicator
        as first field, and time of event or time of censoring as
        second field. Otherwise, it is assumed that a boolean array
        representing the event indicator is passed.

    *args : list of array-likes
        Any number of array-like objects representing time information.
        Elements that are `None` are passed along in the return value.

    allow_all_censored : bool, optional, default: False
        Whether to allow all events to be censored.

    Returns
    -------
    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    if len(args) == 0:
        y = y_or_event

        if (
            not isinstance(y, np.ndarray)
            or y.dtype.fields is None
            or len(y.dtype.fields) != 2
        ):
            raise ValueError(
                "y must be a structured array with the first field"
                " being a binary class event indicator and the second field"
                " the time of the event/censoring"
            )

        event_field, time_field = y.dtype.names
        y_event = y[event_field]
        time_args = (y[time_field],)
    else:
        y_event = np.asanyarray(y_or_event)
        time_args = args

    event = check_array(y_event, ensure_2d=False)
    if not np.issubdtype(event.dtype, np.bool_):
        raise ValueError(
            "elements of event indicator must be boolean, but found {0}".format(
                event.dtype
            )
        )

    if not (allow_all_censored or np.any(event)):
        raise ValueError("all samples are censored")

    return_val = [event]
    for i, yt in enumerate(time_args):
        if yt is None:
            return_val.append(yt)
            continue

        yt = check_array(yt, ensure_2d=False)
        if not np.issubdtype(yt.dtype, np.number):
            raise ValueError(
                "time must be numeric, but found {} for argument {}".format(
                    yt.dtype, i + 2
                )
            )

        return_val.append(yt)

    return tuple(return_val)
