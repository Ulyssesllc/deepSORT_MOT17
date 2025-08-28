# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2
import colorsys
from .image_viewer import ImageViewer


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1.0 - (int(tag * hue_step) % 4) / 5.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


class NoVisualization(object):
    """
    A headless visualization that supports drawing on frames and writing video
    without creating GUI windows. Used in environments without a display.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        h, w = seq_info["image_size"]
        self._window_shape = (w, h)  # (width, height)
        self._update_ms = int(seq_info.get("update_ms", 40) or 40)
        self._video_writer = None

        # Drawing state compatible with ImageViewer
        self.image = np.zeros((h, w, 3), dtype=np.uint8)
        self._color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.thickness = 1

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("color must be tuple of 3")
        self._color = tuple(int(c) for c in value)

    # Methods used by Visualization
    def set_image(self, image):
        # Expect BGR image
        if image is None:
            return
        self.image = image

    def rectangle(self, x, y, w, h, label=None):
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        if label is not None:
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness
            )
            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
            cv2.rectangle(self.image, pt1, pt2, self._color, -1)
            cv2.putText(
                self.image,
                label,
                center,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255),
                self.thickness,
            )

    def draw_groundtruth(self, track_ids, boxes):
        # Visualization class handles ID coloring; here we only support drawing if called directly
        for box in boxes:
            self.rectangle(*box.astype(np.int64))

    def draw_detections(self, detections):
        for det in detections:
            self.rectangle(*det.tlwh)

    def draw_trackers(self, trackers):
        for track in trackers:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.rectangle(*track.to_tlwh().astype(np.int64))

    def enable_videowriter(self, output_filename, fourcc_string="MJPG", fps=None):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
        if fps is None:
            fps = int(1000.0 / max(1, self._update_ms))
        self._video_writer = cv2.VideoWriter(
            output_filename, fourcc, fps, self._window_shape
        )

    def disable_videowriter(self):
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
        self._video_writer = None

    def run(self, update_fun=None):
        """Headless run loop compatible with ImageViewer.run.

        Parameters
        ----------
        update_fun : Optional[Callable[] -> bool]
            A callable invoked at each iteration. Should return True to
            continue, False to terminate. It is expected to update self.image
            via Visualization._update_fun.
        """
        if update_fun is None:
            return
        # Run until update_fun signals termination
        terminate = False
        while not terminate:
            # update_fun will manage frame index and termination
            terminate = not update_fun()
            if self._video_writer is not None and self.image is not None:
                frame = cv2.resize(self.image, self._window_shape)
                self._video_writer.write(frame)


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms, headless=False):
        self.headless = headless
        if self.headless:
            self.viewer = NoVisualization(seq_info)
        else:
            image_shape = seq_info["image_size"][::-1]
            aspect_ratio = float(image_shape[1]) / image_shape[0]
            image_shape = 1024, int(aspect_ratio * 1024)
            self.viewer = ImageViewer(
                update_ms, image_shape, "Figure %s" % seq_info["sequence_name"]
            )
            self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int64), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int64), label=str(track.track_id)
            )
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)


#
