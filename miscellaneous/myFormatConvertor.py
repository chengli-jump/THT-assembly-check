from PyQt5.QtGui import QImage, QPixmap


class FormatConvertor:

    @staticmethod
    def convert_to_QPixmap(capture_frame_raw):
        """convert image to QPixmap.

        Args:
            capture_frame_raw (numpy.ndarray): raw image

        Returns:
            [QPixmap]: QPixmap format.
        """
        tmpHeight, tmpWidth, tmpChannel = capture_frame_raw.shape
        bytesPerLine = 3 * tmpWidth
        tmp_QImage = QImage(capture_frame_raw, tmpWidth, tmpHeight, bytesPerLine, QImage.Format_BGR888)
        tmp_QPixmap = QPixmap.fromImage(tmp_QImage)
        return tmp_QPixmap
