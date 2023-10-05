from datetime import datetime
from uuid import uuid4

from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve
from qtpy.QtGui import QPainter, QColor, QFont, QPen, QPolygon, QBrush
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSlider, QDockWidget
from qtpy.QtCore import Qt, Signal, QLine, QPoint, QRectF, QSize, QRect, QPropertyAnimation, QEasingCurve
from qtpy.QtGui import QColor, QFont, QPalette, QPainter, QPen, QPolygon, QBrush, QPainterPath, QAction, QCursor
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget, QSlider, QDockWidget, QMenu


class KeyFrame:
    def __init__(self, uid, valueType, position, value, color=Qt.darkYellow):
        self.uid = uid
        self.valueType = valueType
        self.position = position
        self.value = value
        self.color = color


__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(0, 0, 0)
__font__ = QFont('Decorative', 10)



class OurTimeline(QWidget):
    keyFramesUpdated = Signal()
    selectionChanged = Signal(object)

    def __init__(self, duration, length):
        super().__init__()
        self.duration = duration
        self.length = length

        # Set variables
        self.backgroundColor = __backgroudColor__
        self.textColor = __textColor__
        self.font = __font__
        self.pos = None
        self.oldPos = None
        self.pointerPos = None
        self.pointerValue = None
        self.pointerTimePos = 0
        self.selectedSample = None
        self.clicking = False  # Check if mouse left button is being pressed
        self.is_in = False  # check if user is in the widget
        self.videoSamples = []  # List of videos samples
        self.middleHover = False
        self.setMouseTracking(True)  # Mouse events
        self.setAutoFillBackground(True)  # background
        self.edgeGrab = False
        self.scale = None
        self.middleHoverActive = False
        self.selectedValueType = "strength"
        self.keyHover = False
        self.hoverKey = None
        self.selectedKey = None
        self.moveSelectedKey = False
        self.posy = 50
        self.yMiddlePoint = 200
        self.verticalScale = 10
        self.keyFrameList = []
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, self.length, 200)
        self.setWindowTitle("TESTE")

        # Set Background
        pal = QPalette()
        pal.setColor(QPalette.Base, self.backgroundColor)
        self.setPalette(pal)

    def mixed_order(self, a):
        return (a.valueType, a.position)

    def paintEvent(self, event):
        self.keyFrameList.sort(key=self.mixed_order)
        self.yMiddlePoint = self.height() / 2

        qp = QPainter()
        # qp.device()
        qp.begin(self)
        qp.setPen(self.textColor)
        qp.setFont(self.font)
        qp.setRenderHint(QPainter.Antialiasing)
        w = 0
        # Draw time
        scale = self.getScale()
        while w <= self.width():
            qp.drawText(w - 50, 0, 100, 100, Qt.AlignHCenter, self.get_time_string(w * scale))
            w += 100
        # Draw down line
        qp.setPen(QPen(Qt.darkCyan, 5, Qt.SolidLine))
        qp.drawLine(0, 40, self.width(), 40)

        # Draw Middle Line for 0 Value of Keyframes
        qp.setPen(QPen(Qt.darkGreen, 2, Qt.SolidLine))
        qp.drawLine(0, int(self.yMiddlePoint), int(self.width()), int(self.yMiddlePoint))
        # Draw dash lines
        point = 0
        qp.setPen(QPen(self.textColor))
        qp.drawLine(0, 40, self.width(), 40)
        while point <= self.width():
            if point % 30 != 0:
                qp.drawLine(3 * point, 40, 3 * point, 30)
            else:
                qp.drawLine(3 * point, 40, 3 * point, 20)
            point += 10

        if self.pos is not None and self.is_in:
            qp.drawLine(int(self.pos), 0, int(self.pos), 40)

        if self.pointerPos is not None:
            self.pointerTimePos = int(self.pointerTimePos)
            line = QLine(QPoint(int(self.pointerTimePos / self.getScale()), 40),
                         QPoint(int(self.pointerTimePos / self.getScale()), self.height()))
            poly = QPolygon([QPoint(int(self.pointerTimePos / self.getScale() - 10), 20),
                             QPoint(int(self.pointerTimePos / self.getScale() + 10), 20),
                             QPoint(int(self.pointerTimePos / self.getScale()), 40)])
        else:
            line = QLine(QPoint(0, 0), QPoint(0, self.height()))
            poly = QPolygon([QPoint(-10, 20), QPoint(10, 20), QPoint(0, 40)])
        self.oldY = None
        self.oldX = None
        if self.selectedValueType is not None:
            for i in self.keyFrameList:
                if i is not None:

                    if i.valueType == self.selectedValueType:
                        kfStartPoint = int(int(i.position) / self.getScale())
                        kfYPos = int(self.yMiddlePoint - i.value * self.verticalScale)
                        if self.oldY is not None:
                            qp.setPen(QPen(Qt.darkMagenta, 2, Qt.SolidLine))
                            # line = QLine(self.oldX, self.oldY, kfStartPoint, kfYPos)
                            ##print(self.oldX, self.oldY, kfStartPoint, kfYPos)
                            qp.drawLine(self.oldX, self.oldY, kfStartPoint, kfYPos)
                        kfbrush = QBrush(Qt.darkRed)

                        ##print(kfYPos)
                        scaleMod = 5
                        kfPoly = QPolygon(
                            [QPoint(int(kfStartPoint - scaleMod), kfYPos), QPoint(kfStartPoint, kfYPos - scaleMod),
                             QPoint(kfStartPoint + scaleMod, kfYPos), QPoint(kfStartPoint, kfYPos + scaleMod)])
                        qp.setPen(Qt.darkRed)
                        qp.setBrush(kfbrush)
                        qp.drawPolygon(kfPoly)

                        self.oldY = kfYPos
                        self.oldX = kfStartPoint

        # Draw samples
        t = 0
        for sample in self.videoSamples:
            # Clear clip path
            path = QPainterPath()

            path.addRoundedRect(QRectF((t + sample.startPos) / scale, 50, sample.duration / scale, 200), 10, 10)

            qp.setClipPath(path)

            # Draw sample
            path = QPainterPath()
            qp.setPen(sample.color)
            qp.setBrush(sample.color)

            # path.addRoundedRect(QRectF(((t + sample.startPos)/scale), 50, (sample.duration / scale), 50), 10, 10)
            path.addRect((t + sample.startPos) / scale, 50, (sample.duration / scale), 50)
            # sample.startPos = (t + sample.startPos)*scale
            sample.endPos = (t + sample.startPos) / scale + sample.duration / scale
            qp.fillPath(path, sample.color)
            qp.drawPath(path)

            # Draw preview pictures
            if sample.picture is not None:
                if sample.picture.size().width() < sample.duration / scale:
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(t / scale, 52.5, sample.picture.size().width(), 45), 10, 10)
                    qp.setClipPath(path)
                    qp.drawPixmap(QRect(int(t / scale), 52.5, sample.picture.size().width(), 45), sample.picture)
                else:
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(t / scale, 52.5, sample.duration / scale, 45), 10, 10)
                    qp.setClipPath(path)
                    pic = sample.picture.copy(0, 0, sample.duration / scale, 45)
                    qp.drawPixmap(QRect(int(t / scale), 52.5, sample.duration / scale, 45), pic)
            t += sample.duration

        # Clear clip path
        path = QPainterPath()
        path.addRect(self.rect().x(), self.rect().y(), self.rect().width(), self.rect().height())
        qp.setClipPath(path)

        # Draw pointer
        qp.setPen(Qt.darkCyan)
        qp.setBrush(QBrush(Qt.darkCyan))

        qp.drawPolygon(poly)
        qp.drawLine(line)
        qp.end()

    # Mouse movement
    def mouseMoveEvent(self, e):

        self.pos = e.position().x()
        self.posy = e.position().y()
        self.pointerValue = self.posy  # if mouse is being pressed, update pointer

        self.checkKeyframeHover(self.pos)

        if self.clicking:

            self.oldPos = self.pointerPos
            self.oldValue = self.pointerValue
            x = self.pos
            y = self.posy
            self.pointerPos = x

            self.pointerTimePos = self.pointerPos * self.getScale()

            if self.keyHover == True:
                for item in self.keyFrameList:
                    if self.selectedKey is item.uid:
                        item.position = int(self.pointerPos * self.scale)
                        if item.position <= 0:
                            item.position = 0
                        value = (self.pointerValue - self.yMiddlePoint) / self.verticalScale
                        item.value = -value
                        self.keyFramesUpdated.emit()
                        # print(item.value)
                        # print(self.posy)
                        # print(self.yMiddlePoint)
            if self.edgeGrabActive == True:
                for sample in self.videoSamples:
                    sample.duration = sample.duration + ((self.pointerPos - self.oldPos) * self.scale)
            elif self.middleHoverActive == True:
                self.scale = self.getScale()
                for sample in self.videoSamples:
                    change = (x - self.oldPos)
                    change = (change * self.scale)
                    ##print(change)
                    sample.startPos = sample.startPos + change
                    sample.endPos = sample.endPos + change
        self.update()

    # Mouse pressed
    def checkKeyframeHover(self, x):
        for item in self.keyFrameList:
            kfStartPoint = int(int(item.position) / self.getScale())
            kfYPos = int(self.yMiddlePoint - item.value * self.verticalScale)

            if kfStartPoint - 5 < x < kfStartPoint + 5 and kfYPos + 5 > self.posy > kfYPos - 5:
                self.keyHover = True
                ##print(item.uid)
                self.hoverKey = item.uid
        self.update()

    def checkKeyClicked(self):
        for item in self.keyFrameList:
            if self.hoverKey is item.uid:
                self.selectedKey = self.hoverKey
                self.keyHover = True
        self.update()

    def mousePressEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()
            self.checkKeyClicked()

            ##print(self.keyClicked)
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos * self.getScale()

            self.clicking = True  # Set clicking check to true
            if self.edgeGrab == True:
                self.edgeGrabActive = True
            else:
                self.edgeGrabActive = False
            if self.middleHover == True:
                self.middleHoverActive = True
            else:
                self.middleHoverActive = False
        elif e.button() == Qt.RightButton:
            self.popMenu = QMenu()
            menuPosition = QCursor.pos()
            x = self.pos
            self.checkKeyframeHover(x)
            self.checkKeyClicked()
            ##print(self.hoverKey)
            ##print(self.keyHover)
            ##print(self.selectedKey)
            self.popMenu.clear()
            # populate
            self.populateBtnContext()

            if self.selectedKey is None:
                self.popMenu.delete_action.setEnabled(False)

            # show
            self.popMenu.move(menuPosition)
            self.popMenu.show()
            self.pointerPos = e.pos().x()
            self.popMenu.delete_action.triggered.connect(self.delete_action)
            self.popMenu.add_action.triggered.connect(self.add_action)
        self.update()

    def populateBtnContext(self):

        # Do some if here :
        self.popMenu.add_action = QAction('add keyframe', self)
        self.popMenu.delete_action = QAction('delete keyframe', self)
        self.popMenu.addAction(self.popMenu.delete_action)
        self.popMenu.addAction(self.popMenu.add_action)

    # Mouse release
    def add_action(self):
        ##print(self.keyClicked)
        # self.pointerPos
        self.pointerTimePos = self.pointerPos * self.getScale()

        matchFound = False
        value = (self.pointerValue - self.yMiddlePoint) / self.verticalScale
        value = -value
        valueType = self.selectedValueType
        position = int(self.pointerTimePos)
        keyframe = {}
        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        keyframe[position] = KeyFrame(uid, valueType, position, value)
        for items in self.keyFrameList:
            if items.valueType == valueType:
                if items.position == position:
                    items.value = value
                    matchFound = True
        if matchFound == False:
            self.keyFrameList.append(keyframe[position])
        self.update()
        # print(self.keyFrameList)
        # self.updateAnimKeys()

    def delete_action(self):
        for idx, item in enumerate(self.keyFrameList):
            # print(idx)
            # print(item)
            if self.hoverKey is item.uid:
                self.keyFrameList.pop(idx)
        self.update()
        # item.remove()
        # return

    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            self.clicking = False  # Set clicking check to false
            self.selectedKey = None
            self.keyHover = False
            self.hoverKey = None

        self.update()

    # Enter
    def enterEvent(self, e):
        self.is_in = True
        self.update()

    # Leave
    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    # check selection
    def checkSelection(self, x):
        # Check if user clicked in video sample
        for sample in self.videoSamples:
            if sample.startPos + 25 < x < sample.endPos - 25:
                sample.color = Qt.darkCyan
                self.middleHover = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    # self.selectionChanged.emit(sample)
            else:
                sample.color = sample.defColor
                self.middleHover = False
        self.update()

    def checkEdges(self, x, y=50):

        for sample in self.videoSamples:
            if sample.startPos < x < sample.startPos + 24:
                sample.color = Qt.darkMagenta
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    # self.selectionChanged.emit(sample)
            elif sample.endPos - 24 < x < sample.endPos:
                sample.color = Qt.darkGreen
                self.edgeGrab = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    # self.selectionChanged.emit(sample)
            else:
                sample.color = sample.defColor
                self.edgeGrab = False
        self.update()

    # Get time string from seconds
    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        # return "%02d:%02d:%02d" % (h, m, s)
        return "%05d" % (seconds)

    # Get scale from length
    def getScale(self):
        return float(self.duration) / float(self.width())

    # Get duration
    def getDuration(self):
        return self.duration

    # Get selected sample
    def getSelectedSample(self):
        return self.selectedSample

    # Set background color
    def setBackgroundColor(self, color):
        self.backgroundColor = color

    # Set text color
    def setTextColor(self, color):
        self.textColor = color

    # Set Font
    def setTextFont(self, font):
        self.font = font



class Timeline(QDockWidget):
    def __init__(self, parent=None):
        super(Timeline, self).__init__(parent)

        self.timeline = OurTimeline(1000, 1000)
        self.zoomSlider = QSlider(Qt.Horizontal)

        self.zoomSlider.valueChanged.connect(self.onZoomChanged)

        layout = QVBoxLayout()
        layout.addWidget(self.timeline)
        layout.addWidget(self.zoomSlider)

        container = QWidget()
        container.setLayout(layout)
        self.setWidget(container)

        #self.initAnimations()

    def onZoomChanged(self, value):
        # Handle zoom logic
        pass

    def initAnimations(self):
        self.hideAnimation = QPropertyAnimation(self, b"maximumHeight")
        self.hideAnimation.setDuration(500)
        self.hideAnimation.setStartValue(self.height())
        self.hideAnimation.setEndValue(0)
        self.hideAnimation.setEasingCurve(QEasingCurve.Linear)

        self.showAnimation = QPropertyAnimation(self, b"maximumHeight")
        self.showAnimation.setDuration(500)
        self.showAnimation.setStartValue(0)
        self.showAnimation.setEndValue(self.parent().height())
        self.showAnimation.setEasingCurve(QEasingCurve.Linear)

    def showWithAnimation(self):
        self.showAnimation.start()

    def hideWithAnimation(self):
        self.hideAnimation.start()