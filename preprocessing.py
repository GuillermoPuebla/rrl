import itertools
from scipy.spatial import distance as dist
import numpy as np
from collections import OrderedDict, namedtuple
import cv2
import matplotlib.pyplot as plt


# Fact class. The last two arguments are optional
Fact = namedtuple('Fact', ['type', 'value', 'name', 'obj1', 'obj2', 'obj3'], defaults=[None, None])

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DemonAttackPreprocessor:
    """
    Class to preprocess Demon Attack screens and get features to form a relational state.
    Object detection is done by color for the player, player missile and enemy missile.
    The enemies are located by the number of points of their contours.
    """
    def __init__(self):
        # Initialize img and lab
        self.img = None
        self.img_lab = None

        # Initialize the colors dictionary, containing the color name as the key and the RGB tuple as the value
        self.colors = OrderedDict()
        self.colors['player'] = (184, 70, 162)
        self.colors['player_missile'] = (212, 140, 252)
        self.colors['enemy_missile'] = (252, 144, 144)

        # Allocate memory for the L*a*b* image, then initialize the color names list
        self.lab = np.zeros((len(self.colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # Loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(self.colors.items()):
            # Update the L*a*b* array and the color names list.
            self.lab[i] = rgb
            self.colorNames.append(name)

        # Convert the L*a*b* array from the RGB color space to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

        # Initialize boundaries in BGR. Here colors are exact but kept for easy extention if needed
        self.boundaries = OrderedDict()
        for (name, rgb) in self.colors.items():
            to_bgr = list(rgb)
            to_bgr[0], to_bgr[2] = to_bgr[2], to_bgr[0]
            self.boundaries[name] = (np.array(to_bgr), np.array(to_bgr))

    def read_from_file(self, file_name):
        """Read from file"""
        self.img = cv2.imread(file_name)
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)

    def read_from_array(self, array):
        """Read from rgb array, write bgr img."""
        self.img = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)

    def get_contours(self):
        """Return a list of contours."""
        # If the screen is not black return empty list inmediatly
        mask = np.zeros(self.img_lab.shape[:2], dtype="uint8")
        mask[5:10, 0:5] = 1
        mean = np.mean(cv2.mean(self.img, mask=mask)[:3])
        if mean > 0.0:
            return []
        
        # Draw filled black rectangles at the top and bottom of the image
        cv2.rectangle(self.img, (0, 0), (160, 15), (0, 0, 0), -1)
        cv2.rectangle(self.img, (0, 188), (160, 210), (0, 0, 0), -1)

        # Generate masks based on color
        labels_and_masked_imgs = []
        # Loop over the boundaries
        for (name, boundaries) in self.boundaries.items():
            # Find the colors within the specified boundaries and apply the mask.
            mask = cv2.inRange(self.img, boundaries[0], boundaries[1])
            masked = cv2.bitwise_and(self.img, self.img, mask=mask)
            labels_and_masked_imgs.append((name, masked))
        labels_and_contours = []
        # Convert masked images to grayscale
        for elm in labels_and_masked_imgs:
            label, masked = elm
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

            # find contour in the thresholded image
            contour = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contour = contour[0] if len(contour) == 2 else contour[1]
            for cnt in contour:
                labels_and_contours.append((label, cnt))

        # Black out the labeled objects
        image = self.img.copy()
        for elm in labels_and_contours:
            label, c = elm
            cv2.drawContours(image, c, -1, (0, 0, 0), -1)
        # Find contours.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        contour = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #RETR_LIST — Python: cv.RETR_LIST
        contour = contour[0] if len(contour) == 2 else contour[1]
        # Save all contours with 16+ vertices
        for cnt in contour:
            if len(cnt) < 16:
                pass
            elif 16 <= len(cnt) <= 22:
                label = 'enemy_small'
                labels_and_contours.append((label, cnt))
            else:
                label = 'enemy_big'
                labels_and_contours.append((label, cnt))
        
        return labels_and_contours

    def get_info(self, array):
        """Returns a dictionary of objects and (x, y) positions."""
        # Initialize object dictionary
        info = {}
        # Read
        self.read_from_array(array)
        # Get label-contour pais
        labels_and_contours = self.get_contours()
        # If there are contours populate dictionary
        player_data = []
        enemy_missile_data = []
        enemy_big_data = []
        enemy_small_data = []
        
        for label, contour in labels_and_contours:
            if label == 'player':
                x, y, w, h = cv2.boundingRect(contour)
                player_data.append([x, y])
                player_data.append([x+w, y+h])
            elif label == 'player_missile':
                # There is only one player missile in each screen
                x, y, w, h = cv2.boundingRect(contour)
                info['player_missile'] = (Point(x, y), Point(x+w, y+h))
            elif label == 'enemy_missile':
                x, y, w, h = cv2.boundingRect(contour)
                # Append save all points of bounding rectangle
                enemy_missile_data.append([x, y])
                enemy_missile_data.append([x+w, y+h])
            elif label == 'enemy_big':
                x, y, w, h = cv2.boundingRect(contour)
                # Append save all points of bounding rectangle
                enemy_big_data.append([x, y, x+w, y+h])
            elif label == 'enemy_small':
                x, y, w, h = cv2.boundingRect(contour)
                # Append save all points of bounding rectangle
                enemy_small_data.append([x, y, x+w, y+h])
            else:
                raise ValueError('Unrecogsized label!')
        
        # Get player data.
        if player_data:
            min_x = min([x[0] for x in player_data])
            max_x = max([x[0] for x in player_data])
            min_y = min([x[1] for x in player_data])
            max_y = max([x[1] for x in player_data])
            info['player'] = (Point(min_x, min_y), Point(max_x, max_y))

        if enemy_missile_data:
            min_x = min([x[0] for x in enemy_missile_data])
            max_x = max([x[0] for x in enemy_missile_data])
            min_y = min([x[1] for x in enemy_missile_data])
            max_y = max([x[1] for x in enemy_missile_data])
            info['enemy_missile'] = (Point(min_x, min_y), Point(max_x, max_y))

        # Big enemies are sorted by y-coordinate.
        if enemy_big_data:
            sorted_enemy_big_data = sorted(enemy_big_data, key = lambda x: 210-x[1])
            for i, elm in enumerate(sorted_enemy_big_data):
                info[f'enemy_big_{i}'] = (Point(elm[0], elm[1]), Point(elm[2], elm[3]))

        # Small enemies are sorted by y and x position.
        if enemy_small_data:
            sorted_enemy_small_data = sorted(enemy_small_data, key = lambda x: (210-x[1], x[0]))
            for i, elm in enumerate(sorted_enemy_small_data):
                info[f'enemy_small_{i}'] = (Point(elm[0], elm[1]), Point(elm[2], elm[3]))
        
        return info
    
    def draw_contours(self):
        """
        Draw contours of objects detected in image. 
        This method is useful for debuggin.
        """
        labels_and_contours = self.get_contours()
        image = self.img.copy()
        for l, c in labels_and_contours:
            cv2.drawContours(image, c, -1, (0, 255, 0), 1)
        b,g,r=cv2.split(image)
        imge_matplotlib=cv2.merge([r,g,b])
        return imge_matplotlib
    
    def show_img(self):
        """
        Shows image. 
        This method is useful for debuggin.
        """
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.show()

class BreakoutPreprocessor:
    """
    On Breakout the paddle and the ball are red, so object identification 
    is based on position and shape instead.
    """
    def __init__(self):
        # Initialize img and lab
        self.img = None
        self.img_lab = None

        # Initialize the colors dictionary, containing the color
        # name as the key and the RGB tuple as the value
        self.colors = OrderedDict()
        self.colors['red'] = (200, 72, 72)

        # Allocate memory for the L*a*b* image, then initialize
        # the color names list
        self.lab = np.zeros((len(self.colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # Loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(self.colors.items()):
            # update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)

        # Convert the L*a*b* array from the RGB color space to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

        # Initialize boundaries in BGR
        self.boundaries = OrderedDict()
        for (name, rgb) in self.colors.items():
            to_bgr = list(rgb)
            to_bgr[0], to_bgr[2] = to_bgr[2], to_bgr[0]
            self.boundaries[name] = (np.array(to_bgr)-20, np.array(to_bgr)+20)

    def read_from_file(self, file_name):
        """Read from file."""
        self.img = cv2.imread(file_name)
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)

    def read_from_array(self, array):
        """Read from rgb array, write bgr img."""
        self.img = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)

    def get_contours(self):
        """Return a list of contours."""
        # Draw filled black rectangles at the top, left and right of the img
        cv2.rectangle(self.img, (0, 0), (160, 32), (0, 0, 0), -1)
        cv2.rectangle(self.img, (0, 0), (7, 210), (0, 0, 0), -1)
        cv2.rectangle(self.img, (152, 0), (160, 210), (0, 0, 0), -1)
        # Also cover the red bricks
        cv2.rectangle(self.img, (8, 56), (152, 62), (0, 0, 0), -1)
        # Generate masks based on color
        masked_imgs = []
        # Loop over the boundaries
        for (name, boundaries) in self.boundaries.items():
            # find the colors within the specified boundaries and apply the mask
            mask = cv2.inRange(self.img, boundaries[0], boundaries[1])
            masked = cv2.bitwise_and(self.img, self.img, mask=mask)
            masked_imgs.append(masked)

        # Convert masked images to grayscale
        for masked in masked_imgs:
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

            # Find contours in the thresholded image
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = contours[0] if len(contours) == 2 else contours[1]

        return contours

    def show_img(self):
        """
        Shows image. 
        This method is useful for debuggin.
        """
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.show()

    def get_area(self, bounding_rect):
        """Get the area of a single opencv bounding rectangle"""
        # Every bounding box is composed of: x,y,w,h
        return bounding_rect[2] * bounding_rect[3]

    def separate_ball_paddle(self, bounding_rect):
        """Get coordinates of ball and paddle in case they are touching."""
        # Bounding rectangle coordiantes
        bounding_x = bounding_rect[0]
        bounding_y = bounding_rect[1]
        bounding_w = bounding_rect[2]
        bounding_h = bounding_rect[3]

        # Grayscale and thresh
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

        # Find contours in the thresholded image
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # Contours is a list of numpy arrays (one is this case)
        # The array contains x,y indexes of the vertices
        vertices = contours[0]
        # Remove unnecessary dimension
        vertices = np.squeeze(vertices, axis=1)

        # If the bounding rectangle is above the fixed y-coordinate of the paddle (189), the ball is above the paddle
        if bounding_y < 189:
            # The highest, leftmost vertex is the ball (x, y) coordinate
            ball_x = vertices[0][0]
            ball_y = vertices[0][1]
            # Difference between the y-coordinates of the first and last vertices corresponds to the  width of the ball
            # Min width value is 1
            ball_w = max(abs(ball_x - vertices[-1][0]) + 1, 1)
            # Sssume ball height is always 4
            ball_h = 4
            # Paddle y-coordinate is always 189, paddle height is always 4
            paddle_y = 189
            paddle_h = 4
            # If the bounding rectangle has the same width as the paddle the x-coordinate is the same
            if bounding_w <= 16:
                paddle_x = bounding_x
                paddle_w = bounding_w
            # If the width of the bounding rectangle is higher than the standard width of the paddle (16)...
            if bounding_w > 16:
                if ball_x == bounding_x:
                    paddle_x = ball_x + ball_w  # is always at the next pixel
                else:
                    paddle_x = bounding_x
                paddle_w = 16

        # If the bounding rectangle is at the level of the paddle, the ball is bellow the paddle
        else:
            # The highest, leftmost vertex is the ball (x, y) coordinate
            paddle_x = vertices[0][0]
            paddle_y = vertices[0][1]
            # The width of the paddle is the standard (16), unless the width of the bounding box is less than that
            if bounding_w >= 16:
                paddle_w = 16
            else:
                paddle_w = bounding_w
            # Assume the height of the paddle to be always 4
            paddle_h = 4
            # Assume ball width and height
            ball_h = 4
            ball_w = 2

            # Infer ball y-coordinate based on vertex most at the bottom
            y_values = []
            for vector in vertices:
                y_values.append(vector[1])
            ball_y = max(y_values) - (ball_h - 1)

            # Infer ball x-coordinate from the leftmost vertex under the paddle...
            x_values = []
            for vector in vertices:
                if vector[1] >= paddle_y + paddle_h-1:
                    x_values.append(vector[0])
            ball_x = min(x_values)

        # Check the vertices
        counter = 1
        for cnt in contours[0]:
            indexes = cnt[0, :]
            if counter % 2 == 0:
                # Green
                self.img[indexes[1], indexes[0], :] = [0, 255, 0]
            else:
                # Blue
                self.img[indexes[1], indexes[0], :] = [0, 0, 255]
            counter += 1

        return [ball_x, ball_y, ball_w, ball_h], [paddle_x, paddle_y, paddle_w, paddle_h]

    def get_boxes(self):
        """
        Returns array of boxes in order: orange paddle, green paddle, ball
        """
        # Get contours
        contours = self.get_contours()

        # Get boxes
        boxes = [list(cv2.boundingRect(c)) for c in contours]

        # Check whether the ball is touching the paddle,
        # in which case there will be just one box with area bigger than 16*4
        if len(boxes) == 1 and self.get_area(boxes[0]) > 64:
            # Get rectangle
            rect = boxes[0]
            # Get x,y,w,h for ball and paddle
            ball, paddle = self.separate_ball_paddle(rect)
        # If there is more than one box get ball and paddle individually
        else:
            # Ball has shape (w, h) (2, 4) and paddle (16, 4)
            # The ball has always a width of max 2 and a max height of 4
            ball = [x for x in boxes if x[2] <= 2 and x[3] <= 4]
            if not ball:
                ball = [0, 0, 0, 0]
            else:
                ball = ball[0]

            # For the paddle include the y coordinate as it never changes
            paddle = [x for x in boxes if x[1] == 189 and 8 <= x[2] <= 16 and x[3] == 4]
            if not paddle:
                paddle = [0, 0, 0, 0]
            else:
                paddle = paddle[0]

        ordered_boxes = ball + paddle

        return np.array(ordered_boxes)

    def get_info(self, array):
        """Returns a dictionary of objects and (x, y) positions."""
        # Initialize object dictionary
        info = {}
        info['labels'] = {}
        # Read
        self.read_from_array(array)
        # Get features
        feature_array = self.get_boxes()
        # get_screen_features returns (x, y, w, h) ball and paddle
        x_ball = feature_array[0] + feature_array[2] / 2
        y_ball = feature_array[1] + feature_array[3] / 2
        x_paddle = feature_array[4] + feature_array[6] / 2
        y_paddle = feature_array[5] + feature_array[7] / 2
        # Save info to dictionary
        info['labels']['player_x'] = x_paddle
        info['labels']['player_y'] = y_paddle
        info['labels']['ball_x'] = x_ball
        info['labels']['ball_y'] = y_ball
        return info

class PongPreprocessor:
    """
    On Pong the left paddle is red, the ball is white and the right paddle is green, 
    so object identification is based on color
    """
    def __init__(self):
        # Initialize img and lab
        self.img = None
        self.img_lab = None

        # Initialize the colors dictionary, containing the color name as the key and the RGB tuple as the value
        self.colors = OrderedDict()
        self.colors['orange'] = (213, 130, 74)
        self.colors['green'] = (92, 186, 92)
        self.colors['white'] = (236, 236, 236)

        # Allocate memory for the L*a*b* image, then initialize the color names list
        self.lab = np.zeros((len(self.colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # Loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(self.colors.items()):
            # Update the L*a*b* array and the color names list
            self.lab[i] = rgb
            self.colorNames.append(name)

        # Convert the L*a*b* array from the RGB color space to L*a*b*
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

        # Initialize boundaries in BGR. No used here but a good idea in general.
        self.boundaries = OrderedDict()
        for (name, rgb) in self.colors.items():
            to_bgr = list(rgb)
            to_bgr[0], to_bgr[2] = to_bgr[2], to_bgr[0]
            self.boundaries[name] = (np.array(to_bgr), np.array(to_bgr))

    def read_from_file(self, file_name):
        """Read from file"""
        self.img = cv2.imread(file_name)
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)

    def read_from_array(self, array):
        """Read from rgb array, write bgr img."""
        self.img = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)

    def label(self, c):
        """Return a color label for an individual contour c."""
        # Construct a mask for the contour, then compute the
        # average L*a*b* value for the masked region
        mask = np.zeros(self.img_lab.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mean = cv2.mean(self.img_lab, mask=mask)[:3]

        # Initialize the minimum distance found thus far
        min_dist = (np.inf, None)

        # Loop over the known L*a*b* color values
        for (i, row) in enumerate(self.lab):
            # Compute the distance between the current L*a*b*
            # color value and the mean of the image
            d = dist.euclidean(row[0], mean)

            # If the distance is smaller than the current distance,
            # then update the bookkeeping variable
            if d < min_dist[0]:
                min_dist = (d, i)

        # Return the name of the color with the smallest distance
        return self.colorNames[min_dist[1]]

    def get_contours(self):
        """Return a list of contours"""
        # Draw filled black rectangles at the top and bottom of the img
        cv2.rectangle(self.img, (0, 0), (320, 33), (17, 72, 144), -1)
        cv2.rectangle(self.img, (0, 194), (320, 210), (17, 72, 144), -1)

        # Generate masks based on color
        masked_imgs = []
        # Loop over the boundaries
        for (name, boundaries) in self.boundaries.items():
            # Find the colors within the specified boundaries and apply the mask
            mask = cv2.inRange(self.img, boundaries[0], boundaries[1])
            masked = cv2.bitwise_and(self.img, self.img, mask=mask)
            masked_imgs.append(masked)

        contours = []
        # Convert masked images to grayscale
        for masked in masked_imgs:
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]

            # Find contour in the thresholded image
            contour = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contour = contour[0] if len(contour) == 2 else contour[1]

            if contour:
                contours.append(contour[0])  # contour[0] because is a list

        return contours

    def draw_contours(self):
        """
        Draw contours of objects detected in image. 
        This method is useful for debuggin.
        """
        contours = self.get_contours()
        image = self.img.copy()
        for c, (n, bs) in zip(contours, self.boundaries.items()):
            cv2.drawContours(image, c, -1, (int(bs[0][0]), int(bs[0][1]), int(bs[0][2])), 1)
        b,g,r=cv2.split(image)
        imge_matplotlib=cv2.merge([r,g,b])
        return imge_matplotlib
    
    def show_img(self):
        """
        Shows image. 
        This method is useful for debuggin.
        """
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.show()

    def get_boxes(self):
        """Returns an array of boxes in order: orange paddle, green paddle, ball.
        Each box is x,y,w,h"""
        # Get contours
        contours = self.get_contours()

        # Get labels
        labels = [self.label(c) for c in contours]

        # Get boxes
        boxes = [list(cv2.boundingRect(c)) for c in contours]

        # Order boxes
        ordered_boxes = []
        ordered_colors = []
        for color, rgb in self.colors.items():
            if color in labels:
                ordered_boxes.append(boxes[labels.index(color)])
                ordered_colors.append(color)
            else:
                ordered_boxes.append([0, 0, 0, 0])
                ordered_colors.append(color)

        ordered_boxes = np.array(ordered_boxes)
        return ordered_boxes

    def get_info(self, array):
        """Returns a dictionary of objects and (x, y) positions."""
        # Initialize object dictionary
        info = {}
        info['labels'] = {}
        # Read
        self.read_from_array(array)
        # Get features
        feature_array = self.get_boxes()
        # get_boxes returns (x, y, w, h) orange paddle, green paddle, ball
        x_enemy = feature_array[0][0] + feature_array[0][2] / 2
        y_enemy = feature_array[0][1] + feature_array[0][3] / 2
        x_player = feature_array[1][0] + feature_array[1][2] / 2
        y_player = feature_array[1][1] + feature_array[1][3] / 2
        x_ball = feature_array[2][0] + feature_array[2][2] / 2
        y_ball = feature_array[2][1] + feature_array[2][3] / 2
        # Save info to dictionary
        info['labels']['enemy_x'] = x_enemy
        info['labels']['enemy_y'] = y_enemy
        info['labels']['player_x'] = x_player
        info['labels']['player_y'] = y_player
        info['labels']['ball_x'] = x_ball
        info['labels']['ball_y'] = y_ball

        return info

def get_comparative_values(left1, right1, left2, right2, x_tolerance=3, y_tolerance=3):
    """
    Calculates the comparative values from a set of Point objects.
    The points are represented in OpenCV coordinates.

    Args:
        left1: x- and y-coordinates of the top-left corner of object 1.
        right1: x- and y-coordinates of the bottom-right corner of object 1.
        left2: x- and y-coordinates of the top-left corner of object 2.
        right2: x- and y-coordinates of the bottom-right corner of object 2.
    Returns:
        string: 'more', 'same' or 'less'.
    """
    # Check if either rectangle is actually a line
    if (left1.x == right1.x or left1.y == right1.y or left2.x == right2.x or left2.y == right2.y):
        raise ValueError('one of the rectangles is a line!')
    # X-axis
    # If rectangle 1 is on the right side of rectangle 2
    if (left1.x - right2.x) >= x_tolerance:
        x_value = 'more'
    # If rectangle 2 is on the right side of rectangle 1
    elif (left2.x - right1.x) >= x_tolerance:
        x_value = 'less'
    else:
        x_value = 'same'
    # Y-axis
    if (left1.y - right2.y) >= y_tolerance:
        y_value = 'less'
    elif (left2.y - right1.y) >= y_tolerance:
        y_value = 'more'
    else:
        y_value = 'same'
    return x_value, y_value

def get_state_comparative_demon_attack(current_info, previous_info, include_incomplete_states=False):
    """
    Builds a relational state from current_info and previous_info.
    These are dictionaries from the DemonAttackPreprosessor where the values are the 
    x- and y-coordinates of the top-left corner and the bottom-right corner of the
    bounding box of the corresponding object.

    Args:
        current_info: dictionary of the current time step.
        previous_info: dictionary of the previous time step.
    Returns:
        A set of mostly comparative relations that hold in the state. 
    """
    # Tolerance levels for x and y dimensions.
    x_tolerance = 3
    y_tolerance = 3
    # Initialize state.
    rel_state = set()
    # Get combinations of objects and order according to object_hierarchy
    object_hierarchy = [
        'player',
        'player_missile',
        'enemy_missile',
        'enemy_big_0',
        'enemy_big_1',
        'enemy_big_2',
        'enemy_small_0',
        'enemy_small_1',
        'enemy_small_2',
        'enemy_small_3',
        'enemy_small_4',
        'enemy_small_5'
        ]
    object_combinations = list(itertools.combinations(object_hierarchy, 2))
    # Relations between objects in the current screen
    for obj1, obj2 in object_combinations:
        # Current screen variablesvg
        if obj1 in current_info.keys() and obj2 in current_info.keys():
            obj1_left, obj1_right = current_info[obj1]
            obj2_left, obj2_right = current_info[obj2]
            x_value, y_value = get_comparative_values(obj1_left, obj1_right, obj2_left, obj2_right, x_tolerance=x_tolerance, y_tolerance=y_tolerance)
            # Encode x-relation
            rel_state.add(Fact(name='x', type='comparative', value=x_value, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            # Encode y-relation
            rel_state.add(Fact(name='y', type='comparative', value=y_value, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
    # Relations between sigle objects across time
    if previous_info:
        for obj in object_hierarchy:
            if obj in current_info.keys() and obj in previous_info.keys():
                current_left, current_right = current_info[obj]
                previous_left, previous_right = previous_info[obj]
                # Get absolute comparison on x-axis
                if current_left.x - previous_left.x > 0:
                    x_value = 'more'
                elif current_left.x - previous_left.x < 0:
                    x_value = 'less'
                else:
                    x_value = 'same'
                # Get absolute comparison on y-axis
                if current_left.y - previous_left.y > 0:
                    y_value = 'less'
                elif current_left.y - previous_left.y < 0:
                    y_value = 'more'
                else:
                    y_value = 'same'
                # Encode objects' x-trajectory
                rel_state.add(Fact(name='x', type='comparative', value=x_value, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                # Encode objects' y-trajectory
                rel_state.add(Fact(name='y', type='comparative', value=y_value, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))

    if include_incomplete_states:
        return rel_state
    else:
        objects_to_check = ['enemy_missile_t']
        objects_in_state = [fact.obj2 for fact in rel_state if fact.obj2.endswith('t')]
        rel_state = rel_state if len(set(objects_in_state).intersection(objects_to_check)) > 0 else set()
        return rel_state

def get_state_logical_demon_attack(current_info, previous_info, include_incomplete_states=False):
    """
    Builds a relational state from current_info and previous_info.
    These are dictionaries from the DemonAttackPreprosessor where the values are the 
    x- and y-coordinates of the top-left corner and the bottom-right corner of the
    bounding box of the corresponding object.

    Args:
        current_info: dictionary of the current time step.
        previous_info: dictionary of the previous time step.
    Returns:
        A set of logical relations that hold in the state.  
    """
    # Tolerance levels for x and y dimensions
    x_tol = 3
    y_tol = 3
    # Initialize state
    rel_state = set()
    # Get combinations of objects and order according to object_hierarchy
    object_hierarchy = [
        'player',
        'player_missile',
        'enemy_missile',
        'enemy_big_0',
        'enemy_big_1',
        'enemy_big_2',
        'enemy_small_0',
        'enemy_small_1',
        'enemy_small_2',
        'enemy_small_3',
        'enemy_small_4',
        'enemy_small_5'
        ]
    object_combinations = list(itertools.combinations(object_hierarchy, 2))
    # Relations between objects in the current screen
    for obj1, obj2 in object_combinations:
        # Current screen variables
        if obj1 in current_info.keys() and obj2 in current_info.keys():
            obj1_left, obj1_right = current_info[obj1]
            obj2_left, obj2_right = current_info[obj2]
            x_value, y_value = get_comparative_values(obj1_left, obj1_right, obj2_left, obj2_right, x_tolerance=x_tol, y_tolerance=y_tol)
            # Encode x-relation
            same_x_val = 'True' if x_value == 'same' else 'False'
            more_x_val = 'True' if x_value == 'more' else 'False'
            less_x_val = 'True' if x_value == 'less' else 'False'
            rel_state.add(Fact(name='same-x', type='logical', value=same_x_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='more-x', type='logical', value=more_x_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='less-x', type='logical', value=less_x_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            # Encode y-relation
            same_y_val = 'True' if y_value == 'same' else 'False'
            more_y_val = 'True' if y_value == 'more' else 'False'
            less_y_val = 'True' if y_value == 'less' else 'False'
            rel_state.add(Fact(name='same-y', type='logical', value=same_y_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='more-y', type='logical', value=more_y_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='less-y', type='logical', value=less_y_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
    # Relations between sigle objects across time
    if previous_info:
        for obj in object_hierarchy:
            if obj in current_info.keys() and obj in previous_info.keys():
                current_left, current_right = current_info[obj]
                previous_left, previous_right = previous_info[obj]
                # Get absolute comparison on x-axis
                if current_left.x - previous_left.x > 0:
                    x_value = 'more'
                elif current_left.x - previous_left.x < 0:
                    x_value = 'less'
                else:
                    x_value = 'same'
                # Get absolute comparison on y-axis
                if current_left.y - previous_left.y > 0:
                    y_value = 'less'
                elif current_left.y - previous_left.y < 0:
                    y_value = 'more'
                else:
                    y_value = 'same'
                # Encode objects' x-trajectory
                same_x_val = 'True' if x_value == 'same' else 'False'
                more_x_val = 'True' if x_value == 'more' else 'False'
                less_x_val = 'True' if x_value == 'less' else 'False'
                rel_state.add(Fact(name='same-x', type='logical', value=same_x_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='more-x', type='logical', value=more_x_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='less-x', type='logical', value=less_x_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                # Encode objects' y-trajectory
                same_y_val = 'True' if y_value == 'same' else 'False'
                more_y_val = 'True' if y_value == 'more' else 'False'
                less_y_val = 'True' if y_value == 'less' else 'False'
                rel_state.add(Fact(name='same-y', type='logical', value=same_y_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='more-y', type='logical', value=more_y_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='less-y', type='logical', value=less_y_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
    if include_incomplete_states:
        return rel_state
    else:
        objects_to_check = ['enemy_missile_t']
        objects_in_state = [fact.obj2 for fact in rel_state if fact.obj2.endswith('t')]
        rel_state = rel_state if len(set(objects_in_state).intersection(objects_to_check)) > 0 else set()
        return rel_state

def get_state_logical_breakout(current_info, previous_info, include_incomplete_states=False):
    """
    Builds a relational state from current_info and previous_info.
    These are dictionaries from the BreakoutPreprocessor where the 'labels' key
    contains a dictionary of attribute values for all the objects in the screen.

    Args:
        current_info: dictionary of the current time step.
        previous_info: dictionary of the previous time step.
        include_no_ball_states: wheather to consider states where the ball is not present.
    Returns:
       A set of logical relations that hold in the state.
    """

    # Tolerance levels for x and y dimensions.
    x_tolerance = 6
    y_tolerance = 6
    
    # Current screen variables.
    ball_present_current = True if current_info['labels']['ball_x'] != 0 and current_info['labels']['ball_y'] != 0 else False
    player_x_current = current_info['labels']['player_x']
    player_y_current = 210 - 190.5 # invert y axis (constant not given by the dictionary)
    ball_x_current = None if not ball_present_current else current_info['labels']['ball_x']
    ball_y_current = None if not ball_present_current else current_info['labels']['ball_y']
    
    # Invert y axis
    if ball_y_current:
        ball_y_current = 210 - ball_y_current
    
    # Next screen variable.
    if previous_info:
        ball_present_previous = True if previous_info['labels']['ball_x'] != 0 and previous_info['labels']['ball_y'] != 0 else False
        player_x_previous = previous_info['labels']['player_x']
        player_y_previous = 210 - 190.5 # invert y axis (constant not given by the dictionary)
        ball_x_previous = None if not ball_present_previous else previous_info['labels']['ball_x']
        ball_y_previous = None if not ball_present_previous else previous_info['labels']['ball_y']
        # Invert y axis
        if ball_y_previous:
            ball_y_previous = 210 - ball_y_previous
    else:
        ball_present_previous = False

    # Initialize state.
    relational_state = set()

    # Encode palyers's presence (constant)
    presence_player_current = Fact(name='present', type='logical', value='True', obj1='player_t')
    relational_state.add(presence_player_current)
    
    # Build state.
    #  Relations between current objects (ball-dependent).
    if ball_present_current:
        # Encode ball's presence.
        presence_ball_current = Fact(name='present', type='logical', value='True', obj1='ball_t')
        relational_state.add(presence_ball_current)

        # Encode x-relation.
        if abs(player_x_current - ball_x_current) <= x_tolerance:
            relational_state.add(Fact(name='same-x', type='logical', value='True', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        elif (player_x_current - ball_x_current) > x_tolerance:
            relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='more-x', type='logical', value='True', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        elif (player_x_current - ball_x_current) < -x_tolerance:
            relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='less-x', type='logical', value='True', obj1='player_t', obj2='ball_t'))
        
        # Encode y-relation.
        if abs(player_y_current - ball_y_current) <= y_tolerance:
            relational_state.add(Fact(name='same-y', type='logical', value='True', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        elif (player_y_current - ball_y_current) > y_tolerance:
            relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='more-y', type='logical', value='True', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        elif (player_y_current - ball_y_current) < -y_tolerance:
            relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
            relational_state.add(Fact(name='less-y', type='logical', value='True', obj1='player_t', obj2='ball_t'))

        # Encode contact.
        if abs(player_x_current - ball_x_current) <= 11 and abs(player_y_current - ball_y_current) <= 6:
            contact_relation = Fact(name='incontact', type='logical', value='True', obj1='player_t', obj2='ball_t')
        else:
            contact_relation = Fact(name='incontact', type='logical', value='False', obj1='player_t', obj2='ball_t')
        relational_state.add(contact_relation)

    else:
        # Encode ball's presence.
        presence_ball_current = Fact(name='present', type='logical', value='False', obj1='ball_t')
        relational_state.add(presence_ball_current)
        # Encode x-relation.
        relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        # Encode y-relation.
        relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='player_t', obj2='ball_t'))
        # Encode contact.
        relational_state.add(Fact(name='incontact', type='logical', value='False', obj1='player_t', obj2='ball_t'))
    # Relations between sigle objects across time.
    if previous_info:
        # Encode previous player's presence.
        presence_player_previous = Fact(name='present', type='logical', value='True', obj1='player_t-1')
        relational_state.add(presence_player_previous)
        # Encode previous ball's presence.
        if ball_present_previous:
            presence_ball_previous = Fact(name='present', type='logical', value='True', obj1='ball_t-1')
        else:
            presence_ball_previous = Fact(name='present', type='logical', value='False', obj1='ball_t-1')
        relational_state.add(presence_ball_previous)

        # Encode ball relations.
        if ball_present_previous and ball_present_current:            
            # Encode ball's x-trajectory.
            if ball_x_current > ball_x_previous:
                relational_state.add(Fact(name='more-x', type='logical', value='True', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            elif ball_x_current < ball_x_previous:
                relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='less-x', type='logical', value='True', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            elif ball_x_current == ball_x_previous:
                relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='same-x', type='logical', value='True', obj1='ball_t', obj2='ball_t-1'))

            # Encode ball's y-trajectory.
            if ball_y_current > ball_y_previous:
                relational_state.add(Fact(name='more-y', type='logical', value='True', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            elif ball_y_current < ball_y_previous:
                relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='less-y', type='logical', value='True', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            elif ball_y_current == ball_y_previous:
                relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
                relational_state.add(Fact(name='same-y', type='logical', value='True', obj1='ball_t', obj2='ball_t-1'))
        else:
            # Encode ball's x-trajectory
            relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            # Encode ball's y-trajectory
            relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
            relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))

        # Encode player's x-trajectory.
        if player_x_current > player_x_previous:
            relational_state.add(Fact(name='more-x', type='logical', value='True', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        elif player_x_current < player_x_previous:
            relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='less-x', type='logical', value='True', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        elif player_x_current == player_x_previous:
            relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='same-x', type='logical', value='True', obj1='player_t', obj2='player_t-1'))
    
        # Encode player's y-trajectory (CONSTANT).
        if player_y_current > player_y_previous:
            relational_state.add(Fact(name='more-y', type='logical', value='True', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        elif player_y_current < player_y_previous:
            relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='less-y', type='logical', value='True', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        elif player_y_current == player_y_previous:
            relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
            relational_state.add(Fact(name='same-y', type='logical', value='True', obj1='player_t', obj2='player_t-1'))
    else:
        # Encode previous player's presence.
        presence_player_previous = Fact(name='present', type='logical', value='False', obj1='player_t-1')
        relational_state.add(presence_player_previous)
        # Encode previous ball's presence.
        presence_ball_previous = Fact(name='present', type='logical', value='False', obj1='ball_t-1')
        relational_state.add(presence_ball_previous)

        # Encode ball's x-trajectory
        relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
        relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
        relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
        # Encode ball's y-trajectory
        relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
        relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
        relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='ball_t', obj2='ball_t-1'))
        # Encode player's x-trajectory
        relational_state.add(Fact(name='more-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        relational_state.add(Fact(name='less-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        relational_state.add(Fact(name='same-x', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        # Encode player's y-trajectory
        relational_state.add(Fact(name='more-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        relational_state.add(Fact(name='less-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
        relational_state.add(Fact(name='same-y', type='logical', value='False', obj1='player_t', obj2='player_t-1'))
    
    if include_incomplete_states:
        return relational_state
    else:
        fact1 = Fact(name='present', type='logical', value='False', obj1='ball_t')
        fact2 = Fact(name='present', type='logical', value='False', obj1='ball_t-1')
        if fact1 in relational_state or fact2 in relational_state:
            return set()
        else:
            return relational_state 

def get_state_comparative_breakout(current_info, previous_info, include_incomplete_states=False):
    """
    Build a relational state from current_info and previous_info.
    These are dictionaries from the BreakoutPreprocessor where the 'labels' key
    contains a dictionary of attribute values for all the objects in the screen.

    Args:
        current_info: dictionary of the current time step.
        previous_info: dictionary of the previous time step.
        include_no_ball_states: wheather to consider states where the ball is not present.
    Returns:
        A set of mostly comparative relations that hold in the state. 
    """
    # Tolerance levels for x and y dimensions.
    x_tolerance = 6
    y_tolerance = 6
    
    # Current screen variables. Manually correct RAM information.
    ball_present_current = True if current_info['labels']['ball_x'] != 0 and current_info['labels']['ball_y'] != 0 else False
    player_x_current = current_info['labels']['player_x']
    player_y_current = 210 - 190.5 # invert y axis (constant)
    ball_x_current = None if not ball_present_current else current_info['labels']['ball_x']
    ball_y_current = None if not ball_present_current else current_info['labels']['ball_y']
    # Invert y axis
    if ball_y_current:
        ball_y_current = 210 - ball_y_current
    
    # Next screen variable.
    if previous_info:
        ball_present_previous = True if previous_info['labels']['ball_x'] != 0 and previous_info['labels']['ball_y'] != 0 else False
        player_x_previous = previous_info['labels']['player_x']
        player_y_previous = 210 - 190.5 # invert y axis (constant not given by the dictionary)
        ball_x_previous = None if not ball_present_previous else previous_info['labels']['ball_x']
        ball_y_previous = None if not ball_present_previous else previous_info['labels']['ball_y']
        # Invert y axis
        if ball_y_previous:
            ball_y_previous = 210 - ball_y_previous
    else:
        ball_present_previous = False

    # Initialize state.
    relational_state = set()

    # Encode palyers's presence (constant)
    presence_player_current = Fact(name='present', type='logical', value='True', obj1='player_t')
    relational_state.add(presence_player_current)
    
    # Build state.
    #  Relations between current objects (ball-dependent).
    if ball_present_current:
        # Encode ball's presence.
        presence_ball_current = Fact(name='present', type='logical', value='True', obj1='ball_t')
        relational_state.add(presence_ball_current)

        # Encode x-relation.
        if abs(player_x_current - ball_x_current) <= x_tolerance:
            x_relation = Fact(name='x', type='comparative', value='same', obj1='player_t', obj2='ball_t')
        elif (player_x_current - ball_x_current) > x_tolerance:
            x_relation = Fact(name='x', type='comparative', value='more', obj1='player_t', obj2='ball_t')
        elif (player_x_current - ball_x_current) < -x_tolerance:
            x_relation = Fact(name='x', type='comparative', value='less', obj1='player_t', obj2='ball_t')
        relational_state.add(x_relation)
        
        # Encode y-relation.
        if abs(player_y_current - ball_y_current) <= y_tolerance:
            y_relation = Fact(name='y', type='comparative', value='same', obj1='player_t', obj2='ball_t')
        elif (player_y_current - ball_y_current) > y_tolerance:
            y_relation = Fact(name='y', type='comparative', value='more', obj1='player_t', obj2='ball_t')
        elif (player_y_current - ball_y_current) < -y_tolerance:
            y_relation = Fact(name='y', type='comparative', value='less', obj1='player_t', obj2='ball_t')
        relational_state.add(y_relation)

        # Encode contact.
        if abs(player_x_current - ball_x_current) <= 11 and abs(player_y_current - ball_y_current) <= 6:
            contact_relation = Fact(name='incontact', type='logical', value='True', obj1='player_t', obj2='ball_t')
        else:
            contact_relation = Fact(name='incontact', type='logical', value='False', obj1='player_t', obj2='ball_t')
        relational_state.add(contact_relation)

    else:
        # Encode ball's presence.
        presence_ball_current = Fact(name='present', type='logical', value='False', obj1='ball_t')
        relational_state.add(presence_ball_current)
    # Relations between sigle objects across time.
    if previous_info:
        # Encode previous player's presence.
        presence_player_previous = Fact(name='present', type='logical', value='True', obj1='player_t-1')
        relational_state.add(presence_player_previous)
        # Encode previous ball's presence.
        if ball_present_previous:
            presence_ball_previous = Fact(name='present', type='logical', value='True', obj1='ball_t-1')
        else:
            presence_ball_previous = Fact(name='present', type='logical', value='False', obj1='ball_t-1')
        relational_state.add(presence_ball_previous)

        # Encode ball relations.
        if ball_present_previous and ball_present_current:            
            # Encode ball's x-trajectory.
            if ball_x_current > ball_x_previous:
                tx_relation_ball = Fact(name='x', type='comparative', value='more', obj1='ball_t', obj2='ball_t-1')
            elif ball_x_current < ball_x_previous:
                tx_relation_ball = Fact(name='x', type='comparative', value='less', obj1='ball_t', obj2='ball_t-1')
            elif ball_x_current == ball_x_previous:
                tx_relation_ball = Fact(name='x', type='comparative', value='same', obj1='ball_t', obj2='ball_t-1')
            relational_state.add(tx_relation_ball)

            # Encode ball's y-trajectory.
            if ball_y_current > ball_y_previous:
                ty_relation_ball = Fact(name='y', type='comparative', value='more', obj1='ball_t', obj2='ball_t-1')
            elif ball_y_current < ball_y_previous:
                ty_relation_ball = Fact(name='y', type='comparative', value='less', obj1='ball_t', obj2='ball_t-1')
            elif ball_y_current == ball_y_previous:
                ty_relation_ball = Fact(name='y', type='comparative', value='same', obj1='ball_t', obj2='ball_t-1')
            relational_state.add(ty_relation_ball)

        # Encode player's x-trajectory.
        if player_x_current > player_x_previous:
            tx_relation_player = Fact(name='x', type='comparative', value='more', obj1='player_t', obj2='player_t-1')
        elif player_x_current < player_x_previous:
            tx_relation_player = Fact(name='x', type='comparative', value='less', obj1='player_t', obj2='player_t-1')
        elif player_x_current == player_x_previous:
            tx_relation_player = Fact(name='x', type='comparative', value='same', obj1='player_t', obj2='player_t-1')
        relational_state.add(tx_relation_player)
    
        # Encode player's y-trajectory (CONSTANT).
        if player_y_current > player_y_previous:
            ty_relation_player = Fact(name='y', type='comparative', value='more', obj1='player_t', obj2='player_t-1')
        elif player_y_current < player_y_previous:
            ty_relation_player = Fact(name='y', type='comparative', value='less', obj1='player_t', obj2='player_t-1')
        elif player_y_current == player_y_previous:
            ty_relation_player = Fact(name='y', type='comparative', value='same', obj1='player_t', obj2='player_t-1')
        relational_state.add(ty_relation_player)
    else:
        # Encode previous player's presence.
        presence_player_previous = Fact(name='present', type='logical', value='False', obj1='player_t-1')
        relational_state.add(presence_player_previous)
        # Encode previous ball's presence.
        presence_ball_previous = Fact(name='present', type='logical', value='False', obj1='ball_t-1')
        relational_state.add(presence_ball_previous)

    if include_incomplete_states:
        return relational_state
    else:
        return relational_state if len(relational_state) == 11 else set()

def get_state_comparative_pong(current_info, previous_info, include_incomplete_states=False):
    """
    Infers relations between objects from current_info and previous_info.
    These are dictionaries from the PongPreprocessor where the 'labels' key
    contains a dictionary of attribute values for all the objects in the screen.

    Args:
        current_info: dictionary of the current time step.
        previous_info: dictionary of the previous time step.

    Returns:
        A set of mostly comparative relations that hold in the state. 
    """
    # Tolerance levels for x and y dimensions
    x_tol_ball = 4
    y_tol_ball = 4
    x_tol_paddle = 4
    y_tol_paddle = 4

    # Current objects (player is always present)
    current_objects = ['player']
    if current_info['labels']['ball_x'] != 0 and current_info['labels']['ball_y'] != 0:
        current_objects.append('ball')
    if current_info['labels']['enemy_x'] != 0 and current_info['labels']['enemy_y'] != 0:
        current_objects.append('enemy')
    # Previous objects
    if previous_info:
        previous_objects = ['player']
        if previous_info['labels']['ball_x'] != 0 and previous_info['labels']['ball_y'] != 0:
            previous_objects.append('ball')
        if previous_info['labels']['enemy_x'] != 0 and previous_info['labels']['enemy_y'] != 0:
            previous_objects.append('enemy')
    else:
        previous_objects = []

    # Initialize state
    rel_state = set()
    # Get combinations of objects and order according to object_hierarchy
    object_hierarchy = ['player', 'ball', 'enemy']
    object_combinations = list(itertools.combinations(object_hierarchy, 2))

    # Relations between objects in the current screen
    for obj1, obj2 in object_combinations:
        # Current screen variables
        if obj1 in current_objects and obj2 in current_objects:
            # Get x_value
            x_tol = x_tol_ball if (obj1 == 'ball' or obj2 == 'ball') else x_tol_paddle
            obj1_x = current_info['labels'][f'{obj1}_x']
            obj2_x = current_info['labels'][f'{obj2}_x']
            if abs(obj1_x - obj2_x) <= x_tol:
                x_value = 'same'
            elif (obj1_x - obj2_x) > x_tol:
                x_value = 'more'
            elif (obj1_x - obj2_x) < -x_tol:
                x_value = 'less'
            # Get y_value
            y_tol = y_tol_ball if (obj1 == 'ball' or obj2 == 'ball') else y_tol_paddle
            obj1_y = 210 - current_info['labels'][f'{obj1}_y']
            obj2_y = 210 - current_info['labels'][f'{obj2}_y']
            if abs(obj1_y - obj2_y) <= y_tol:
                y_value = 'same'
            elif (obj1_y - obj2_y) > y_tol:
                y_value = 'more'
            elif (obj1_y - obj2_y) < -y_tol:
                y_value = 'less'
            # Encode x-relation
            rel_state.add(Fact(name='x', type='comparative', value=x_value, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            # Encode y-relation
            rel_state.add(Fact(name='y', type='comparative', value=y_value, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            # Encode contact
            x_diff = obj1_x - obj2_x
            y_diff = obj1_y - obj2_y
            if obj1 == 'player' and obj2 == 'enemy':
                if abs(y_diff) <= 8 and abs(x_diff) <= 6:
                    rel_state.add(Fact(name='incontact', type='logical', value='True', obj1='player_t', obj2='enemy_t'))
                else:
                    rel_state.add(Fact(name='incontact', type='logical', value='False', obj1='player_t', obj2='enemy_t'))
            # player-ball, ball-enemy
            else:
                if abs(y_diff) <= 8 and x_diff >= 0 and x_diff <= 6:
                    rel_state.add(Fact(name='incontact', type='logical', value='True', obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
                else:
                    rel_state.add(Fact(name='incontact', type='logical', value='False', obj1=f"{obj1}_t", obj2=f"{obj2}_t"))

    # Relations between sigle objects across time
    if previous_info:
        for obj in object_hierarchy:
            if obj in current_objects and obj in previous_objects:
                current_x = current_info['labels'][f'{obj}_x']
                previous_x = previous_info['labels'][f'{obj}_x']
                current_y = 210 - current_info['labels'][f'{obj}_y']
                previous_y = 210 - previous_info['labels'][f'{obj}_y']
                # Get absolute comparison on x-axis
                if current_x - previous_x > 0:
                    x_value = 'more'
                elif current_x - previous_x < 0:
                    x_value = 'less'
                else:
                    x_value = 'same'
                # Get absolute comparison on y-axis
                if current_y - previous_y > 0:
                    y_value = 'less'
                elif current_y - previous_y < 0:
                    y_value = 'more'
                else:
                    y_value = 'same'
                # Encode objects' x-trajectory
                rel_state.add(Fact(name='x', type='comparative', value=x_value, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                # Encode objects' y-trajectory
                rel_state.add(Fact(name='y', type='comparative', value=y_value, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
    
    if include_incomplete_states:
        return rel_state
    else:
        return rel_state if len(rel_state) == 15 else set()

def get_state_logical_pong(current_info, previous_info, include_incomplete_states=False):
    """
    Infers relations between objects from current_info and previous_info.
    These are dictionaries from the PongPreprocessor where the 'labels' key
    contains a dictionary of attribute values for all the objects in the screen.

    Args:
        current_info: dictionary of the current time step.
        previous_info: dictionary of the previous time step.

    Returns:
        A set of logical relations that hold in the state. 
    """
    # Tolerance levels for x and y dimensions
    x_tol_ball = 4
    y_tol_ball = 4
    x_tol_paddle = 4
    y_tol_paddle = 4

    # Current objects (player is always present)
    current_objects = ['player']
    if current_info['labels']['ball_x'] != 0 and current_info['labels']['ball_y'] != 0:
        current_objects.append('ball')
    if current_info['labels']['enemy_x'] != 0 and current_info['labels']['enemy_y'] != 0:
        current_objects.append('enemy')
    # Previous objects
    if previous_info:
        previous_objects = ['player']
        if previous_info['labels']['ball_x'] != 0 and previous_info['labels']['ball_y'] != 0:
            previous_objects.append('ball')
        if previous_info['labels']['enemy_x'] != 0 and previous_info['labels']['enemy_y'] != 0:
            previous_objects.append('enemy')
    else:
        previous_objects = []

    # Initialize state
    rel_state = set()
    # Get combinations of objects and order according to object_hierarchy
    object_hierarchy = ['player', 'ball', 'enemy']
    object_combinations = list(itertools.combinations(object_hierarchy, 2))

    # Relations between objects in the current screen
    for obj1, obj2 in object_combinations:
        # Current screen variables
        if obj1 in current_objects and obj2 in current_objects:
            # Get x_value
            x_tol = x_tol_ball if (obj1 == 'ball' or obj2 == 'ball') else x_tol_paddle
            obj1_x = current_info['labels'][f'{obj1}_x']
            obj2_x = current_info['labels'][f'{obj2}_x']
            if abs(obj1_x - obj2_x) <= x_tol:
                x_value = 'same'
            elif (obj1_x - obj2_x) > x_tol:
                x_value = 'more'
            elif (obj1_x - obj2_x) < -x_tol:
                x_value = 'less'
            # Get y_value
            y_tol = y_tol_ball if (obj1 == 'ball' or obj2 == 'ball') else y_tol_paddle
            obj1_y = 210 - current_info['labels'][f'{obj1}_y']
            obj2_y = 210 - current_info['labels'][f'{obj2}_y']
            if abs(obj1_y - obj2_y) <= y_tol:
                y_value = 'same'
            elif (obj1_y - obj2_y) > y_tol:
                y_value = 'more'
            elif (obj1_y - obj2_y) < -y_tol:
                y_value = 'less'
            # Encode x-relation
            same_x_val = 'True' if x_value == 'same' else 'False'
            more_x_val = 'True' if x_value == 'more' else 'False'
            less_x_val = 'True' if x_value == 'less' else 'False'
            rel_state.add(Fact(name='same-x', type='logical', value=same_x_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='more-x', type='logical', value=more_x_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='less-x', type='logical', value=less_x_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            # Encode y-relation
            same_y_val = 'True' if y_value == 'same' else 'False'
            more_y_val = 'True' if y_value == 'more' else 'False'
            less_y_val = 'True' if y_value == 'less' else 'False'
            rel_state.add(Fact(name='same-y', type='logical', value=same_y_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='more-y', type='logical', value=more_y_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            rel_state.add(Fact(name='less-y', type='logical', value=less_y_val, obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
            # Encode contact
            x_diff = obj1_x - obj2_x
            y_diff = obj1_y - obj2_y
            if obj1 == 'player' and obj2 == 'enemy':
                if abs(y_diff) <= 8 and abs(x_diff) <= 6:
                    rel_state.add(Fact(name='incontact', type='logical', value='True', obj1='player_t', obj2='enemy_t'))
                else:
                    rel_state.add(Fact(name='incontact', type='logical', value='False', obj1='player_t', obj2='enemy_t'))
            # player-ball, ball-enemy
            else:
                if abs(y_diff) <= 8 and x_diff >= 0 and x_diff <= 6:
                    rel_state.add(Fact(name='incontact', type='logical', value='True', obj1=f"{obj1}_t", obj2=f"{obj2}_t"))
                else:
                    rel_state.add(Fact(name='incontact', type='logical', value='False', obj1=f"{obj1}_t", obj2=f"{obj2}_t"))

    # Relations between sigle objects across time
    if previous_info:
        for obj in object_hierarchy:
            if obj in current_objects and obj in previous_objects:
                current_x = current_info['labels'][f'{obj}_x']
                previous_x = previous_info['labels'][f'{obj}_x']
                current_y = 210 - current_info['labels'][f'{obj}_y']
                previous_y = 210 - previous_info['labels'][f'{obj}_y']
                # Get absolute comparison on x-axis
                if current_x - previous_x > 0:
                    x_value = 'more'
                elif current_x - previous_x < 0:
                    x_value = 'less'
                else:
                    x_value = 'same'
                # Get absolute comparison on y-axis
                if current_y - previous_y > 0:
                    y_value = 'less'
                elif current_y - previous_y < 0:
                    y_value = 'more'
                else:
                    y_value = 'same'
                # Encode objects' x-trajectory
                same_x_val = 'True' if x_value == 'same' else 'False'
                more_x_val = 'True' if x_value == 'more' else 'False'
                less_x_val = 'True' if x_value == 'less' else 'False'
                rel_state.add(Fact(name='same-x', type='logical', value=same_x_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='more-x', type='logical', value=more_x_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='less-x', type='logical', value=less_x_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                # Encode objects' y-trajectory
                same_y_val = 'True' if y_value == 'same' else 'False'
                more_y_val = 'True' if y_value == 'more' else 'False'
                less_y_val = 'True' if y_value == 'less' else 'False'
                rel_state.add(Fact(name='same-y', type='logical', value=same_y_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='more-y', type='logical', value=more_y_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
                rel_state.add(Fact(name='less-y', type='logical', value=less_y_val, obj1=f"{obj}_t", obj2=f"{obj}_t-1"))
    if include_incomplete_states:
        return rel_state
    else:
        return rel_state if len(rel_state) == 39 else set()
