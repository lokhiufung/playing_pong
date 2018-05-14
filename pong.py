import pygame
from pygame.locals import *
import sys
import cv2

import numpy as np

# frames per second
fps = 30
# global variables for the whole pygame
width = 600
height = 600
LineThickness = 10
PaddleSize = 200
PaddleOffset = 20  # how far the paddle is from the arena

PaddleSpeed = 5
BallSpeed = 5

# colors
black = (0, 0, 0)
white = (255, 255, 255)

pygame.font.init()
DisplaySurf = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
pygame.display.set_caption('Pong')


def DisplayScore(score):
    BasicFrontSize = 20
    BasicFront = pygame.font.Font('freesansbold.ttf', BasicFrontSize)
    ResultSurf = BasicFront.render('Score = {}'.format(score), True, white)
    # create a new rectangle with the same size of the surface
    ResultRect = ResultSurf.get_rect()
    ResultRect.topleft = (50, 25)

    DisplaySurf.blit(ResultSurf, ResultRect)


def DrawArena():
    """
    This funciton draws a boundary around the edge of our window.
    """
    DisplaySurf.fill(black)
    # DisplaySurf is the surface we want to draw
    # white is the color we want to fill in
    # third argument defines a rectanle
    # (0, 0) is the top left hand corner
    # (width, height) is the low right hand corner
    # when we define thickness of arena, we actually locate the bounary with
    # its middle position. Half will be outside and half will be inside
    pygame.draw.rect(DisplaySurf, white,
                     ((0, 0), (width, height)), LineThickness*2)

    # This draws the centre line of the court
    # no mean with LineThickness/4 ... just want a very thin line
    pygame.draw.line(DisplaySurf, white,
                     (width/2, 0), (width/2, height), int(LineThickness/4))


def DrawPaddle1(PlayerOnePosition):
    """
    paddle: pygame Rect object
    """
    paddle = pygame.Rect(LineThickness + PaddleOffset, PlayerOnePosition,
                         LineThickness, PaddleSize)

    pygame.draw.rect(DisplaySurf, white, paddle)


def DrawPaddle2(PlayerTwoPosition):
    """
    paddle: pygame Rect object
    """
    paddle = pygame.Rect(width - 2*LineThickness - PaddleOffset,
                         PlayerTwoPosition, LineThickness, PaddleSize)

    pygame.draw.rect(DisplaySurf, white, paddle)


def DrawBall(ball_x, ball_y):
    """
    ball: pygame Rect object
    """
    ball = pygame.Rect(ball_x, ball_y, LineThickness, LineThickness)
    pygame.draw.rect(DisplaySurf, white, ball)


def Updateball(ball_x, ball_y, BallDirX, BallDirY, PlayerOnePosition,
               PlayerTwoPosition, score):

    ball_x += BallDirX * BallSpeed
    ball_y += BallDirY * BallSpeed

    # ceiling and ground
    if (ball_y <= LineThickness or
        ball_y + LineThickness >= height - LineThickness):

        BallDirY = BallDirY * -1

    if BallDirX == -1:
        # left hand side
        if (PaddleOffset + 2 * LineThickness == ball_x and
            ball_y + LineThickness >= PlayerOnePosition and
                PlayerOnePosition + PaddleSize >= ball_y):
            BallDirX = BallDirX * -1
            score = +1
    elif BallDirX == 1:
        # right hand side
        if (width - PaddleOffset - 2*LineThickness == ball_x and
            ball_y + LineThickness >= PlayerTwoPosition and
                PlayerTwoPosition + PaddleSize >= ball_y):
            BallDirX = BallDirX * -1

    if ball_x <= LineThickness:
        score = -1
        BallDirX = BallDirX * -1

    if ball_x >= width - 2 * LineThickness:
        score = +1
        BallDirX = BallDirX * -1

    return [ball_x, ball_y, BallDirX, BallDirY, PlayerOnePosition, PlayerTwoPosition, score]


def UpdatePaddle1(action, PlayerOnePosition):

    if action == 2:
        PlayerOnePosition += PaddleSpeed

    elif action == 0:
        PlayerOnePosition -= PaddleSpeed

    elif not (action in [0, 1, 2]):
        raise ValueError('No such action!')

    # lower bound
    if PlayerOnePosition + PaddleSize > height - LineThickness:
        PlayerOnePosition = height - LineThickness - PaddleSize
    # upper bound
    if PlayerOnePosition < LineThickness:
        PlayerOnePosition = LineThickness

    return PlayerOnePosition


def UpdatePaddle2(PlayerTwoPosition, BallDirX, ball_y):


    # if BallDirX == 1:
    if PlayerTwoPosition + PaddleSize / 2 < ball_y + LineThickness / 2:
        PlayerTwoPosition += BallSpeed
    elif PlayerTwoPosition + PaddleSize / 2 > ball_y + LineThickness / 2:
        PlayerTwoPosition -= BallSpeed

    if PlayerTwoPosition + PaddleSize > height - LineThickness:
        PlayerTwoPosition = height - LineThickness - PaddleSize
    # upper bound
    if PlayerTwoPosition < LineThickness:
        PlayerTwoPosition = LineThickness

    return PlayerTwoPosition


class Pong():
    def __init__(self, mode='low_dims'):
        self.mode = mode
        self.ball_x = width/2 - LineThickness/2
        self.ball_y = height/2 - LineThickness/2
        self.PlayerOnePosition = (height - PaddleSize)/2
        self.PlayerTwoPosition = (height - PaddleSize)/2

        self.BallDirX = -1
        self.BallDirY = -1

        self.total_score = 0

    def GetPresentFrame(self, normalize=False):
        # throw away all events, only internal events
        pygame.event.pump()
        DrawArena()
        DrawPaddle1(self.PlayerOnePosition)
        DrawPaddle2(self.PlayerTwoPosition)

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        low_dim_data = [self.ball_x, self.ball_y,
                        self.BallDirX, self.BallDirY, self.PlayerOnePosition]
        if normalize:
            low_dim_data = [self.ball_x/width, self.ball_y/height,
                            self.BallDirX, self.BallDirY,
                            self.PlayerOnePosition/height]

        # update screen
        DisplayScore(self.total_score)

        pygame.display.flip()
        clock.tick(fps)
        if self.mode == 'high_dims':
            return image_data
        elif self.mode == 'low_dims':
            return np.array([low_dim_data]).reshape((5, 1))
        else:
            raise ValueError

    def GetNextFrame(self, action, normalize=False):
        pygame.event.pump()
        DrawArena()
        score = 0

        self.PlayerOnePosition = UpdatePaddle1(action, self.PlayerOnePosition)
        DrawPaddle1(self.PlayerOnePosition)

        self.PlayerTwoPosition = UpdatePaddle2(self.PlayerTwoPosition,
                                               self.BallDirY, self.ball_y)
        DrawPaddle2(self.PlayerTwoPosition)

        [self.ball_x, self.ball_y, self.BallDirX,
            self.BallDirY, self.PlayerOnePosition, self.PlayerTwoPosition, score] = Updateball(self.ball_x, self.ball_y,
                                                    self.BallDirX,
                                                    self.BallDirY,
                                                    self.PlayerOnePosition,
                                                    self.PlayerTwoPosition,
                                                    score)
        DrawBall(self.ball_x, self.ball_y)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        self.total_score += score
        DisplayScore(self.total_score)
        low_dim_data = [self.ball_x, self.ball_y,
                        self.BallDirX, self.BallDirY, self.PlayerOnePosition]
        if normalize:
            low_dim_data = [self.ball_x/width, self.ball_y/height,
                            self.BallDirX, self.BallDirY,
                            self.PlayerOnePosition/height]
        pygame.display.flip()
        clock.tick(fps)
        if self.mode == 'high_dims':
            return score, image_data
        elif self.mode == 'low_dims':
            return score, np.array(low_dim_data).reshape((5, 1))


def main():

    input_dims = [84, 84, 1]
    pong = Pong(mode='high_dims')
    frame = pong.GetPresentFrame()
    frame = cv2.cvtColor(cv2.resize(frame, (input_dims[0], input_dims[1])), cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    state_1 = np.stack((frame, frame, frame, frame), axis=2)
    state_1 = state_1.reshape((1, input_dims[0], input_dims[1], 4))

    print(state_1.shape)
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        action = np.random.randint(3)
        score, frame = pong.GetNextFrame(action)

        frame = cv2.cvtColor(cv2.resize(frame, (input_dims[0], input_dims[1])), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (1, input_dims[0], input_dims[1], 1))
        state_2 = np.append(frame, state_1[:, :, :, 0:3], axis=3)

        print(state_2.shape)
        state_1 = state_2


if __name__ == '__main__':
    main()
