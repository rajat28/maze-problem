import numpy as np
import matplotlib.pyplot as plt
import random

with_path = 0
without_path = 0

for i in range(10):
    maze = np.array([
        [random.randrange(0., 2., 1) for _ in range(20)],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0.],
        [random.randrange(0., 2., 1) for _ in range(20)],
        [0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.]
    ])

    maze[0][0] = 1.
    duplicate_maze = np.copy(maze)
    flag = 0


    def search_path(x, y):
        if x == 19 and y == 19:
            print('Found at %d,%d' % (x, y))
            save_image(maze, x, y)
            global flag
            flag = 1


        elif duplicate_maze[x][y] == 0:
            print('Wall at %d,%d' % (x, y))
            return False

        elif duplicate_maze[x][y] == 3:
            print('Visited at %d,%d' % (x, y))
            return False

        print('Visiting %d,%d' % (x, y))

        duplicate_maze[x][y] = 3

        if ((x < len(duplicate_maze) - 1 and search_path(x + 1, y))
                or (y > 0 and search_path(x, y - 1))
                or (x > 0 and search_path(x - 1, y))
                or (y < len(duplicate_maze) - 1 and search_path(x, y + 1))):
            return True

        return False


    def save_image(maze, x, y):
        plt.grid('on')
        qmaze = np.array(maze)
        nrows, ncols = qmaze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        canvas = np.copy(maze)
        img = plt.imshow(canvas, interpolation='none', cmap='gray')
        if x == 19 and y == 19:
            global with_path
            with_path += 1
            plt.savefig(
                "D:/FIT Study Material/Semester 2/Artificial Intelligence/Project - Maze Problem/Dataset/Testing"
                "/Maze with path " + str(with_path))

        else:
            global without_path
            without_path += 1
            plt.savefig(
                "D:/FIT Study Material/Semester 2/Artificial Intelligence/Project - Maze Problem/Dataset/Testing"
                "/Maze without path " + str(without_path))
        return img


    search_path(0, 0)

    if flag != 1:
        save_image(maze, 0, 0)