import os, time, random, sys

try:
    import curses
except ImportError:
    sys.exit("Please install the curses library to run this program.")

def create_char_matrix(w, h):
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return [[random.choice(characters) for _ in range(w)] for _ in range(h)]

def matrix(window):
    w, h = os.get_terminal_size()
    char_matrix = create_char_matrix(w, h)
    drops = [1] * w

    while True:
        for i in range(len(drops)):
            for j in range(drops[i]):
                if j < drops[i] - 2:
                    window.addstr(j, i, char_matrix[j][i], curses.color_pair(2))
                elif j == drops[i] - 2:
                    window.addstr(j, i, char_matrix[j][i], curses.color_pair(3))
                else:
                    window.addstr(j, i, char_matrix[j][i], curses.color_pair(4))

            if drops[i] * 2 > h or random.random() > 0.975:
                drops[i] = 0
                char_matrix = create_char_matrix(w, h)

            drops[i] += 1
            window.timeout(50)
        window.refresh()

def main(window):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
    window.bkgd(" ", curses.color_pair(1))
    matrix(window)

if __name__ == "__main__":
    curses.wrapper(main)
