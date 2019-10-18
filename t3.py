#python 3.5.2

class Solution:
    
    def numIslands(self, grid):
        # your code here
        row, col = len(grid), len(grid[0])
        max_area = 0
        for height in range(row,0,-1):
            for i in range(0, 1+row-height):
                for width in range(col,0,-1):
                    for j in range(0, 1+col-width):
                        if all(all(grid[i+r][j+c] == '1' \
                                for c in range(width)) \
                                for r in range(height)):
                            max_area = max(height*width, max_area)

        return max_area

    
def get_matrix():
    row = int(input())
    col = int(input())
    grid = [["0"]*col]*row

    for i in range(row):
        line = input()
        grid[i] = list(line)[0:col]
    return grid

        
if __name__ == "__main__":

    matrix = ([list('10111'),
            list('10111'),
            list('11111'),
            list('10010')],
            [list('10100'),
            list('10111'),
            list('11111'),
            list('10010')])

    sol = Solution()
    # matrix = get_matrix()
    for mat in matrix:
        islands = sol.numIslands(mat)
        print(islands)
