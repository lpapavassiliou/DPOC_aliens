# decompyle3 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.13 (default, Mar 28 2022, 11:38:47) 
# [GCC 7.5.0]
# Embedded file name: C:\Users\utente\Documents\ETH\Autumn_semester_2022-2023\DPOC\DPOC_ProgEx22\DPOC_PE_2022_Python\MakePlots.py
# Compiled at: 2022-11-09 07:56:28
# Size of source mod 2**32: 17495 bytes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from Constants import *

def PlotCell(map_world, *args):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(9.5, 9.5, forward=True)
    CustomMap(map_world, Constants.UPPER, axs[Constants.UPPER], args[0])
    axs[Constants.UPPER].set_title('Upper World')
    axs[Constants.UPPER].set_aspect('equal')
    CustomMap(map_world, Constants.LOWER, axs[Constants.LOWER], args[0])
    axs[Constants.LOWER].set_title('Lower World')
    axs[Constants.LOWER].set_aspect('equal')
    fig.suptitle(f"Map (width={map_world.shape[0]}, height={map_world.shape[1]})", fontsize=30)
    plt.draw()
    plt.pause(0.001)

def MakePlots(map_world, *args):
    if len(args) < 2:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(9.5, 9.5, forward=True)
        PlotMap(map_world, Constants.UPPER, axs[Constants.UPPER])
        axs[Constants.UPPER].set_title('Upper World')
        axs[Constants.UPPER].set_aspect('equal')
        PlotMap(map_world, Constants.LOWER, axs[Constants.LOWER])
        axs[Constants.LOWER].set_title('Lower World')
        axs[Constants.LOWER].set_aspect('equal')
        fig.suptitle(f"Map (width={map_world.shape[0]}, height={map_world.shape[1]})", fontsize=30)
        plt.draw()
        plt.pause(0.001)
    else:
        stateSpace = args[0]
        stateSpace = np.array(stateSpace)
        J_opt = args[1]
        u = args[2]
        terminal_state_index = args[3]
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(9.5, 9.5, forward=True)
        stateSpace = np.delete(stateSpace, terminal_state_index, 0)
        J_opt = np.delete(J_opt, terminal_state_index)
        u = np.delete(u, terminal_state_index)
        J_max = np.max(J_opt)
        with_gems_upper = np.where(np.equal(stateSpace[:, 2], Constants.GEMS) & np.equal(stateSpace[:, 3], Constants.UPPER))[0]
        with_gems_lower = np.where(np.equal(stateSpace[:, 2], Constants.GEMS) & np.equal(stateSpace[:, 3], Constants.LOWER))[0]
        without_gems_upper = np.where(np.equal(stateSpace[:, 2], Constants.EMPTY) & np.equal(stateSpace[:, 3], Constants.UPPER))[0]
        without_gems_lower = np.where(np.equal(stateSpace[:, 2], Constants.EMPTY) & np.equal(stateSpace[:, 3], Constants.LOWER))[0]
        PlotMap(map_world, Constants.UPPER, axs[Constants.EMPTY][Constants.UPPER], stateSpace[without_gems_upper, :], J_opt[without_gems_upper], u[without_gems_upper], 'Upper World without Gems', J_max)
        PlotMap(map_world, Constants.UPPER, axs[Constants.GEMS][Constants.UPPER], stateSpace[with_gems_upper, :], J_opt[with_gems_upper], u[with_gems_upper], 'Upper World with gems', J_max)
        PlotMap(map_world, Constants.LOWER, axs[Constants.EMPTY][Constants.LOWER], stateSpace[without_gems_lower, :], J_opt[without_gems_lower], u[without_gems_lower], 'Lower World without Gems', J_max)
        PlotMap(map_world, Constants.LOWER, axs[Constants.GEMS][Constants.LOWER], stateSpace[with_gems_lower, :], J_opt[with_gems_lower], u[with_gems_lower], 'Lower World with gems', J_max)
        fig.suptitle('Solution', fontsize=30)
        plt.draw()
        plt.pause(0.001)


def PlotMap(map_world, *args):
    """
      Plot a map, the costs for each cell and the control action in
      each cell.

        Input arguments:
          map:
              A (M x N)-matrix describing the terrain of the estate map.
              Positive values indicate cells that are inaccessible (e.g.
              trees, bushes or the mansion) and negative values indicate
              ponds or pools.

          *args (optional):
                Input argument list:
                1:      A (K x 2)-matrix 'stateSpace', where each row
                      represents an element of the state space.
                2:  A (K x 1 )-matrix 'J' containing the optimal cost-to-go
                        for each element of the state space.
                3:  A (K x 1 )-matrix containing the index of the optimal
                        control input for each element of the state space.
                  4:  Title
    """
    obstacleColor = [
     0.39215686274509803, 0.39215686274509803, 0.39215686274509803]
    portalColor = [
     0.19607843137254902, 0.803921568627451, 0.19607843137254902]
    alienColor = [
     0.9490196078431372, 0.23921568627450981, 0.0]
    mineColor = [
     1.0, 0.9019607843137255, 0.0]
    labColor = [
     0.8313725490196079, 0.16470588235294117, 1.0]
    baseColor = [
     0.4392156862745098, 0.7098039215686275, 1.0]
    world = args[0]
    ax = args[1]
    if len(args) > 2:
        x = args[2]
        J = args[3]
        u_opt_ind = args[4]
        alg = args[5]
        maxJ = args[6]
        localMaxJ = np.max(J)
        minJ = np.min(J)
        cMap = plt.cm.jet(np.arange(plt.cm.jet.N))
        cMap = [c * 0.5 for c in cMap]
        cMap = ListedColormap(cMap)
        cNorm = colors.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cMap)
        for i in range(0, len(J)):
            if world == Constants.UPPER:
                xCorner, yCorner = getUpperTriangle(x[(i, 0)], x[(i, 1)])
            else:
                xCorner, yCorner = getLowerTriangle(x[(i, 0)], x[(i, 1)])
            vertices = np.array([xCorner, yCorner]).transpose()
            colorVal = scalarMap.to_rgba(J[i] / maxJ)
            triangle = plt.Polygon(vertices, color=colorVal, linewidth=0)
            ax.add_patch(triangle)

        ax.set_title(alg)
        for i in range(0, len(J)):
            x_i = x[i, :]
            if Constants.PLOT_POLICY:
                center = [
                x_i[0] + 1, x_i[1] * 2 + 1]
                if u_opt_ind[i] == Constants.SOUTH:
                    u_i = np.array([0, -1])
                else:
                    if u_opt_ind[i] == Constants.NORTH:
                        u_i = np.array([0, 1])
                    else:
                        if u_opt_ind[i] == Constants.WEST:
                            u_i = np.array([-1, 0])
                        else:
                            if u_opt_ind[i] == Constants.EAST:
                                u_i = np.array([1, 0])
                            else:
                                if u_opt_ind[i] == Constants.STAY:
                                    u_i = np.array([0, 0])
                startPt = np.copy(center)
                endPt = center + 0.4 * u_i
                arrow(startPt, endPt, ax)
            if Constants.PLOT_COST:
                if world == Constants.UPPER:
                    if x_i[0] % 2 == 0:
                        if x_i[1] % 2 == 0:
                            ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 0.2), (round(J[i], 1)), fontsize=8, color='black')
                        else:
                            ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 1.5), (round(J[i], 1)), fontsize=8, color='black')
                    else:
                        if x_i[1] % 2 == 0:
                            ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 1.5), (round(J[i], 1)), fontsize=8, color='black')
                        else:
                            ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 0.2), (round(J[i], 1)), fontsize=8, color='black')
                        if x_i[0] % 2 == 0:
                            if x_i[1] % 2 == 0:
                                ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 1.5), (round(J[i], 1)), fontsize=8, color='black')
                            else:
                                ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 0.2), (round(J[i], 1)), fontsize=8, color='black')
                        else:
                            if x_i[1] % 2 == 0:
                                ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 0.2), (round(J[i], 1)), fontsize=8, color='black')
                            else:
                                ax.text((x_i[0] + 0.5), (x_i[1] * 2 + 1.5), (round(J[i], 1)), fontsize=8, color='black')
                            new_cmap = LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'.format(name=(cMap.name), a=(minJ / maxJ), b=(localMaxJ / maxJ)), cMap(np.linspace(minJ / maxJ, localMaxJ / maxJ, cMap.N)))
                            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=new_cmap)
                            cbar = plt.colorbar(scalarMap, ax=ax, ticks=[0, 1])
                            cbar.ax.set_yticklabels([round(minJ, 2), round(localMaxJ, 2)])
    else:
        free_cell = np.where(map_world == Constants.FREE)
        for i in range(len(free_cell[0])):
            if world == Constants.UPPER:
                xCorner, yCorner = getUpperTriangle(free_cell[0][i], free_cell[1][i])
            else:
                xCorner, yCorner = getLowerTriangle(free_cell[0][i], free_cell[1][i])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=obstacleColor, fill=None)
            ax.add_patch(triangle)

        obstacle = np.where(map_world == Constants.OBSTACLE)
        for i in range(len(obstacle[0])):
            if world == Constants.UPPER:
                xCorner, yCorner = getUpperTriangle(obstacle[0][i], obstacle[1][i])
            else:
                xCorner, yCorner = getLowerTriangle(obstacle[0][i], obstacle[1][i])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=obstacleColor)
            ax.add_patch(triangle)
    
        portals = np.where(map_world == Constants.PORTAL)
        for i in range(len(portals[0])):
            if world == Constants.UPPER:
                xCorner, yCorner = getUpperTriangle(portals[0][i], portals[1][i])
            else:
                xCorner, yCorner = getLowerTriangle(portals[0][i], portals[1][i])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=portalColor)
            ax.add_patch(triangle)

        aliens = np.where(map_world == Constants.ALIEN)
        for i in range(len(aliens[0])):
            if world == Constants.LOWER:
                xCorner, yCorner = getLowerTriangle(aliens[0][i], aliens[1][i])
                vertices = np.array([xCorner, yCorner]).transpose()
                triangle = plt.Polygon(vertices, color=alienColor)
                ax.add_patch(triangle)
            
        mine = np.where(map_world == Constants.MINE)
        if world == Constants.LOWER:
            xCorner, yCorner = getLowerTriangle(mine[0][0], mine[1][0])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=mineColor)
            ax.add_patch(triangle)

        lab = np.where(map_world == Constants.LAB)
        if world == Constants.UPPER:
            xCorner, yCorner = getUpperTriangle(lab[0][0], lab[1][0])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=labColor)
            ax.add_patch(triangle)

        base = np.where(map_world == Constants.BASE)
        if world == Constants.UPPER:
            xCorner, yCorner = getUpperTriangle(base[0][0], base[1][0])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=baseColor)
            ax.add_patch(triangle)
        
        if not Constants.PLOT_COST or len(args) <= 2:
            xPortal, yPortal = np.where(map_world == Constants.PORTAL)
            for i in range(0, len(xPortal)):
                if world == Constants.UPPER and xPortal[i] % 2 == 0 and yPortal[i] % 2 == 0 or xPortal[i] % 2 != 0 and yPortal[i] % 2 != 0:
                    ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 0.2), 'P', fontsize=8)
                else:
                    ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 1.1), 'P', fontsize=8)
                
                if not (xPortal[i] % 2 == 0 and yPortal[i] % 2 == 0):
                    if not xPortal[i] % 2 != 0 or yPortal[i] % 2 != 0:
                        ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 1.1), 'P', fontsize=8)
                    else:
                        ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 0.2), 'P', fontsize=8)

                if world == Constants.UPPER:
                    xBase, yBase = np.where(map_world == Constants.BASE)
                    xLab, yLab = np.where(map_world == Constants.LAB)
                    if xBase % 2 == 0 and yBase % 2 == 0 or xBase % 2 != 0 and yBase % 2 != 0:
                        ax.text((xBase[0] + 0.6), (yBase[0] * 2 + 0.2), 'B', fontsize=8)
                    else:
                        ax.text((xBase[0] + 0.6), (yBase[0] * 2 + 1.1), 'B', fontsize=8)
                    if xLab % 2 == 0 and yLab % 2 == 0 or xLab % 2 != 0 and yLab % 2 != 0:
                        ax.text((xLab[0] + 0.6), (yLab[0] * 2 + 0.2), 'L', fontsize=8)
                    else:
                        ax.text((xLab[0] + 0.6), (yLab[0] * 2 + 1.1), 'L', fontsize=8)
                else:
                    xMine, yMine = np.where(map_world == Constants.MINE)
                    xAlien, yAlien = np.where(map_world == Constants.ALIEN)
                    if xMine % 2 == 0 and yMine % 2 == 0 or xMine % 2 != 0 and yMine % 2 != 0:
                        ax.text((xMine[0] + 0.6), (yMine[0] * 2 + 1.1), 'M', fontsize=8)
                    else:
                        ax.text((xMine[0] + 0.6), (yMine[0] * 2 + 0.2), 'M', fontsize=8)

                    for i in range(0, len(xAlien)):
                        if xAlien[i] % 2 == 0:
                            #if not yAlien[i] % 2 == 0:
                            #    if not xAlien[i] % 2 != 0 or yAlien[i] % 2 != 0:
                            #        ax.text((xAlien[i] + 0.6), (yAlien[i] * 2 + 1.1), 'A', fontsize=8)
                            #else:
                            ax.text((xAlien[i] + 0.6), (yAlien[i] * 2 + 0.2), 'A', fontsize=8)

    mapSize = map_world.shape
    if world == Constants.UPPER:
        ax.plot((np.array([*range(0, mapSize[1] + 1)]) % 2), (np.array([*range(0, 2 * mapSize[1] + 1, 2)])),
          'k', linewidth=2)
        if mapSize[0] % 2 == 0:
            ax.plot([0, mapSize[0]], [0, 0], c='k', linewidth=2)
            ax.plot((np.array([*range(0, mapSize[1] + 1)]) % 2 + mapSize[0]), (np.array([*range(0, 2 * mapSize[1] + 1, 2)])),
              'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0], 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0] + 1, 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
        else:
            ax.plot([0, mapSize[0] + 1], [0, 0], 'k', linewidth=2)
            ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2) + mapSize[0] + 1), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
              'k',
              linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0] + 1, 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0], 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
    else:
        ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2) + 1), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
          'k', linewidth=2)
        if mapSize[0] % 2 == 0:
            ax.plot([1, mapSize[0] + 1], [0, 0], 'k', linewidth=2)
            ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2) + mapSize[0] + 1), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
              'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0] + 1, 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0], 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
        else:
            ax.plot([1, mapSize[0]], [0, 0], 'k', linewidth=2)
            ax.plot((np.array([*range(0, mapSize[1] + 1)]) % 2 + mapSize[0]), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
              'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0], 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0] + 1, 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
    ax.set_xticks(range(1, mapSize[0] + 1))
    ax.set_xticklabels(range(1, mapSize[0] + 1))
    ax.set_yticks(range(1, 2 * mapSize[1] + 1, 2))
    ax.set_yticklabels(range(1, mapSize[1] + 1))
    ax.tick_params(axis='both', which='major', labelsize=4)
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

def getTriangleDirection(state):
    return ((state[0] + state[1] + state[3]) % 2) == 0

def CustomMap(map_world, *args):
    """
      Plot a map, the costs for each cell and the control action in
      each cell.

        Input arguments:
          map:
              A (M x N)-matrix describing the terrain of the estate map.
              Positive values indicate cells that are inaccessible (e.g.
              trees, bushes or the mansion) and negative values indicate
              ponds or pools.

          *args (optional):
                Input argument list:
                1:      A (K x 2)-matrix 'stateSpace', where each row
                      represents an element of the state space.
                2:  A (K x 1 )-matrix 'J' containing the optimal cost-to-go
                        for each element of the state space.
                3:  A (K x 1 )-matrix containing the index of the optimal
                        control input for each element of the state space.
                  4:  Title
    """
    obstacleColor = [
     0.39215686274509803, 0.39215686274509803, 0.39215686274509803]
    portalColor = [
     0.19607843137254902, 0.803921568627451, 0.19607843137254902]
    alienColor = [
     0.9490196078431372, 0.23921568627450981, 0.0]
    mineColor = [
     1.0, 0.9019607843137255, 0.0]
    labColor = [
     0.8313725490196079, 0.16470588235294117, 1.0]
    baseColor = [
     0.4392156862745098, 0.7098039215686275, 1.0]
    
    world = args[0]
    ax = args[1]
    cellState = args[2]

    

    free_cell = np.where(map_world == Constants.FREE)
    for i in range(len(free_cell[0])):
        if world == Constants.UPPER:
            xCorner, yCorner = getUpperTriangle(free_cell[0][i], free_cell[1][i])
        else:
            xCorner, yCorner = getLowerTriangle(free_cell[0][i], free_cell[1][i])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color="black", fill=None)
        ax.add_patch(triangle)

    obstacle = np.where(map_world == Constants.OBSTACLE)
    for i in range(len(obstacle[0])):
        if world == Constants.UPPER:
            xCorner, yCorner = getUpperTriangle(obstacle[0][i], obstacle[1][i])
        else:
            xCorner, yCorner = getLowerTriangle(obstacle[0][i], obstacle[1][i])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=obstacleColor)
        ax.add_patch(triangle)

    portals = np.where(map_world == Constants.PORTAL)
    for i in range(len(portals[0])):
        if world == Constants.UPPER:
            xCorner, yCorner = getUpperTriangle(portals[0][i], portals[1][i])
        else:
            xCorner, yCorner = getLowerTriangle(portals[0][i], portals[1][i])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=portalColor)
        ax.add_patch(triangle)
    aliens = np.where(map_world == Constants.ALIEN)
    for i in range(len(aliens[0])):
        if world == Constants.LOWER:
            xCorner, yCorner = getLowerTriangle(aliens[0][i], aliens[1][i])
            vertices = np.array([xCorner, yCorner]).transpose()
            triangle = plt.Polygon(vertices, color=alienColor)
            ax.add_patch(triangle)
            
            if aliens[0][i]+1 < Constants.M:
                xCorner, yCorner = getLowerTriangle(aliens[0][i]+1, aliens[1][i])
                vertices = np.array([xCorner, yCorner]).transpose()
                triangle = plt.Polygon(vertices, color=alienColor, alpha=0.5)
                ax.add_patch(triangle)

            if aliens[0][i]-1 >= 0:
                xCorner, yCorner = getLowerTriangle(aliens[0][i]-1, aliens[1][i])
                vertices = np.array([xCorner, yCorner]).transpose()
                triangle = plt.Polygon(vertices, color=alienColor, alpha=0.5)
                ax.add_patch(triangle)

            if getTriangleDirection([aliens[0][i], aliens[1][i], 0,0]) == False:
                if aliens[1][i]-1 >= 0:
                    xCorner, yCorner = getLowerTriangle(aliens[0][i], aliens[1][i]-1)
                    vertices = np.array([xCorner, yCorner]).transpose()
                    triangle = plt.Polygon(vertices, color=alienColor, alpha=0.5)
                    ax.add_patch(triangle)
            else:
                if aliens[1][i]+1 < Constants.N:
                    xCorner, yCorner = getLowerTriangle(aliens[0][i], aliens[1][i]+1)
                    vertices = np.array([xCorner, yCorner]).transpose()
                    triangle = plt.Polygon(vertices, color=alienColor, alpha=0.5)
                    ax.add_patch(triangle)

        
    mine = np.where(map_world == Constants.MINE)
    if world == Constants.LOWER:
        xCorner, yCorner = getLowerTriangle(mine[0][0], mine[1][0])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=mineColor)
        ax.add_patch(triangle)
    lab = np.where(map_world == Constants.LAB)
    if world == Constants.UPPER:
        xCorner, yCorner = getUpperTriangle(lab[0][0], lab[1][0])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=labColor)
        ax.add_patch(triangle)
    base = np.where(map_world == Constants.BASE)
    if world == Constants.UPPER:
        xCorner, yCorner = getUpperTriangle(base[0][0], base[1][0])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color=baseColor)
        ax.add_patch(triangle)

    if cellState[3] == world:
        if world == Constants.UPPER:
            xCorner, yCorner = getUpperTriangle(cellState[0], cellState[1])
        else:
            xCorner, yCorner = getLowerTriangle(cellState[0], cellState[1])
        vertices = np.array([xCorner, yCorner]).transpose()
        triangle = plt.Polygon(vertices, color="lime")
        ax.add_patch(triangle)

    xPortal, yPortal = np.where(map_world == Constants.PORTAL)
    for i in range(0, len(xPortal)):
        if world == Constants.UPPER and xPortal[i] % 2 == 0 and yPortal[i] % 2 == 0 or xPortal[i] % 2 != 0 and yPortal[i] % 2 != 0:
            ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 0.2), 'P', fontsize=8)
        else:
            ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 1.1), 'P', fontsize=8)
        
        if not (xPortal[i] % 2 == 0 and yPortal[i] % 2 == 0):
            if not xPortal[i] % 2 != 0 or yPortal[i] % 2 != 0:
                ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 1.1), 'P', fontsize=8)
            else:
                ax.text((xPortal[i] + 0.7), (yPortal[i] * 2 + 0.2), 'P', fontsize=8)
        if world == Constants.UPPER:
            xBase, yBase = np.where(map_world == Constants.BASE)
            xLab, yLab = np.where(map_world == Constants.LAB)
            if xBase % 2 == 0 and yBase % 2 == 0 or xBase % 2 != 0 and yBase % 2 != 0:
                ax.text((xBase[0] + 0.6), (yBase[0] * 2 + 0.2), 'B', fontsize=8)
            else:
                ax.text((xBase[0] + 0.6), (yBase[0] * 2 + 1.1), 'B', fontsize=8)
            if xLab % 2 == 0 and yLab % 2 == 0 or xLab % 2 != 0 and yLab % 2 != 0:
                ax.text((xLab[0] + 0.6), (yLab[0] * 2 + 0.2), 'L', fontsize=8)
            else:
                ax.text((xLab[0] + 0.6), (yLab[0] * 2 + 1.1), 'L', fontsize=8)
        else:
            xMine, yMine = np.where(map_world == Constants.MINE)
            xAlien, yAlien = np.where(map_world == Constants.ALIEN)
            if xMine % 2 == 0 and yMine % 2 == 0 or xMine % 2 != 0 and yMine % 2 != 0:
                ax.text((xMine[0] + 0.6), (yMine[0] * 2 + 1.1), 'M', fontsize=8)
            else:
                ax.text((xMine[0] + 0.6), (yMine[0] * 2 + 0.2), 'M', fontsize=8)
            for i in range(0, len(xAlien)):
                if xAlien[i] % 2 == 0:
                    #if not yAlien[i] % 2 == 0:
                    #    if not xAlien[i] % 2 != 0 or yAlien[i] % 2 != 0:
                    #        ax.text((xAlien[i] + 0.6), (yAlien[i] * 2 + 1.1), 'A', fontsize=8)
                    #else:
                    ax.text((xAlien[i] + 0.6), (yAlien[i] * 2 + 0.2), 'A', fontsize=8)

    mapSize = map_world.shape
    if world == Constants.UPPER:
        ax.plot((np.array([*range(0, mapSize[1] + 1)]) % 2), (np.array([*range(0, 2 * mapSize[1] + 1, 2)])),
          'k', linewidth=2)
        if mapSize[0] % 2 == 0:
            ax.plot([0, mapSize[0]], [0, 0], c='k', linewidth=2)
            ax.plot((np.array([*range(0, mapSize[1] + 1)]) % 2 + mapSize[0]), (np.array([*range(0, 2 * mapSize[1] + 1, 2)])),
              'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0], 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0] + 1, 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
        else:
            ax.plot([0, mapSize[0] + 1], [0, 0], 'k', linewidth=2)
            ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2) + mapSize[0] + 1), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
              'k',
              linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0] + 1, 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0], 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
    else:
        ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2) + 1), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
          'k', linewidth=2)
        if mapSize[0] % 2 == 0:
            ax.plot([1, mapSize[0] + 1], [0, 0], 'k', linewidth=2)
            ax.plot((-(np.array([*range(0, mapSize[1] + 1)]) % 2) + mapSize[0] + 1), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
              'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0] + 1, 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0], 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
        else:
            ax.plot([1, mapSize[0]], [0, 0], 'k', linewidth=2)
            ax.plot((np.array([*range(0, mapSize[1] + 1)]) % 2 + mapSize[0]), (np.array([*range(0, 2 * mapSize[1] + 2, 2)])),
              'k', linewidth=2)
            if mapSize[1] % 2 == 0:
                ax.plot([mapSize[0], 1], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
            else:
                ax.plot([mapSize[0] + 1, 0], [2 * mapSize[1], 2 * mapSize[1]], 'k', linewidth=2)
    ax.set_xticks(range(1, mapSize[0] + 1))
    ax.set_xticklabels(range(1, mapSize[0] + 1))
    ax.set_yticks(range(1, 2 * mapSize[1] + 1, 2))
    ax.set_yticklabels(range(1, mapSize[1] + 1))
    ax.tick_params(axis='both', which='major', labelsize=4)
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

def getUpperTriangle(column, row):
    if column % 2 == row % 2:
        xCorner = [
         column, column + 1, column + 2]
        yCorner = [2 * row, 2 * row + 2, 2 * row]
    else:
        xCorner = [
         column, column + 1, column + 2]
        yCorner = [2 * row + 2, 2 * row, 2 * row + 2]
    return [xCorner, yCorner]


def getLowerTriangle(column, row):
    if column % 2 != row % 2:
        xCorner = [
         column, column + 1, column + 2]
        yCorner = [2 * row, 2 * row + 2, 2 * row]
    else:
        xCorner = [
         column, column + 1, column + 2]
        yCorner = [2 * row + 2, 2 * row, 2 * row + 2]
    return [xCorner, yCorner]


def arrow(startPt, endPt, ax):
    color = [
     0.39215686274509803, 0.39215686274509803, 0.39215686274509803]
    if endPt[1] == startPt[1] and endPt[0] == startPt[0]:
        ax.plot((endPt[0]), (endPt[1]), marker='o', markersize=2, markeredgecolor=color, markerfacecolor=color)
    else:
        alpha = np.arctan2(endPt[1] - startPt[1], endPt[0] - startPt[0])
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        arrowHead = np.array([[0, 0],
         [
          -0.1, 0.1],
         [
          0, 0],
         [
          -0.1, -0.1]])
        for i in range(0, arrowHead.shape[0]):
            arrowHead[i, :] = np.transpose(np.matmul(R, np.transpose(arrowHead[i, :])))
            arrowHead[i, :] = arrowHead[i, :] + endPt
        else:
            arrowLines = np.array([[startPt[0], startPt[1]],
             [
              endPt[0], endPt[1]]])
            ax.plot((arrowLines[:, 0]), (arrowLines[:, 1]), color=color, linewidth=1)
            ax.plot((arrowHead[:, 0]), (arrowHead[:, 1]), color=color, linewidth=1)
# okay decompiling MakePlots.pyc
