"""
Contains layouts for different grid warehouses
"""

# from models import GridWarehouse, NavigableTileCell, ShelvingCell, Direction
# import utils
import typing


def init_grid(dimensions):  # type: (typing.Tuple[int, int]) -> [[None]]
    """
    Creates grid in (width, height) dimensions.
    """
    width, height = dimensions
    grid = []
    for col_num in range(width):
        row = [None for row_num in range(height)]
        grid.append(row)
    return grid


def manhattan_distance(first, second):  # type: (typing.Tuple[int, int], typing.Tuple[int, int]) -> int
    first_x, first_y = first
    second_x, second_y = second
    return abs(first_x - second_x) + abs(first_y - second_y)

import copy

import typing
import networkx as nx
from tsp import tsp_circuit
from utils import manhattan_distance


PositionType = typing.Tuple[int, int]


class GridWarehouse(object):
    dimensions = None  # type: PositionType
    grid = None  # type: [[GridWarehouseCell]]
    graph = None  # type: nx.Graph

    def __init__(self, dimensions, grid):  # type: (PositionType, [[GridWarehouseCell]]) -> None
        self.dimensions = dimensions
        self.grid = grid
        self.graph = self._construct_graph()
    
    def find_path(self, from_node, to_node):  # type: (PositionType, PositionType) -> [PositionType]
        return nx.shortest_path(self.graph, from_node, to_node)

    def find_pick_path(self, from_node, intermediate_nodes):
        # type: (PositionType, PositionType, [PositionType]) -> [PositionType]

        # 1. Create a graph on the nodes in this order (removes all nodes from graph not in this path)
        nodes = [from_node] + intermediate_nodes
        G_tsp = nx.Graph()
        for node in nodes:
            G_tsp.add_node(node)
        # Add edges into the graph corresponding to the manhattan distance between two points
        for idx_i, i in enumerate(nodes):
            for idx_j, j in enumerate(nodes):
                if i is not j:
                    G_tsp.add_edge(i, j, weight=self.distance(i, j))

        # 2. Compute the TSP
        tsp_tour = tsp_circuit(G_tsp, from_node)
        if tsp_tour is None:
            raise ValueError("Couldn't find picking path.")

        # 3. Now, re-introduce the intermediate nodes to compute the real path in the warehouse
        final_path = [tsp_tour[0]]
        for next_cell in tsp_tour[1:]:
            # Get the last cell of the path
            current_cell = final_path[-1]
            # Find the path to the next node (remove the first element with is equal to current_node)
            path_to_next_cell = nx.shortest_path(self.graph, current_cell, next_cell)[1:]
            # Add in all cells from the path to the next cell
            for path_cell in path_to_next_cell:
                final_path.append(path_cell)

        return final_path


    def distance(self, from_cell, to_cell):
        """
        Admissible distance heuristic for this warehouse
        """
        return manhattan_distance(from_cell, to_cell)

    def __str__(self):
        row_strings = []
        for column in self.grid:
            column_strings = []
            for cell in column:
                if isinstance(cell, NavigableTileCell):
                    column_strings.append("+")
                elif isinstance(cell, ShelvingCell):
                    column_strings.append("x")
                else:
                    raise TypeError("Unknown cell type: %s" % type(cell))
            row_strings.append("\t".join(column_strings) + "\n")
        return \
            "Grid Warehouse (%d height x %d width)\n" % self.dimensions + \
            "".join(row_strings)

    def _construct_graph(self):  # type: () -> nx.Graph
        graph = nx.Graph()

        # Add all navigable tiles as nodes
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if not isinstance(cell, NavigableTileCell):
                    continue
                graph.add_node((i, j))

        # Add all edges
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if not isinstance(cell, NavigableTileCell):
                    continue

                cell_coordinates = (i, j)

                neighbor_coordinates = self._get_neighboring_navigation_cells(cell_coordinates)

                for neighbor_cell_coordinate in neighbor_coordinates:  # type: typing.Tuple(int, int)
                    graph.add_edge(cell_coordinates, neighbor_cell_coordinate, weight=1)

        return graph

    def _get_neighboring_navigation_cells(self, origin_cell_coordinates):  # type: () -> [typing.Tuple(int, int)]
        offsets = [
            (-1, 0),
            (0, +1),
            (+1, 0),
            (0, -1),
        ]

        neighbors = []

        for offset_x, offset_y in offsets:
            neighbor_coordinate_to_examine = (origin_cell_coordinates[0] + offset_x, origin_cell_coordinates[1] + offset_y)

            if neighbor_coordinate_to_examine[0] < 0 or neighbor_coordinate_to_examine[0] >= self.dimensions[0]:
                continue

            if neighbor_coordinate_to_examine[1] < 0 or neighbor_coordinate_to_examine[1] >= self.dimensions[1]:
                continue

            neighbor_cell = self.grid[neighbor_coordinate_to_examine[0]][neighbor_coordinate_to_examine[1]]

            if not isinstance(neighbor_cell, NavigableTileCell):
                continue

            neighbors.append(neighbor_coordinate_to_examine)

        return neighbors


class GridWarehouseCell(object):
    pass


class Direction(object):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class ShelvingCell(GridWarehouseCell):
    items = None  # type: [Item]
    direction = None  # type: Direction

    def __init__(self, direction, items=None):
        self.direction = direction
        self.items = items or []


class NavigableTileCell(GridWarehouseCell):
    pass


class Item(object):
    name = None  # type: str
    rfid = None  # type: str

    def __init__(self, name, rfid):
        self.name = name
        self.rfid = rfid


def get_simple_warehouse():
    dimensions = (4, 4)

    grid = utils.init_grid(dimensions)

    # Fill out first column, bottom to top
    grid[0][0] = NavigableTileCell()
    grid[0][1] = NavigableTileCell()
    grid[0][2] = NavigableTileCell()
    grid[0][3] = NavigableTileCell()

    # Fill out second column, bottom to top
    grid[1][0] = NavigableTileCell()
    grid[1][1] = ShelvingCell(Direction.SOUTH)
    grid[1][2] = ShelvingCell(Direction.NORTH)
    grid[1][3] = NavigableTileCell()

    # Fill out third column, bottom to top
    grid[2][0] = NavigableTileCell()
    grid[2][1] = ShelvingCell(Direction.SOUTH)
    grid[2][2] = ShelvingCell(Direction.NORTH)
    grid[2][3] = NavigableTileCell()

    # Fill out fourth column, bottom to top
    grid[3][0] = NavigableTileCell()
    grid[3][1] = NavigableTileCell()
    grid[3][2] = NavigableTileCell()
    grid[3][3] = NavigableTileCell()

    warehouse = GridWarehouse(dimensions, grid)

    return warehouse


def get_larger_warehouse():
    dimensions = (8, 4)

    grid = utils.init_grid(dimensions)

    for row_num in range(4):
        grid[0][row_num] = NavigableTileCell()

    for col_num in range(1, 7):
        grid[col_num][0] = NavigableTileCell()
        grid[col_num][1] = ShelvingCell(Direction.NORTH)
        grid[col_num][2] = ShelvingCell(Direction.SOUTH)
        grid[col_num][3] = NavigableTileCell()

    for row_num in range(4):
        grid[7][row_num] = NavigableTileCell()

    warehouse = GridWarehouse(dimensions, grid)

    return warehouse


def get_georgia_tech_library_warehouse():
    larger_warehouse = get_larger_warehouse()
    larger_warehouse_grid = larger_warehouse.grid

    dimensions = (8, 16)
    grid = utils.init_grid(dimensions)

    for col_num in range(8):
        for row_num in range(16):
            grid[col_num][row_num] = copy.copy(larger_warehouse_grid[col_num][row_num % 4])

    return GridWarehouse(dimensions, grid)
