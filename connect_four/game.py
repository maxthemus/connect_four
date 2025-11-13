import numpy as np
import torch


def create_board(rows=6, cols=7):
    """
    Create an empty Connect 4 board.
    
    Args:
        rows: Number of rows (default 6)
        cols: Number of columns (default 7)
    
    Returns:
        2D numpy array of shape (rows, cols) filled with zeros
    """
    return np.zeros((rows, cols), dtype=np.int32)


def board_to_tensor(board):
    """
    Convert board to a PyTorch tensor.
    
    Args:
        board: 2D numpy array representing the board
    
    Returns:
        PyTorch tensor of shape (rows, cols)
    """
    return torch.from_numpy(board).float()


def apply_move(board, col, player):
    """
    Apply a move to the board by dropping a piece in the specified column.
    
    Args:
        board: 2D numpy array representing the board
        col: Column index (0-indexed) where to drop the piece
        player: Player number (1 for player 1, -1 for player 2)
    
    Returns:
        tuple: (success, row) where success is True if move was valid, 
               and row is the row where the piece landed (-1 if invalid)
    """
    rows, cols = board.shape
    
    # Check if column is valid
    if col < 0 or col >= cols:
        return False, -1
    
    # Check if column is full
    if board[0, col] != 0:
        return False, -1
    
    # Find the lowest empty row in the column
    for row in range(rows - 1, -1, -1):
        if board[row, col] == 0:
            board[row, col] = player
            return True, row
    
    return False, -1


def print_board(board):
    """
    Print the board in a readable format.
    
    Args:
        board: 2D numpy array representing the board
    """
    rows, cols = board.shape
    
    # Print column numbers
    print("\n " + " ".join(str(i) for i in range(cols)))
    print("+" + "-" * (cols * 2 - 1) + "+")
    
    # Print each row
    for row in range(rows):
        row_str = "|"
        for col in range(cols):
            if board[row, col] == 0:
                row_str += " "
            elif board[row, col] == 1:
                row_str += "X"
            elif board[row, col] == -1:
                row_str += "O"
            
            if col < cols - 1:
                row_str += " "
        row_str += "|"
        print(row_str)
    
    print("+" + "-" * (cols * 2 - 1) + "+")
