import torch
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from pathlib import Path
from django.conf import settings

from connect_four.model import ConnectFourModel
from connect_four.game import apply_move, board_to_tensor

# Global variable to store the loaded model
_model = None
_model_path = settings.BASE_DIR.parent / 'connect_four_model.pth'


def get_model():
    """Load and return the Connect Four model (singleton pattern)."""
    global _model
    if _model is None:
        _model = ConnectFourModel()
        if _model_path.exists():
            _model.load_state_dict(torch.load(_model_path, map_location=torch.device('cpu'), weights_only=True))
            _model.eval()
        else:
            raise FileNotFoundError(f"Model file not found at {_model_path}")
    return _model


def get_valid_moves(board):
    """Get list of valid column indices where a piece can be placed."""
    cols = board.shape[1]
    return [col for col in range(cols) if board[0, col] == 0]


def select_move(model, board, player):
    """
    Select the best move using the trained model.
    
    Args:
        model: The neural network model
        board: Current board state (numpy array)
        player: Current player (1 or -1)
    
    Returns:
        Selected column index
    """
    valid_moves = get_valid_moves(board)
    
    if not valid_moves:
        return None
    
    model.eval()
    with torch.no_grad():
        # Prepare board for model (from player's perspective)
        board_input = board * player  # Flip perspective for player -1
        tensor_board = board_to_tensor(board_input).unsqueeze(0).unsqueeze(0)
        
        q_values = model(tensor_board).squeeze()
        
        # Mask invalid moves
        mask = torch.full((7,), float('-inf'))
        mask[valid_moves] = 0
        q_values = q_values + mask
        
        return q_values.argmax().item()


@api_view(['POST'])
def play_move(request):
    """
    API endpoint to play a move using the trained model.
    
    Expected POST data:
    {
        "board": [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],
        "current_turn": 1  # or -1
    }
    
    Returns:
    {
        "board": updated board,
        "move": column where piece was placed,
        "success": true/false
    }
    """
    try:
        # Validate request data
        if not request.data:
            return Response(
                {"error": "Request body is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        board_data = request.data.get('board')
        current_turn = request.data.get('current_turn')
        
        if board_data is None:
            return Response(
                {"error": "Missing 'board' in request data"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if current_turn is None:
            return Response(
                {"error": "Missing 'current_turn' in request data"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate current_turn
        if current_turn not in [1, -1]:
            return Response(
                {"error": "'current_turn' must be 1 or -1"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Convert board to numpy array
        try:
            board = np.array(board_data, dtype=np.int32)
        except (ValueError, TypeError):
            return Response(
                {"error": "Invalid board format"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate board shape
        if board.shape != (6, 7):
            return Response(
                {"error": f"Board must be 6x7, got {board.shape}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Load model and select move
        model = get_model()
        col = select_move(model, board, current_turn)
        
        if col is None:
            return Response(
                {"error": "No valid moves available (board is full)"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Apply the move
        success, row = apply_move(board, col, current_turn)
        
        if not success:
            return Response(
                {"error": f"Failed to apply move to column {col}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Return updated board
        return Response({
            "board": board.tolist(),
            "move": col,
            "row": row,
            "success": True
        }, status=status.HTTP_200_OK)
        
    except FileNotFoundError:
        return Response(
            {"error": "Model not found"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    except Exception:
        return Response(
            {"error": "Internal server error"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
