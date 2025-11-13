from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status
import json


class PlayMoveAPITestCase(TestCase):
    """Test cases for the play_move API endpoint."""
    
    def setUp(self):
        """Set up test client."""
        self.client = APIClient()
        self.url = '/game/play/'
        
    def test_empty_board_valid_move(self):
        """Test making a move on an empty board."""
        data = {
            'board': [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ],
            'current_turn': 1
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('board', response.data)
        self.assertIn('move', response.data)
        self.assertIn('row', response.data)
        self.assertIn('success', response.data)
        self.assertTrue(response.data['success'])
        
        # Check that a move was made (column should be 0-6)
        self.assertGreaterEqual(response.data['move'], 0)
        self.assertLessEqual(response.data['move'], 6)
        
        # Check that the board was updated
        board = response.data['board']
        self.assertEqual(len(board), 6)
        self.assertEqual(len(board[0]), 7)
        
    def test_partially_filled_board(self):
        """Test making a move on a partially filled board."""
        data = {
            'board': [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, -1, 0, 0, 0]
            ],
            'current_turn': 1
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        
    def test_player_minus_one(self):
        """Test making a move as player -1."""
        data = {
            'board': [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0]
            ],
            'current_turn': -1
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        
    def test_missing_board(self):
        """Test request with missing board."""
        data = {
            'current_turn': 1
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        
    def test_missing_current_turn(self):
        """Test request with missing current_turn."""
        data = {
            'board': [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ]
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        
    def test_invalid_current_turn(self):
        """Test request with invalid current_turn value."""
        data = {
            'board': [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ],
            'current_turn': 2
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        
    def test_invalid_board_shape(self):
        """Test request with invalid board shape."""
        data = {
            'board': [
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            'current_turn': 1
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        
    def test_full_board(self):
        """Test request with a full board (no valid moves)."""
        data = {
            'board': [
                [1, -1, 1, -1, 1, -1, 1],
                [1, -1, 1, -1, 1, -1, 1],
                [1, -1, 1, -1, 1, -1, 1],
                [1, -1, 1, -1, 1, -1, 1],
                [1, -1, 1, -1, 1, -1, 1],
                [1, -1, 1, -1, 1, -1, 1]
            ],
            'current_turn': 1
        }
        
        response = self.client.post(self.url, data, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
        
    def test_empty_request_body(self):
        """Test request with empty body."""
        response = self.client.post(self.url, {}, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
